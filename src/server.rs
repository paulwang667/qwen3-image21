//! OpenAI-compatible HTTP image API server with multi-GPU worker pool.
//!
//! Endpoints:
//!   POST /v1/images/generations  — text-to-image (JSON body)
//!   POST /v1/images/edits        — image-conditioned generation (multipart)
//!   GET  /health                 — health check
//!
//! Each GPU gets a dedicated OS-thread worker that serializes requests. N GPUs
//! → N concurrent inferences. Models are loaded on demand per request (a few
//! seconds of overhead, same as the CLI).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use base64::{engine::general_purpose, Engine};
use serde::{Deserialize, Serialize};

use candle_core::Device;

use crate::service::{self, ConditionImageInput, EngineConfig, GenRequest, ModelPaths};

// ── OpenAI-compatible request/response types ────────────────────────────

#[derive(Deserialize)]
struct GenerationsBody {
    prompt: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    n: Option<usize>,
    #[serde(default)]
    size: Option<String>,
    // Extension params (not OpenAI standard, backward-compatible).
    #[serde(default)]
    steps: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    negative_prompt: Option<String>,
    #[serde(default)]
    true_cfg_scale: Option<f32>,
    #[serde(default)]
    output_resolution: Option<usize>,
}

#[derive(Serialize)]
struct ImageItem {
    b64_json: String,
}

#[derive(Serialize)]
struct GenerationsResponse {
    created: u64,
    data: Vec<ImageItem>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
}

fn err_resp(status: StatusCode, msg: impl Into<String>) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: msg.into(),
            },
        }),
    )
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Parse OpenAI `size` field ("1024x1024") into (width, height).
/// Returns None for "auto" or unparseable values (caller uses defaults).
fn parse_size(s: Option<&str>) -> Option<(usize, usize)> {
    let s = s?;
    if s == "auto" {
        return None;
    }
    let (w_s, h_s) = s.split_once('x')?;
    let w: usize = w_s.parse().ok()?;
    let h: usize = h_s.parse().ok()?;
    if w % 16 != 0 || h % 16 != 0 {
        return None;
    }
    Some((w, h))
}

// ── Worker pool ──────────────────────────────────────────────────────────

struct WorkerJob {
    req: GenRequest,
    reply: tokio::sync::oneshot::Sender<anyhow::Result<service::GenResult>>,
}

pub struct WorkerPool {
    senders: Vec<tokio::sync::mpsc::Sender<WorkerJob>>,
    next: AtomicUsize,
}

impl WorkerPool {
    /// Create a pool with one worker per device ID. Each worker is a dedicated
    /// OS thread that owns its GPU and processes requests serially.
    pub fn new(
        device_ids: &[usize],
        paths: ModelPaths,
        engine: EngineConfig,
    ) -> anyhow::Result<Self> {
        let mut senders = Vec::with_capacity(device_ids.len());
        for &id in device_ids {
            let (tx, rx) = tokio::sync::mpsc::channel::<WorkerJob>(8);
            let device = match Device::cuda_if_available(id)? {
                Device::Cpu => Device::metal_if_available(id)?,
                d => d,
            };
            eprintln!("Worker {}: device {:?}", id, device);
            let paths = paths.clone();
            let engine = engine.clone();
            std::thread::Builder::new()
                .name(format!("worker-{id}"))
                .spawn(move || {
                    let mut rx = rx;
                    while let Some(job) = rx.blocking_recv() {
                        let result =
                            service::generate(&job.req, &paths, &engine, &device);
                        let _ = job.reply.send(result);
                    }
                    eprintln!("Worker {id}: channel closed, exiting");
                })?;
            senders.push(tx);
        }
        Ok(Self {
            senders,
            next: AtomicUsize::new(0),
        })
    }

    /// Dispatch a request to the least-busy worker and await the result.
    async fn generate(&self, req: GenRequest) -> anyhow::Result<service::GenResult> {
        let idx = self.least_busy();
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.senders[idx]
            .send(WorkerJob { req, reply: tx })
            .await
            .map_err(|_| anyhow::anyhow!("worker {idx} closed"))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("worker {idx} dropped reply"))?
    }

    /// Pick the worker with the most channel capacity (least busy).
    fn least_busy(&self) -> usize {
        if self.senders.len() == 1 {
            return 0;
        }
        let mut best = 0;
        let mut best_cap = 0usize;
        for (i, s) in self.senders.iter().enumerate() {
            let cap = s.capacity();
            if cap > best_cap {
                best_cap = cap;
                best = i;
            }
        }
        best
    }

    /// Number of pending requests across all workers.
    pub fn pending(&self) -> usize {
        self.senders.iter().map(|s| 8 - s.capacity()).sum()
    }
}

// ── HTTP handlers ───────────────────────────────────────────────────────

async fn handle_generations(
    State(pool): State<Arc<WorkerPool>>,
    Json(body): Json<GenerationsBody>,
) -> Result<Json<GenerationsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let size = parse_size(body.size.as_deref());
    let n = body.n.unwrap_or(1).clamp(1, 10);
    let base_seed = body.seed;

    let mut items = Vec::with_capacity(n);
    for i in 0..n {
        let seed = base_seed.map(|s| s.wrapping_add(i as u64));
        let req = GenRequest {
            prompt: body.prompt.clone(),
            negative_prompt: body.negative_prompt.clone(),
            true_cfg_scale: body.true_cfg_scale.unwrap_or(1.0),
            width: size.map(|(w, _)| w),
            height: size.map(|(_, h)| h),
            steps: body.steps.unwrap_or(40),
            seed,
            images: Vec::new(),
            output_resolution: body.output_resolution.unwrap_or(1024),
            no_kv_cache: false,
        };
        match pool.generate(req).await {
            Ok(result) => {
                let b64 = general_purpose::STANDARD.encode(&result.png);
                items.push(ImageItem { b64_json: b64 });
            }
            Err(e) => {
                return Err(err_resp(StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")));
            }
        }
    }
    Ok(Json(GenerationsResponse {
        created: now_unix(),
        data: items,
    }))
}

#[derive(Default)]
struct EditForm {
    prompt: Option<String>,
    images: Vec<Vec<u8>>,
    size: Option<String>,
    n: Option<usize>,
    steps: Option<usize>,
    seed: Option<u64>,
    negative_prompt: Option<String>,
    true_cfg_scale: Option<f32>,
    output_resolution: Option<usize>,
}

async fn handle_edits(
    State(pool): State<Arc<WorkerPool>>,
    mut multipart: Multipart,
) -> Result<Json<GenerationsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mut form = EditForm::default();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        let data = match field.bytes().await {
            Ok(d) => d,
            Err(e) => {
                return Err(err_resp(StatusCode::BAD_REQUEST, format!("field read error: {e}")))
            }
        };
        match name.as_str() {
            "image" => form.images.push(data.to_vec()),
            "prompt" => form.prompt = Some(String::from_utf8_lossy(&data).into_owned()),
            "size" => form.size = Some(String::from_utf8_lossy(&data).into_owned()),
            "n" => form.n = std::str::from_utf8(&data).ok().and_then(|s| s.parse().ok()),
            "steps" => form.steps = std::str::from_utf8(&data).ok().and_then(|s| s.parse().ok()),
            "seed" => form.seed = std::str::from_utf8(&data).ok().and_then(|s| s.parse().ok()),
            "negative_prompt" => {
                form.negative_prompt = Some(String::from_utf8_lossy(&data).into_owned())
            }
            "true_cfg_scale" => {
                form.true_cfg_scale = std::str::from_utf8(&data).ok().and_then(|s| s.parse().ok())
            }
            "output_resolution" => {
                form.output_resolution =
                    std::str::from_utf8(&data).ok().and_then(|s| s.parse().ok())
            }
            _ => {} // model, response_format, user, mask, etc. — ignored
        }
    }

    let prompt = match form.prompt {
        Some(p) if !p.is_empty() => p,
        _ => return Err(err_resp(StatusCode::BAD_REQUEST, "prompt is required")),
    };

    let size = parse_size(form.size.as_deref());
    let n = form.n.unwrap_or(1).clamp(1, 10);
    let base_seed = form.seed;

    let mut items = Vec::with_capacity(n);
    for i in 0..n {
        let seed = base_seed.map(|s| s.wrapping_add(i as u64));
        let req = GenRequest {
            prompt: prompt.clone(),
            negative_prompt: form.negative_prompt.clone(),
            true_cfg_scale: form.true_cfg_scale.unwrap_or(1.0),
            width: size.map(|(w, _)| w),
            height: size.map(|(_, h)| h),
            steps: form.steps.unwrap_or(40),
            seed,
            images: form
                .images
                .iter()
                .map(|b| ConditionImageInput::Bytes(b.clone()))
                .collect(),
            output_resolution: form.output_resolution.unwrap_or(1024),
            no_kv_cache: false,
        };
        match pool.generate(req).await {
            Ok(result) => {
                let b64 = general_purpose::STANDARD.encode(&result.png);
                items.push(ImageItem { b64_json: b64 });
            }
            Err(e) => {
                return Err(err_resp(StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")));
            }
        }
    }
    Ok(Json(GenerationsResponse {
        created: now_unix(),
        data: items,
    }))
}

async fn health() -> &'static str {
    "ok"
}

async fn root() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "service": "qwen3-image21",
            "endpoints": [
                "POST /v1/images/generations",
                "POST /v1/images/edits",
                "GET /health"
            ]
        })),
    )
}

// ── Server entry point ──────────────────────────────────────────────────

/// Parse comma-separated device IDs ("0,1,2") → Vec<usize>.
/// Defaults to [0] when None.
pub fn parse_device_ids(s: &Option<String>) -> Vec<usize> {
    match s {
        Some(s) => s
            .split(',')
            .filter_map(|id| id.trim().parse().ok())
            .collect(),
        None => vec![0],
    }
}

/// Start the HTTP server. Blocks until the server is shut down.
pub async fn run(
    addr: &str,
    device_ids: Vec<usize>,
    paths: ModelPaths,
    engine: EngineConfig,
) -> anyhow::Result<()> {
    let pool = Arc::new(WorkerPool::new(&device_ids, paths, engine)?);
    eprintln!(
        "Server starting on {} with {} worker(s) (GPUs: {:?})",
        addr,
        device_ids.len(),
        device_ids
    );

    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/v1/images/generations", post(handle_generations))
        .route("/v1/images/edits", post(handle_edits))
        .with_state(pool);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
