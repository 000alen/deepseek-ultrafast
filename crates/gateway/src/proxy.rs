use crate::config::{Config, Pool, bearer, load_token};
use anyhow::{Result, ensure};
use axum::{
    Json, Router,
    body::{Body, Bytes, to_bytes},
    extract::{Request, State},
    http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures::{StreamExt, future::join_all, stream};
use lru::LruCache;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeSet, HashSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use subtle::ConstantTimeEq;
use tokio::{
    sync::{Notify, OwnedSemaphorePermit, Semaphore},
    time::{self, Instant},
};

const SESSION_HEADERS: [&str; 4] = [
    "x-session-id",
    "session-id",
    "x-claude-code-session-id",
    "x-deepseek-harness-session-id",
];
const ADMISSION_HEADER: &str = "x-dsuf-admission";
const LOAD_SCALE: usize = 1000;
const AFFINITY_MARGIN_MS: u64 = 150;
const LOCAL_OCCUPANCY_PENALTY_MS: usize = 150;
const KV_PRESSURE_PENALTY_MS: usize = 100;
const MAX_CAPACITY_DOCUMENT_BYTES: usize = 4096;
const MAX_MODEL_OUTPUT_TOKENS: u64 = 256_000;
type ScopedSession = [u8; 32];

pub(super) struct Gateway {
    req: reqwest::Client,
    tenants: Vec<Tenant>,
    backends: Vec<Backend>,
    standard_cohorts: BTreeSet<String>,
    long_cohorts: BTreeSet<String>,
    max_active: usize,
    outstanding: Arc<Semaphore>,
    bytes: Arc<Semaphore>,
    max_body_bytes: usize,
    queue_timeout: Duration,
    interval: Duration,
    probe_timeout: Duration,
    health_ttl: Duration,
    max_attempts: usize,
    standard_max_output_tokens: u64,
    standard_accepts_images: bool,
    accepting: AtomicBool,
    state: Mutex<DispatchState>,
    changed: Notify,
}

struct Tenant {
    id: String,
    credential_digest: [u8; 32],
    max_active: usize,
    outstanding: Arc<Semaphore>,
    bytes: Arc<Semaphore>,
    max_body_bytes: usize,
    output_tokens: Option<Arc<Semaphore>>,
}

struct Backend {
    id: String,
    origin: reqwest::Url,
    pool: Pool,
    cohort: String,
    max_inflight: usize,
    readiness_path: String,
    probe_capacity: bool,
    vllm_cache_salt: bool,
    authorization: Option<HeaderValue>,
}

#[derive(Deserialize)]
struct Capacity {
    epoch: String,
    queued: usize,
    active: usize,
    available: usize,
    kv_free_bytes: Option<usize>,
    kv_reserved_bytes: Option<usize>,
    #[serde(default, deserialize_with = "optional_work")]
    work: Option<Work>,
}

#[derive(Deserialize)]
struct Work {
    prefill_ms: Option<u64>,
    decode_ms: Option<u64>,
    restore_ms: Option<u64>,
}

fn optional_work<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Work>, D::Error> {
    let value = Value::deserialize(deserializer)?;
    // Keep the backend's admission counters even if it reports unusable estimates.
    Ok(serde_json::from_value(value).ok())
}

impl Work {
    fn backlog_ms(&self, legacy_ms: u64) -> Option<u64> {
        let measured = [self.prefill_ms, self.decode_ms, self.restore_ms];
        let known = measured.iter().flatten().count() as u64;
        if known == 0 {
            return None;
        }
        let measured_ms = measured.into_iter().flatten().fold(0, u64::saturating_add);
        // Keep part of the count-based estimate when stages are unreported.
        Some(measured_ms.max(legacy_ms * (3 - known) / 3))
    }
}

impl Capacity {
    fn kv_pressure_ms(&self) -> u64 {
        let (Some(free), Some(reserved)) = (self.kv_free_bytes, self.kv_reserved_bytes) else {
            return 0;
        };
        (load(reserved, free.saturating_add(reserved)) * KV_PRESSURE_PENALTY_MS / LOAD_SCALE) as u64
    }
}

struct Health {
    observed: Instant,
    capacity: Option<Capacity>,
    overloaded: bool,
}

impl Health {
    fn epoch(&self) -> Option<&str> {
        self.capacity
            .as_ref()
            .map(|capacity| capacity.epoch.as_str())
    }
}

struct Owner {
    backend: usize,
    epoch: Option<String>,
}

struct DispatchState {
    active: usize,
    tenant_active: Vec<usize>,
    backend_active: Vec<usize>,
    health: Vec<Option<Health>>,
    affinity: LruCache<(Pool, ScopedSession), Owner>,
}

struct Active {
    gateway: Arc<Gateway>,
    tenant: usize,
    backend: usize,
}

impl Drop for Active {
    fn drop(&mut self) {
        {
            let mut state = self.gateway.state.lock().unwrap();
            state.active -= 1;
            state.tenant_active[self.tenant] -= 1;
            state.backend_active[self.backend] -= 1;
        }
        self.gateway.changed.notify_waiters();
    }
}

struct Outstanding {
    _global_request: OwnedSemaphorePermit,
    _tenant_request: OwnedSemaphorePermit,
    _body: BodyReservation,
    salted_body: Option<BodyReservation>,
    output: Option<OwnedSemaphorePermit>,
}

struct BodyReservation {
    _global_bytes: OwnedSemaphorePermit,
    _tenant_bytes: OwnedSemaphorePermit,
}

#[derive(Clone, Copy, Debug)]
struct GatewayError(StatusCode, &'static str);

impl GatewayError {
    fn busy() -> Self {
        Self(
            StatusCode::TOO_MANY_REQUESTS,
            "gateway admission capacity exhausted",
        )
    }
    fn unavailable() -> Self {
        Self(
            StatusCode::SERVICE_UNAVAILABLE,
            "no ready replica for the requested pool and compatibility cohort",
        )
    }
    fn draining() -> Self {
        Self(StatusCode::SERVICE_UNAVAILABLE, "gateway is draining")
    }
    fn bad_input() -> Self {
        Self(StatusCode::BAD_REQUEST, "expected a JSON request object")
    }
    fn too_large() -> Self {
        Self(
            StatusCode::PAYLOAD_TOO_LARGE,
            "request body exceeds the permitted byte limit",
        )
    }

    fn response(self, anthropic: bool) -> Response {
        let status = self.0;
        let error = if anthropic {
            let kind = match status {
                StatusCode::UNAUTHORIZED => "authentication_error",
                StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
                StatusCode::SERVICE_UNAVAILABLE => "overloaded_error",
                StatusCode::PAYLOAD_TOO_LARGE => "request_too_large",
                _ if status.is_server_error() => "api_error",
                _ => "invalid_request_error",
            };
            json!({"type":"error", "error":{"type":kind, "message":self.1}})
        } else {
            let kind = match status {
                StatusCode::UNAUTHORIZED => "authentication_error",
                StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
                _ if status.is_server_error() => "server_error",
                _ => "invalid_request_error",
            };
            json!({"error":{"type":kind, "message":self.1, "param":null, "code":null}})
        };
        let mut response = (status, Json(error)).into_response();
        if status == StatusCode::TOO_MANY_REQUESTS {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
        }
        if status == StatusCode::UNAUTHORIZED && !anthropic {
            response
                .headers_mut()
                .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        }
        response
    }
}

struct Session {
    public: String,
    scoped: ScopedSession,
    explicit: bool,
}

impl Session {
    fn parse(tenant: &str, headers: &HeaderMap, multi_turn: bool) -> Result<Self, GatewayError> {
        let mut ids = SESSION_HEADERS
            .iter()
            .flat_map(|name| headers.get_all(*name).iter());
        let first = ids.next();
        if ids.next().is_some() {
            return Err(GatewayError(
                StatusCode::BAD_REQUEST,
                "at most one session ID header is allowed",
            ));
        }
        let public = if let Some(id) = first {
            id.to_str()
                .ok()
                .filter(|s| !s.trim().is_empty() && s.len() <= 1024)
                .ok_or(GatewayError(
                    StatusCode::BAD_REQUEST,
                    "session ID must contain 1..1024 nonblank ASCII bytes",
                ))?
                .to_owned()
        } else {
            if multi_turn {
                return Err(GatewayError(
                    StatusCode::BAD_REQUEST,
                    "multi-turn requests require X-Session-ID",
                ));
            }
            format!("session-{:032x}", rand::random::<u128>())
        };
        let mut hash = Sha256::new();
        hash.update(b"dsuf-session-v1\0");
        hash.update(tenant.as_bytes());
        hash.update(b"\0");
        hash.update(public.as_bytes());
        Ok(Self {
            public,
            scoped: hash.finalize().into(),
            explicit: first.is_some(),
        })
    }

    fn private_header(&self) -> HeaderValue {
        HeaderValue::try_from(Self::hex("dsuf-", self.scoped)).unwrap()
    }

    fn vllm_cache_salt(&self, credential_digest: &[u8; 32]) -> String {
        let mut hash = Sha256::new();
        hash.update(b"dsuf-vllm-prefix-v1\0");
        hash.update(self.scoped);
        hash.update(credential_digest);
        Self::hex("", hash.finalize().into())
    }

    fn hex(prefix: &str, bytes: [u8; 32]) -> String {
        use std::fmt::Write;
        let mut key = String::with_capacity(prefix.len() + 64);
        key.push_str(prefix);
        for b in bytes {
            write!(key, "{b:02x}").unwrap();
        }
        key
    }
}

impl Backend {
    fn endpoint(&self, path: &str, query: Option<&str>) -> reqwest::Url {
        let mut target = self.origin.clone();
        target.set_path(path);
        target.set_query(query);
        target
    }

    fn authorize(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.authorization {
            Some(auth) => request.header(header::AUTHORIZATION, auth),
            None => request,
        }
    }
}

impl Gateway {
    pub fn new(config: Config) -> Result<Arc<Self>> {
        Self::build(config, load_token)
    }

    fn build(config: Config, token: impl Fn(&str) -> Result<String>) -> Result<Arc<Self>> {
        let req = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .connect_timeout(config.connect_timeout())
            .read_timeout(Duration::from_millis(config.upstream_idle_ms))
            .build()?;
        let mut tenants = Vec::with_capacity(config.tenants.len());
        let mut digests = HashSet::new();
        for tenant in config.tenants {
            let credential_digest: [u8; 32] =
                Sha256::digest(token(&tenant.token_env)?.as_bytes()).into();
            ensure!(
                digests.insert(credential_digest),
                "tenant credentials must be unique"
            );
            tenants.push(Tenant {
                id: tenant.id,
                credential_digest,
                max_active: tenant.max_active,
                outstanding: Arc::new(Semaphore::new(tenant.max_active + tenant.max_pending)),
                bytes: Arc::new(Semaphore::new(tenant.max_buffered_bytes)),
                max_body_bytes: config.max_body_bytes.min(tenant.max_buffered_bytes),
                output_tokens: tenant
                    .max_reserved_output_tokens
                    .map(|n| Arc::new(Semaphore::new(n))),
            });
        }
        let mut backends = Vec::with_capacity(config.backends.len());
        let mut standard_cohorts = BTreeSet::new();
        let mut long_cohorts = BTreeSet::new();
        for backend in config.backends {
            match backend.pool {
                Pool::Standard => &mut standard_cohorts,
                Pool::Long => &mut long_cohorts,
            }
            .insert(backend.cohort.clone());
            backends.push(Backend {
                id: backend.id,
                origin: backend.url,
                pool: backend.pool,
                cohort: backend.cohort,
                max_inflight: backend.max_inflight,
                readiness_path: backend.readiness_path,
                probe_capacity: backend.probe_capacity,
                vllm_cache_salt: backend.vllm_cache_salt,
                authorization: backend
                    .token_env
                    .as_deref()
                    .map(|env| bearer(&token(env)?))
                    .transpose()?,
            });
        }
        let state = DispatchState {
            active: 0,
            tenant_active: vec![0; tenants.len()],
            backend_active: vec![0; backends.len()],
            health: backends.iter().map(|_| None).collect(),
            affinity: LruCache::new(config.affinity_entries),
        };
        Ok(Arc::new(Self {
            req,
            tenants,
            backends,
            standard_cohorts,
            long_cohorts,
            max_active: config.max_active,
            outstanding: Arc::new(Semaphore::new(config.max_active + config.max_pending)),
            bytes: Arc::new(Semaphore::new(config.max_buffered_bytes)),
            max_body_bytes: config.max_body_bytes,
            queue_timeout: Duration::from_millis(config.queue_timeout_ms),
            interval: Duration::from_millis(config.probe_interval_ms),
            probe_timeout: Duration::from_millis(config.probe_timeout_ms),
            health_ttl: Duration::from_millis(config.health_ttl_ms),
            max_attempts: config.max_attempts,
            standard_max_output_tokens: config.standard_max_output_tokens,
            standard_accepts_images: config.standard_accepts_images,
            accepting: AtomicBool::new(true),
            state: Mutex::new(state),
            changed: Notify::new(),
        }))
    }

    pub fn router(self: &Arc<Self>) -> Router {
        Router::new()
            .route("/health", get(ready))
            .route("/readyz", get(ready))
            .route("/v1/models", get(proxy))
            .route("/v1/chat/completions", post(proxy))
            .route("/v1/completions", post(proxy))
            .route("/v1/responses", post(proxy))
            .route("/v1/messages", post(proxy))
            .with_state(self.clone())
    }

    pub fn stop(&self) {
        {
            let _state = self.state.lock().unwrap();
            self.accepting.store(false, Ordering::Release);
        }
        self.changed.notify_waiters();
    }

    fn fresh<'a>(&self, health: &'a Option<Health>) -> Option<&'a Health> {
        health
            .as_ref()
            .filter(|h| h.observed.elapsed() <= self.health_ttl)
    }

    pub async fn probe_loop(self: Arc<Self>) {
        let mut intervals = time::interval(self.interval);
        intervals.set_missed_tick_behavior(time::MissedTickBehavior::Skip);
        while self.accepting.load(Ordering::Acquire) {
            intervals.tick().await;
            self.probe_all().await;
        }
    }

    async fn probe_all(&self) {
        join_all(
            self.backends
                .iter()
                .enumerate()
                .map(|(idx, backend)| async move {
                    let health = self.probe(backend).await;
                    self.state.lock().unwrap().health[idx] = health;
                    self.changed.notify_waiters();
                }),
        )
        .await;
    }

    async fn probe(&self, backend: &Backend) -> Option<Health> {
        let ready = backend.authorize(
            self.req
                .get(backend.endpoint(&backend.readiness_path, None)),
        );
        let ready = time::timeout(self.probe_timeout, ready.send())
            .await
            .ok()?
            .ok()?;
        if !ready.status().is_success() {
            return None;
        }
        let capacity = if backend.probe_capacity {
            time::timeout(self.probe_timeout, self.read_capacity(backend))
                .await
                .ok()
                .flatten()
        } else {
            None
        };
        Some(Health {
            observed: Instant::now(),
            capacity,
            overloaded: false,
        })
    }

    async fn read_capacity(&self, backend: &Backend) -> Option<Capacity> {
        let response = backend
            .authorize(self.req.get(backend.endpoint("/internal/admission", None)))
            .send()
            .await
            .ok()?;
        if !response.status().is_success() {
            return None;
        }
        let mut response = response.bytes_stream().boxed();
        let mut data = Vec::new();
        while let Some(chunk) = response.next().await {
            let chunk = chunk.ok()?;
            if chunk.len() > MAX_CAPACITY_DOCUMENT_BYTES - data.len() {
                return None;
            }
            data.extend_from_slice(&chunk);
        }
        let capacity: Capacity = serde_json::from_slice(&data).ok()?;
        (!capacity.epoch.is_empty()).then_some(capacity)
    }

    fn authenticate(&self, headers: &HeaderMap) -> Result<usize, GatewayError> {
        let rejected = || {
            GatewayError(
                StatusCode::UNAUTHORIZED,
                "a valid tenant credential is required",
            )
        };
        let mut auth = headers.get_all(header::AUTHORIZATION).iter();
        let mut key = headers.get_all("x-api-key").iter();
        let token = match (auth.next(), key.next()) {
            (Some(bearer), None) if auth.next().is_none() => {
                let (scheme, token) = bearer
                    .to_str()
                    .map_err(|_| rejected())?
                    .split_once(' ')
                    .ok_or_else(rejected)?;
                if !scheme.eq_ignore_ascii_case("bearer") {
                    return Err(rejected());
                }
                token
            }
            (None, Some(key_header)) if key.next().is_none() => {
                key_header.to_str().map_err(|_| rejected())?
            }
            _ => return Err(rejected()),
        };
        if token.is_empty() || token.len() > 4096 {
            return Err(rejected());
        }
        let hash: [u8; 32] = Sha256::digest(token.as_bytes()).into();
        let mut index = None;
        for (i, tenant) in self.tenants.iter().enumerate() {
            if bool::from(tenant.credential_digest.ct_eq(&hash)) {
                index = Some(i);
            }
        }
        index.ok_or_else(rejected)
    }

    fn reserve(&self, tenant: usize, bytes: usize) -> Result<Outstanding, GatewayError> {
        let busy = |_| GatewayError::busy();
        Ok(Outstanding {
            _global_request: self.outstanding.clone().try_acquire_owned().map_err(busy)?,
            _tenant_request: self.tenants[tenant]
                .outstanding
                .clone()
                .try_acquire_owned()
                .map_err(busy)?,
            _body: self.reserve_body_bytes(tenant, bytes)?,
            salted_body: None,
            output: None,
        })
    }

    fn reserve_body_bytes(
        &self,
        tenant: usize,
        bytes: usize,
    ) -> Result<BodyReservation, GatewayError> {
        let bytes = u32::try_from(bytes).map_err(|_| GatewayError::too_large())?;
        let busy = |_| GatewayError::busy();
        Ok(BodyReservation {
            _global_bytes: self
                .bytes
                .clone()
                .try_acquire_many_owned(bytes)
                .map_err(busy)?,
            _tenant_bytes: self.tenants[tenant]
                .bytes
                .clone()
                .try_acquire_many_owned(bytes)
                .map_err(busy)?,
        })
    }

    fn cohort(&self, pool: Pool, session: &ScopedSession) -> Option<&str> {
        let cohorts = match pool {
            Pool::Standard => &self.standard_cohorts,
            Pool::Long => &self.long_cohorts,
        };
        cohorts
            .iter()
            .max_by_key(|cohort| affinity_score(session, cohort))
            .map(String::as_str)
    }

    fn candidates<'a>(
        &'a self,
        pool: Pool,
        cohort: &'a str,
        exclude: &'a [usize],
    ) -> impl Iterator<Item = (usize, &'a Backend)> {
        self.backends
            .iter()
            .enumerate()
            .filter(move |(idx, backend)| {
                backend.pool == pool && backend.cohort == cohort && !exclude.contains(idx)
            })
    }

    fn select(
        self: &Arc<Self>,
        tenant: usize,
        pool: Pool,
        cohort: &str,
        session: ScopedSession,
        exclude: &[usize],
        remaining: Duration,
    ) -> Option<Active> {
        let mut state = self.state.lock().unwrap();
        if !self.accepting.load(Ordering::Acquire)
            || state.active >= self.max_active
            || state.tenant_active[tenant] >= self.tenants[tenant].max_active
        {
            return None;
        }
        let warm = state
            .affinity
            .get(&(pool, session))
            .map(|owner| (owner.backend, owner.epoch.clone()));
        let mut best = None;
        let mut warm_candidate = None;
        for (idx, backend) in self.candidates(pool, cohort, exclude) {
            let Some(health) = self.fresh(&state.health[idx]) else {
                continue;
            };
            if health.overloaded || state.backend_active[idx] >= backend.max_inflight {
                continue;
            }
            let local = load(state.backend_active[idx], backend.max_inflight);
            let (remote, queue_cost, free, capacity) = match &health.capacity {
                Some(capacity) if capacity.available == 0 => continue,
                Some(capacity) => {
                    let used = capacity.queued.saturating_add(capacity.active);
                    let total = used.saturating_add(capacity.available);
                    (
                        load(used, total),
                        capacity.queued.min(32) * 20,
                        capacity.kv_free_bytes.unwrap_or(0),
                        Some(capacity),
                    )
                }
                None => (0, 0, 0, None),
            };
            let legacy_ms = (local.max(remote) + queue_cost) as u64;
            let mut score_ms = legacy_ms;
            let mut late = false;
            if let Some(capacity) = capacity
                && let Some(backlog_ms) = capacity
                    .work
                    .as_ref()
                    .and_then(|work| work.backlog_ms(legacy_ms))
            {
                // Ingress may contain turns that have yet to reach the measured stages.
                let queue_ms = backlog_ms.max(queue_cost as u64);
                late = Duration::from_millis(queue_ms) > remaining;
                let local_ms = (local * LOCAL_OCCUPANCY_PENALTY_MS / LOAD_SCALE) as u64;
                score_ms = queue_ms
                    .saturating_add(local_ms)
                    .saturating_add(capacity.kv_pressure_ms());
            }
            let key = (
                late,
                score_ms,
                std::cmp::Reverse(free),
                std::cmp::Reverse(affinity_score(&session, &backend.id)),
            );
            if best.as_ref().is_none_or(|(_, best_key)| key < *best_key) {
                best = Some((idx, key));
            }
            if warm.as_ref().is_some_and(|(warm_idx, epoch)| {
                *warm_idx == idx && epoch.as_deref() == health.epoch()
            }) {
                warm_candidate = Some((idx, late, score_ms));
            }
        }
        let (best_idx, (_, best_score, ..)) = best?;
        // Keep warm affinity only when its estimated work fits the remaining
        // dispatch window. Estimates still allow dispatch to the best ready replica.
        let backend = warm_candidate
            .filter(|&(_, late, score)| {
                !late && score <= best_score.saturating_add(AFFINITY_MARGIN_MS)
            })
            .map_or(best_idx, |(idx, ..)| idx);
        state.active += 1;
        state.tenant_active[tenant] += 1;
        state.backend_active[backend] += 1;
        Some(Active {
            gateway: self.clone(),
            tenant,
            backend,
        })
    }

    async fn admit(
        self: &Arc<Self>,
        tenant: usize,
        pool: Pool,
        session: ScopedSession,
        exclude: &[usize],
        deadline: Instant,
    ) -> Result<Active, GatewayError> {
        let cohort = self
            .cohort(pool, &session)
            .ok_or_else(GatewayError::unavailable)?;
        if self.candidates(pool, cohort, exclude).next().is_none() {
            return Err(GatewayError::unavailable());
        }
        loop {
            if !self.accepting.load(Ordering::Acquire) {
                return Err(GatewayError::draining());
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if let Some(active) = self.select(tenant, pool, cohort, session, exclude, remaining) {
                return Ok(active);
            }
            if time::timeout_at(deadline, changed).await.is_err() {
                break;
            }
        }
        let state = self.state.lock().unwrap();
        let ready = self
            .candidates(pool, cohort, exclude)
            .any(|(idx, _)| self.fresh(&state.health[idx]).is_some());
        Err(if ready {
            GatewayError::busy()
        } else {
            GatewayError::unavailable()
        })
    }

    fn record_owner(&self, pool: Pool, session: ScopedSession, backend: usize) {
        let mut state = self.state.lock().unwrap();
        let epoch = self
            .fresh(&state.health[backend])
            .and_then(Health::epoch)
            .map(str::to_owned);
        state
            .affinity
            .put((pool, session), Owner { backend, epoch });
    }

    fn backend_rejected(&self, backend: usize, reason: Rejection) {
        {
            let mut state = self.state.lock().unwrap();
            match reason {
                Rejection::Closed => state.health[backend] = None,
                Rejection::Full => {
                    if let Some(health) = &mut state.health[backend] {
                        health.overloaded = true;
                    }
                }
                Rejection::Context => return,
            }
        }
        self.changed.notify_waiters();
    }
}

fn load(used: usize, total: usize) -> usize {
    ((used as u128 * LOAD_SCALE as u128) / total.max(1) as u128).min(LOAD_SCALE as u128) as usize
}

fn affinity_score(session: &ScopedSession, item: &str) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(session);
    hash.update(item.as_bytes());
    hash.finalize().into()
}

async fn ready(State(gateway): State<Arc<Gateway>>) -> StatusCode {
    let state = gateway.state.lock().unwrap();
    if gateway.accepting.load(Ordering::Acquire)
        && gateway
            .backends
            .iter()
            .zip(&state.health)
            .any(|(backend, health)| {
                backend.pool == Pool::Standard && gateway.fresh(health).is_some()
            })
    {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Rejection {
    Context,
    Full,
    Closed,
}

fn rejection(response: &reqwest::Response) -> Option<Rejection> {
    let marker = response.headers().get(ADMISSION_HEADER)?.to_str().ok()?;
    match (response.status(), marker) {
        (StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY, "context") => {
            Some(Rejection::Context)
        }
        (StatusCode::TOO_MANY_REQUESTS, "full") => Some(Rejection::Full),
        (StatusCode::SERVICE_UNAVAILABLE, "closed") => Some(Rejection::Closed),
        _ => None,
    }
}

async fn proxy(State(gateway): State<Arc<Gateway>>, request: Request) -> Response {
    let anthropic = request.uri().path() == "/v1/messages";
    match handle(gateway, request).await {
        Ok(response) => response,
        Err(error) => error.response(anthropic),
    }
}

async fn handle(gateway: Arc<Gateway>, request: Request) -> Result<Response, GatewayError> {
    let deadline = Instant::now() + gateway.queue_timeout;
    if !gateway.accepting.load(Ordering::Acquire) {
        return Err(GatewayError::draining());
    }
    let tenant = gateway.authenticate(request.headers())?;
    let (parts, body) = request.into_parts();
    let max_bytes = gateway
        .max_body_bytes
        .min(gateway.tenants[tenant].max_body_bytes);
    let post = parts.method == Method::POST;
    let body_limit = if !post {
        0
    } else if let Some(length) = parts.headers.get(header::CONTENT_LENGTH) {
        let length = length
            .to_str()
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .ok_or_else(GatewayError::bad_input)?;
        if length > max_bytes {
            return Err(GatewayError::too_large());
        }
        length
    } else {
        max_bytes
    };
    let mut outstanding = gateway.reserve(tenant, body_limit)?;
    let bytes = if post {
        time::timeout_at(deadline, to_bytes(body, body_limit))
            .await
            .map_err(|_| {
                GatewayError(
                    StatusCode::REQUEST_TIMEOUT,
                    "request body read deadline expired",
                )
            })?
            .map_err(|_| GatewayError::too_large())?
    } else {
        Bytes::new()
    };
    let path = parts.uri.path();
    let (multi_turn, mut pool, output_tokens) = if post {
        let value: Value = serde_json::from_slice(&bytes).map_err(|_| GatewayError::bad_input())?;
        if !value.is_object() {
            return Err(GatewayError::bad_input());
        }
        classify(
            path,
            &value,
            gateway.standard_max_output_tokens,
            gateway.standard_accepts_images,
        )
    } else {
        (false, Pool::Standard, 0)
    };
    let session = Session::parse(&gateway.tenants[tenant].id, &parts.headers, multi_turn)?;
    if let Some(budget) = &gateway.tenants[tenant].output_tokens {
        outstanding.output = Some(
            budget
                .clone()
                .try_acquire_many_owned(output_tokens)
                .map_err(|_| {
                    GatewayError(
                        StatusCode::TOO_MANY_REQUESTS,
                        "tenant output token budget exhausted",
                    )
                })?,
        );
    }
    let mut tried = Vec::with_capacity(gateway.max_attempts.min(gateway.backends.len()));
    let mut last_response = None;
    let mut last_error = None;
    let mut salted_body = None;
    for _ in 0..gateway.max_attempts {
        let active = match gateway
            .admit(tenant, pool, session.scoped, &tried, deadline)
            .await
        {
            Ok(active) => active,
            Err(error) => {
                last_error.get_or_insert(error);
                break;
            }
        };
        let index = active.backend;
        let backend = &gateway.backends[index];
        tried.push(index);
        let mut request = gateway.req.request(
            parts.method.clone(),
            backend.endpoint(path, parts.uri.query()),
        );
        for name in [
            "accept",
            "content-type",
            "anthropic-version",
            "anthropic-beta",
            "x-request-id",
        ] {
            for value in parts.headers.get_all(name) {
                request = request.header(name, value);
            }
        }
        if post {
            let body = if backend.vllm_cache_salt {
                if salted_body.is_none() {
                    let mut value: Value =
                        serde_json::from_slice(&bytes).map_err(|_| GatewayError::bad_input())?;
                    value["cache_salt"] = Value::String(
                        session.vllm_cache_salt(&gateway.tenants[tenant].credential_digest),
                    );
                    let salted =
                        serde_json::to_vec(&value).map_err(|_| GatewayError::bad_input())?;
                    outstanding.salted_body =
                        Some(gateway.reserve_body_bytes(tenant, salted.len())?);
                    salted_body = Some(Bytes::from(salted));
                }
                salted_body
                    .as_ref()
                    .expect("initialized salted body")
                    .clone()
            } else {
                bytes.clone()
            };
            request = request
                .header("x-session-id", session.private_header())
                .body(body);
        }
        let response = match backend.authorize(request).send().await {
            Ok(response) => response,
            Err(error) if error.is_connect() => {
                gateway.backend_rejected(index, Rejection::Closed);
                last_error = Some(GatewayError(
                    StatusCode::BAD_GATEWAY,
                    "could not connect to replica",
                ));
                continue;
            }
            Err(error) => {
                return Err(if error.is_timeout() {
                    GatewayError(StatusCode::GATEWAY_TIMEOUT, "replica response timed out")
                } else {
                    GatewayError(
                        StatusCode::BAD_GATEWAY,
                        "replica response failed before headers",
                    )
                });
            }
        };
        if let Some(reason) = rejection(&response) {
            gateway.backend_rejected(index, reason);
            if reason == Rejection::Context {
                if pool == Pool::Long || gateway.long_cohorts.is_empty() {
                    return Ok(relay(
                        response,
                        Some(active),
                        outstanding,
                        post.then_some(&session),
                    ));
                }
                pool = Pool::Long;
            }
            last_response = Some(response);
            continue;
        }
        if post && response.status().is_success() {
            gateway.record_owner(pool, session.scoped, index);
        }
        return Ok(relay(
            response,
            Some(active),
            outstanding,
            post.then_some(&session),
        ));
    }
    if let Some(response) = last_response {
        return Ok(relay(response, None, outstanding, post.then_some(&session)));
    }
    Err(last_error.unwrap_or_else(GatewayError::unavailable))
}

fn classify(
    path: &str,
    value: &Value,
    max_output: u64,
    standard_accepts_images: bool,
) -> (bool, Pool, u32) {
    let (turns, output) = match path {
        "/v1/chat/completions" => (
            value.get("messages"),
            value["max_completion_tokens"]
                .as_u64()
                .or(value["max_tokens"].as_u64()),
        ),
        "/v1/responses" => (value.get("input"), value["max_output_tokens"].as_u64()),
        "/v1/messages" => (value.get("messages"), value["max_tokens"].as_u64()),
        _ => (
            None,
            value["max_completion_tokens"]
                .as_u64()
                .or(value["max_tokens"].as_u64())
                .or(Some(16)),
        ),
    };
    let multi_turn = turns.and_then(Value::as_array).is_some_and(|turns| {
        turns
            .iter()
            .filter(|turn| !matches!(turn["role"].as_str(), Some("system" | "developer")))
            .nth(1)
            .is_some()
    });
    let requires_vision_pool = !standard_accepts_images && turns.is_some_and(has_image);
    let pool = if requires_vision_pool || output.is_none_or(|max| max > max_output) {
        Pool::Long
    } else {
        Pool::Standard
    };
    let output_reservation = output
        .unwrap_or(MAX_MODEL_OUTPUT_TOKENS)
        .min(MAX_MODEL_OUTPUT_TOKENS);
    (multi_turn, pool, output_reservation as u32)
}

fn has_image(value: &Value) -> bool {
    match value {
        Value::Array(items) => items.iter().any(has_image),
        Value::Object(object) => {
            matches!(
                object.get("type").and_then(Value::as_str),
                Some("image" | "image_url" | "input_image")
            ) || object.get("content").is_some_and(has_image)
        }
        _ => false,
    }
}

fn relay(
    upstream: reqwest::Response,
    active: Option<Active>,
    outstanding: Outstanding,
    session: Option<&Session>,
) -> Response {
    let status = upstream.status();
    let mut headers = HeaderMap::new();
    for name in [
        header::CONTENT_TYPE,
        header::CONTENT_ENCODING,
        header::RETRY_AFTER,
        header::CACHE_CONTROL,
        HeaderName::from_static("x-request-id"),
        HeaderName::from_static("request-id"),
    ] {
        for value in upstream.headers().get_all(&name) {
            headers.append(&name, value.clone());
        }
    }
    if let Some(session) = session.filter(|_| status.is_success()) {
        headers.insert(
            "x-session-id",
            HeaderValue::from_str(&session.public).expect("validated session ID"),
        );
        headers.insert(
            "x-session-id-source",
            HeaderValue::from_static(if session.explicit {
                "explicit"
            } else {
                "generated"
            }),
        );
    }
    let upstream = upstream.bytes_stream().boxed();
    let body = stream::unfold(
        (upstream, active, outstanding),
        |(mut upstream, active, outstanding)| async move {
            let part = upstream.next().await?;
            Some((part, (upstream, active, outstanding)))
        },
    );
    let mut response = Response::new(Body::from_stream(body));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}

#[cfg(test)]
#[path = "proxy_tests.rs"]
mod tests;
