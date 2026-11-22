use anyhow::{Context, Result, ensure};
use axum::http::HeaderValue;
use serde::Deserialize;
use std::{collections::HashSet, net::SocketAddr, num::NonZeroUsize, time::Duration};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Pool {
    Standard,
    Long,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TenantConfig {
    pub id: String,
    pub token_env: String,
    pub max_active: usize,
    pub max_pending: usize,
    pub max_buffered_bytes: usize,
    pub max_reserved_output_tokens: Option<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BackendConfig {
    pub id: String,
    #[serde(deserialize_with = "parse_origin")]
    pub url: reqwest::Url,
    pub pool: Pool,
    pub cohort: String,
    pub max_inflight: usize,
    #[serde(default = "default_readiness_path")]
    pub readiness_path: String,
    #[serde(default = "default_capacity_probe")]
    pub probe_capacity: bool,
    #[serde(default)]
    pub vllm_cache_salt: bool,
    pub token_env: Option<String>,
}

fn default_readiness_path() -> String {
    "/readyz".into()
}

fn default_capacity_probe() -> bool {
    true
}

fn parse_origin<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<reqwest::Url, D::Error> {
    let value = String::deserialize(deserializer)?;
    reqwest::Url::parse(&value).map_err(serde::de::Error::custom)
}

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct Config {
    pub listen: SocketAddr,
    pub max_active: usize,
    pub max_pending: usize,
    pub max_buffered_bytes: usize,
    pub max_body_bytes: usize,
    pub queue_timeout_ms: u64,
    pub upstream_idle_ms: u64,
    pub probe_interval_ms: u64,
    pub probe_timeout_ms: u64,
    pub health_ttl_ms: u64,
    pub drain_timeout_ms: u64,
    pub max_attempts: usize,
    pub affinity_entries: NonZeroUsize,
    pub standard_max_output_tokens: u64,
    pub standard_accepts_images: bool,
    pub tenants: Vec<TenantConfig>,
    pub backends: Vec<BackendConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen: ([0, 0, 0, 0], 8080).into(),
            max_active: 512,
            max_pending: 128,
            max_buffered_bytes: 512 << 20,
            max_body_bytes: 64 << 20,
            queue_timeout_ms: 2_000,
            upstream_idle_ms: 120_000,
            probe_interval_ms: 500,
            probe_timeout_ms: 300,
            health_ttl_ms: 2_000,
            drain_timeout_ms: 55_000,
            max_attempts: 3,
            affinity_entries: NonZeroUsize::new(16_384).unwrap(),
            standard_max_output_tokens: 8_192,
            standard_accepts_images: true,
            tenants: Vec::new(),
            backends: Vec::new(),
        }
    }
}

fn valid_label(label: &str) -> bool {
    !label.is_empty()
        && label.len() <= 128
        && label
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

fn valid_limits(active: usize, pending: usize, bytes: usize) -> bool {
    let max = tokio::sync::Semaphore::MAX_PERMITS;
    active > 0
        && active
            .checked_add(pending)
            .is_some_and(|total| total <= max)
        && (1..=max).contains(&bytes)
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            valid_limits(self.max_active, self.max_pending, self.max_buffered_bytes),
            "invalid gateway limits"
        );
        ensure!(
            self.max_body_bytes > 0
                && self.max_body_bytes <= self.max_buffered_bytes
                && self.max_body_bytes <= u32::MAX as usize,
            "max_body_bytes must fit the gateway byte budget and a 32-bit permit count"
        );
        ensure!(
            self.queue_timeout_ms > 0
                && self.upstream_idle_ms > 0
                && self.probe_interval_ms > 0
                && self.probe_timeout_ms > 0
                && self.drain_timeout_ms > 0
                && self.health_ttl_ms
                    > self
                        .probe_interval_ms
                        .saturating_add(self.probe_timeout_ms.saturating_mul(2)),
            "invalid timeouts or health TTL"
        );
        ensure!(
            self.max_attempts > 0 && self.standard_max_output_tokens > 0,
            "invalid routing limits"
        );
        ensure!(
            !self.tenants.is_empty(),
            "configure at least one authenticated tenant"
        );
        ensure!(
            self.backends
                .iter()
                .any(|backend| backend.pool == Pool::Standard),
            "configure a standard pool backend"
        );

        let mut tenants = HashSet::new();
        for tenant in &self.tenants {
            ensure!(
                valid_label(&tenant.id) && tenants.insert(&tenant.id),
                "invalid or duplicate tenant ID"
            );
            ensure!(
                valid_limits(
                    tenant.max_active,
                    tenant.max_pending,
                    tenant.max_buffered_bytes
                ),
                "invalid limits for tenant {}",
                tenant.id
            );
            ensure!(
                tenant.max_reserved_output_tokens.is_none_or(|tokens| (1
                    ..=tokio::sync::Semaphore::MAX_PERMITS)
                    .contains(&tokens)),
                "invalid output token limit for tenant {}",
                tenant.id
            );
        }

        let (mut ids, mut urls) = (HashSet::new(), HashSet::new());
        for backend in &self.backends {
            ensure!(
                valid_label(&backend.id) && ids.insert(&backend.id),
                "invalid or duplicate backend ID"
            );
            ensure!(
                valid_label(&backend.cohort),
                "invalid cohort for backend {}",
                backend.id
            );
            ensure!(
                backend.max_inflight > 0
                    && backend.max_inflight <= tokio::sync::Semaphore::MAX_PERMITS,
                "invalid stream limit for backend {}",
                backend.id
            );
            let url = &backend.url;
            ensure!(
                matches!(url.scheme(), "http" | "https")
                    && url.has_host()
                    && url.username().is_empty()
                    && url.password().is_none()
                    && url.path() == "/"
                    && url.query().is_none()
                    && url.fragment().is_none()
                    && urls.insert(url),
                "invalid or duplicate origin for backend {}",
                backend.id
            );
            ensure!(
                backend.readiness_path.starts_with('/')
                    && !backend.readiness_path.starts_with("//")
                    && !backend.readiness_path.contains(['\\', '?', '#'])
                    && backend
                        .readiness_path
                        .parse::<axum::http::uri::PathAndQuery>()
                        .is_ok(),
                "invalid readiness path for backend {}",
                backend.id
            );
        }
        Ok(())
    }

    pub fn connect_timeout(&self) -> Duration {
        Duration::from_millis(self.probe_timeout_ms)
    }
}

pub(super) fn load_token(env: &str) -> Result<String> {
    let token = std::env::var(env)
        .with_context(|| format!("read credential environment variable {env}"))?;
    ensure!(
        !token.is_empty() && token.len() <= 4096 && token.bytes().all(|b| b.is_ascii_graphic()),
        "invalid credential in environment variable {env}"
    );
    Ok(token)
}

pub(super) fn bearer(token: &str) -> Result<HeaderValue> {
    let mut value = HeaderValue::from_str(&format!("Bearer {token}"))?;
    value.set_sensitive(true);
    Ok(value)
}
