mod config;
mod proxy;

use anyhow::{Context, Result};
use config::Config;
use std::{future::IntoFuture, sync::Arc, time::Duration};
use tokio::{net::TcpListener, sync::Notify, time};

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .context("usage: deepseek-ultrafast-gateway CONFIG.json")?;
    if path == "-h" || path == "--help" {
        println!("usage: deepseek-ultrafast-gateway CONFIG.json");
        return Ok(());
    }
    anyhow::ensure!(
        args.next().is_none(),
        "usage: deepseek-ultrafast-gateway CONFIG.json"
    );
    let bytes = std::fs::read(path).context("read gateway config")?;
    let config: Config = serde_json::from_slice(&bytes).context("parse gateway config")?;
    config.validate()?;
    let listener = TcpListener::bind(config.listen)
        .await
        .context("bind gateway listener")?;
    let drain = Duration::from_millis(config.drain_timeout_ms);
    let gateway = proxy::Gateway::new(config)?;
    let probes = tokio::spawn(gateway.clone().probe_loop());
    let shutdown = Arc::new(Notify::new());
    let drain_signal = shutdown.clone();
    let server = axum::serve(listener, gateway.router()).with_graceful_shutdown(async move {
        drain_signal.notified().await;
    });
    let server = server.into_future();
    tokio::pin!(server);
    let result = tokio::select! {
        result = &mut server => result.context("serve gateway"),
        result = termination() => {
            gateway.stop();
            shutdown.notify_one();
            result?;
            time::timeout(drain, &mut server).await.context("gateway drain deadline expired")?.context("serve gateway")
        }
    };
    probes.abort();
    result
}

#[cfg(unix)]
async fn termination() -> std::io::Result<()> {
    use tokio::signal::unix::{SignalKind, signal};
    let mut term = signal(SignalKind::terminate())?;
    tokio::select! {
        result = tokio::signal::ctrl_c() => result,
        _ = term.recv() => Ok(()),
    }
}

#[cfg(not(unix))]
async fn termination() -> std::io::Result<()> {
    tokio::signal::ctrl_c().await
}
