use super::*;
use crate::config::{BackendConfig, TenantConfig};
use tokio::{net::TcpListener, task::JoinHandle};

struct TestServer {
    url: reqwest::Url,
    task: JoinHandle<()>,
}

impl TestServer {
    async fn start(router: Router) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap())
            .parse()
            .unwrap();
        let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        Self { url, task }
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[derive(Clone)]
struct Mock {
    status: StatusCode,
    marker: Option<&'static str>,
    capacity: Value,
    requests: Arc<Mutex<Vec<(HeaderMap, Bytes)>>>,
    hold_body: Option<Arc<Semaphore>>,
}

impl Mock {
    fn new(status: StatusCode, marker: Option<&'static str>) -> Self {
        Self {
            status,
            marker,
            capacity: json!({"epoch":"test-epoch", "queued":0, "active":0, "available":4}),
            requests: Default::default(),
            hold_body: None,
        }
    }

    async fn serve(&self) -> TestServer {
        TestServer::start(
            Router::new()
                .route("/readyz", get(|| async { StatusCode::OK }))
                .route(
                    "/internal/admission",
                    get(|State(mock): State<Mock>| async move { Json(mock.capacity) }),
                )
                .route("/v1/chat/completions", post(mock_generation))
                .with_state(self.clone()),
        )
        .await
    }
}

async fn mock_generation(State(mock): State<Mock>, headers: HeaderMap, body: Bytes) -> Response {
    mock.requests.lock().unwrap().push((headers, body));
    let mut response = Response::new(if let Some(hold) = mock.hold_body {
        Body::from_stream(
            stream::once(async { Ok::<_, std::io::Error>("data: first\n\n") }).chain(stream::once(
                async move {
                    hold.acquire().await.unwrap().forget();
                    Ok::<_, std::io::Error>("data: last\n\n")
                },
            )),
        )
    } else {
        Body::from("data: expected\n\n")
    });
    *response.status_mut() = mock.status;
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    headers.insert("x-session-id", HeaderValue::from_static("private"));
    if let Some(marker) = mock.marker {
        headers.insert(ADMISSION_HEADER, HeaderValue::from_static(marker));
    }
    response
}

fn config(backends: Vec<BackendConfig>) -> Config {
    Config {
        max_active: 4,
        max_pending: 0,
        max_body_bytes: 1024,
        max_buffered_bytes: 4096,
        queue_timeout_ms: 1000,
        tenants: ["one", "two"]
            .into_iter()
            .map(|id| TenantConfig {
                id: id.into(),
                token_env: format!("tenant-{id}"),
                max_active: 1,
                max_pending: 0,
                max_buffered_bytes: 1024,
                max_reserved_output_tokens: Some(100),
            })
            .collect(),
        backends,
        ..Default::default()
    }
}

fn backend(id: &str, server: &TestServer, pool: Pool) -> BackendConfig {
    BackendConfig {
        id: id.into(),
        url: server.url.clone(),
        pool,
        cohort: "test".into(),
        max_inflight: 4,
        readiness_path: "/readyz".into(),
        probe_capacity: true,
        vllm_cache_salt: false,
        token_env: Some("upstream".into()),
    }
}

async fn gateway(config: Config) -> (Arc<Gateway>, TestServer) {
    config.validate().unwrap();
    let gateway = Gateway::build(config, |name| Ok(name.to_owned())).unwrap();
    gateway.probe_all().await;
    let server = TestServer::start(gateway.router()).await;
    (gateway, server)
}

const COMPLETION: &str = r#"{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"test"}],"max_tokens":100,"stream":true}"#;

fn request(client: &reqwest::Client, server: &TestServer, tenant: &str) -> reqwest::RequestBuilder {
    client
        .post(server.url.join("/v1/chat/completions").unwrap())
        .bearer_auth(format!("tenant-{tenant}"))
        .header("x-claude-code-session-id", "client-session")
        .header(header::CONTENT_TYPE, "application/json")
        .body(COMPLETION)
}

#[tokio::test]
async fn namespaces_sessions_and_holds_permits_through_stream_body() {
    let hold_body = Arc::new(Semaphore::new(0));
    let mut mock = Mock::new(StatusCode::OK, None);
    mock.hold_body = Some(hold_body.clone());
    let replica = mock.serve().await;
    let mut settings = config(vec![backend("small", &replica, Pool::Standard)]);
    settings.tenants[0].max_pending = 1;
    let (_, server) = gateway(settings).await;
    let client = reqwest::Client::new();

    let first = request(&client, &server, "one").send().await.unwrap();
    assert_eq!(first.status(), StatusCode::OK);
    assert_eq!(first.headers()["x-session-id"], "client-session");
    let over_quota = request(&client, &server, "one").send().await.unwrap();
    assert_eq!(over_quota.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(
        over_quota
            .text()
            .await
            .unwrap()
            .contains("tenant output token budget exhausted")
    );
    assert_eq!(
        request(&client, &server, "bad")
            .send()
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );

    let second = request(&client, &server, "two").send().await.unwrap();
    assert_eq!(second.status(), StatusCode::OK);
    hold_body.add_permits(2);
    for response in [first, second] {
        assert_eq!(
            response.text().await.unwrap(),
            "data: first\n\ndata: last\n\n"
        );
    }
    let requests = mock.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_ne!(requests[0].0["x-session-id"], requests[1].0["x-session-id"]);
    for (headers, bytes) in requests.iter() {
        assert!(
            headers["x-session-id"]
                .to_str()
                .unwrap()
                .starts_with("dsuf-")
        );
        assert!(!headers.contains_key("x-claude-code-session-id"));
        assert_eq!(headers[header::AUTHORIZATION], "Bearer upstream");
        assert_eq!(*bytes, COMPLETION);
    }
}

#[tokio::test]
async fn reroutes_context_refusal_and_preserves_ambiguous_failure() {
    let reject = Mock::new(StatusCode::BAD_REQUEST, Some("context"));
    let long = Mock::new(StatusCode::OK, None);
    let busy_with_unknown_acceptance = Mock::new(StatusCode::SERVICE_UNAVAILABLE, None);
    let (a, b, c) = tokio::join!(
        reject.serve(),
        long.serve(),
        busy_with_unknown_acceptance.serve()
    );
    let (gateway, server) = gateway(config(vec![
        backend("small-a", &a, Pool::Standard),
        backend("large", &b, Pool::Long),
        backend("small-b", &c, Pool::Standard),
    ]))
    .await;
    let mut headers = HeaderMap::new();
    headers.insert("x-session-id", HeaderValue::from_static("client-session"));
    let session = Session::parse("one", &headers, false).unwrap();
    let client = reqwest::Client::new();

    gateway.record_owner(Pool::Standard, session.scoped, 0);
    let result = request(&client, &server, "one").send().await.unwrap();
    assert_eq!(result.status(), StatusCode::OK);
    assert!(!result.headers().contains_key(ADMISSION_HEADER));
    assert_eq!(result.text().await.unwrap(), "data: expected\n\n");
    assert_eq!(reject.requests.lock().unwrap().len(), 1);
    assert_eq!(long.requests.lock().unwrap().len(), 1);
    assert!(
        busy_with_unknown_acceptance
            .requests
            .lock()
            .unwrap()
            .is_empty()
    );

    gateway.record_owner(Pool::Standard, session.scoped, 2);
    let result = request(&client, &server, "one").send().await.unwrap();
    assert_eq!(result.status(), StatusCode::SERVICE_UNAVAILABLE);
    let _ = result.bytes().await.unwrap();
    assert_eq!(
        busy_with_unknown_acceptance.requests.lock().unwrap().len(),
        1
    );
    assert_eq!(reject.requests.lock().unwrap().len(), 1);
    assert_eq!(long.requests.lock().unwrap().len(), 1);
}

#[test]
fn classifies_output_and_image_request_shapes_without_clamping() {
    assert_eq!(
        classify(
            "/v1/chat/completions",
            &json!({"messages":[{"role":"system"},{"role":"user"}], "max_completion_tokens":8192, "max_tokens":9000}),
            8192,
            true
        ),
        (false, Pool::Standard, 8192)
    );
    assert_eq!(
        classify(
            "/v1/chat/completions",
            &json!({"messages":[{"role":"user"},{"role":"assistant"}]}),
            8192,
            true
        ),
        (true, Pool::Long, 256_000)
    );
    assert_eq!(
        classify("/v1/completions", &json!({"prompt":[1,2]}), 8192, true),
        (false, Pool::Standard, 16)
    );
    let image = json!({"input":[{"type":"input_image"}], "max_output_tokens":100});
    assert_eq!(
        classify("/v1/responses", &image, 8192, true),
        (false, Pool::Standard, 100)
    );
    assert_eq!(
        classify("/v1/responses", &image, 8192, false),
        (false, Pool::Long, 100)
    );
}

#[tokio::test]
async fn prefers_stage_work_and_kv_headroom_over_legacy_counts() {
    let mut low_count = Mock::new(StatusCode::OK, None);
    low_count.capacity["work"] = json!({"prefill_ms":115,"restore_ms":400,"decode_ms":60});
    let mut low_headroom = Mock::new(StatusCode::OK, None);
    low_headroom.capacity["work"] = json!({"prefill_ms":20,"restore_ms":20,"decode_ms":20});
    low_headroom.capacity["kv_free_bytes"] = json!(10);
    low_headroom.capacity["kv_reserved_bytes"] = json!(90);
    let mut free = Mock::new(StatusCode::OK, None);
    free.capacity["work"] = low_headroom.capacity["work"].clone();
    free.capacity["active"] = json!(2);
    free.capacity["kv_free_bytes"] = json!(90);
    free.capacity["kv_reserved_bytes"] = json!(10);
    let (first, second, third) =
        tokio::join!(low_count.serve(), low_headroom.serve(), free.serve());
    let (_, server) = gateway(config(vec![
        backend("low-count", &first, Pool::Standard),
        backend("low-headroom", &second, Pool::Standard),
        backend("free", &third, Pool::Standard),
    ]))
    .await;

    let response = request(&reqwest::Client::new(), &server, "one")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.text().await.unwrap(), "data: expected\n\n");
    assert_eq!(free.requests.lock().unwrap().len(), 1);
    assert!(low_count.requests.lock().unwrap().is_empty());
    assert!(low_headroom.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn chooses_ready_replica_over_late_affinity_without_rejecting_on_estimates() {
    let mut warm = Mock::new(StatusCode::OK, None);
    warm.capacity["work"] = json!({"prefill_ms":20,"restore_ms":200,"decode_ms":0});
    let mut cold = Mock::new(StatusCode::OK, None);
    cold.capacity["work"] = json!({"prefill_ms":100,"restore_ms":40,"decode_ms":0});
    let (first, second) = tokio::join!(warm.serve(), cold.serve());
    let (gateway, _) = gateway(config(vec![
        backend("warm", &first, Pool::Standard),
        backend("cold", &second, Pool::Standard),
    ]))
    .await;
    let session = [1; 32];
    gateway.record_owner(Pool::Standard, session, 0);
    for (remaining, excluded, expected) in [
        (300, &[][..], 0),
        (180, &[][..], 1),
        (100, &[][..], 1),
        (100, &[1][..], 0),
    ] {
        let selected = gateway.select(
            0,
            Pool::Standard,
            "test",
            session,
            excluded,
            Duration::from_millis(remaining),
        );
        assert_eq!(selected.unwrap().backend, expected);
    }
}

#[test]
fn malformed_work_does_not_drop_authoritative_admission_counts() {
    let capacity: Capacity = serde_json::from_value(json!({
        "epoch":"test",
        "queued":4,
        "active":0,
        "available":0,
        "work":{"prefill_ms": -1}
    }))
    .unwrap();
    assert_eq!(capacity.available, 0);
    assert_eq!(capacity.queued, 4);
    assert!(capacity.work.is_none());
}

#[tokio::test]
async fn overrides_vllm_prefix_cache_salt_by_tenant_and_branch() {
    let mock = Mock::new(StatusCode::OK, None);
    let replica = mock.serve().await;
    let mut upstream = backend("vllm", &replica, Pool::Standard);
    upstream.probe_capacity = false;
    upstream.vllm_cache_salt = true;
    let (_, server) = gateway(config(vec![upstream])).await;
    let client = reqwest::Client::new();
    let mut input: Value = serde_json::from_str(COMPLETION).unwrap();
    input["cache_salt"] = json!("untrusted");

    for (tenant, branch) in [
        ("one", "same"),
        ("one", "same"),
        ("one", "another"),
        ("two", "same"),
    ] {
        let response = client
            .post(server.url.join("/v1/chat/completions").unwrap())
            .bearer_auth(format!("tenant-{tenant}"))
            .header("x-session-id", branch)
            .header(header::CONTENT_TYPE, "application/json")
            .body(input.to_string())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["x-session-id"], branch);
        assert_eq!(response.text().await.unwrap(), "data: expected\n\n");
    }

    let requests = mock.requests.lock().unwrap();
    let mut observed_salts = Vec::new();
    for (_, raw) in requests.iter() {
        let mut body: Value = serde_json::from_slice(raw).unwrap();
        let salt = body.as_object_mut().unwrap().remove("cache_salt").unwrap();
        let salt = salt.as_str().unwrap();
        assert_eq!(salt.len(), 64);
        assert!(salt.bytes().all(|b| b.is_ascii_hexdigit()));
        observed_salts.push(salt.to_owned());
        body["cache_salt"] = json!("untrusted");
        assert_eq!(body, input);
    }
    assert_eq!(observed_salts.len(), 4);
    assert_eq!(observed_salts[0], observed_salts[1]);
    assert_ne!(observed_salts[0], observed_salts[2]);
    assert_ne!(observed_salts[0], observed_salts[3]);
}
