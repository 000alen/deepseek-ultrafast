use super::*;
use axum::extract::rejection::BytesRejection;
use deepseek_recipe::anthropic::messages::response::{MessagesResponse, MessagesStopReason, MessagesStreamEvent};

pub(super) struct MessagesError(ApiError);
impl From<ApiError> for MessagesError {
    fn from(error: ApiError) -> Self {
        Self(error)
    }
}
fn error_value(status: StatusCode, message: String) -> Value {
    let kind = match status {
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        StatusCode::SERVICE_UNAVAILABLE => "overloaded_error",
        StatusCode::PAYLOAD_TOO_LARGE => "request_too_large",
        _ if status.is_server_error() => "api_error",
        _ => "invalid_request_error",
    };
    json!({"type":"error","error":{"type":kind,"message":message}})
}
impl IntoResponse for MessagesError {
    fn into_response(self) -> Response {
        let admission = self.0.admission();
        let mut response = (self.0.0, axum::Json(error_value(self.0.0, self.0.1))).into_response();
        if let Some(value) = admission {
            response.headers_mut().insert(ADMISSION_HEADER, axum::http::HeaderValue::from_static(value));
        }
        response
    }
}
pub(super) fn log_reason(trace: &Trace, reason: Option<&MessagesStopReason>) {
    if let Some(reason) = reason {
        trace.reason(match reason {
            MessagesStopReason::EndTurn | MessagesStopReason::StopSequence => "stop",
            MessagesStopReason::MaxTokens => "length",
            MessagesStopReason::ToolUse => "tool_calls",
            MessagesStopReason::Refusal => "content_filter",
        });
    }
}
pub(super) async fn create_message(
    State(app): State<Arc<App>>,
    Extension(trace): Extension<Trace>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Result<Response, MessagesError> {
    let ingress = ingress(&headers)?;
    let body = body.map_err(|e| ApiError(e.status(), e.body_text()))?;
    let value = request_body(&body)?;
    let seed: Option<i64> = serde_json::from_value(value.get("seed").cloned().unwrap_or(Value::Null)).map_err(bad)?;
    require_session_for_multi_turn(&ingress, conversation_items(&value["messages"]))?;
    let model = app.model.clone();
    let prepared = tokio::task::spawn_blocking(move || crate::messages::prepare(value, &model))
        .await
        .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map_err(bad)?;
    let request = prepared.request;
    let sampling = Sampling {
        max_tokens: request.inference_options.max_tokens.map(|n| n as usize),
        stream: request.stream,
        seed,
        ..Default::default()
    };
    let (tokens, images) = prepare_prompt(&app, prepared.prompt, prepared.literal_breaks, prepared.image_offsets, prepared.image_sources).await?;
    trace.parsed(tokens.len());
    let g = submit(&app, RequestContext { trace: trace.clone(), ingress }, tokens, &sampling, None, images)?;
    let headers = session_headers(&g);
    let failed = g.failed.clone();
    let (id, model) = (rand_id("msg_"), app.model.clone());
    let chunks = crate::messages::events(request, g.into_inference(), id.clone(), model.clone());
    if sampling.stream {
        let (tx, rx) = mpsc::channel::<Value>(64);
        tokio::spawn(async move {
            let mut chunks = std::pin::pin!(chunks);
            loop {
                let next = tokio::select! { _ = tx.closed() => return, next = chunks.next() => next };
                let Some(chunk) = next else { break };
                match response_chunk(chunk, &failed, "message") {
                    Ok(event) => {
                        if let MessagesStreamEvent::MessageDelta { delta, .. } = &event {
                            log_reason(&trace, delta.stop_reason.as_ref());
                        }
                        if tx.send(serde_json::to_value(event).expect("recipe event is JSON")).await.is_err() {
                            return;
                        }
                    }
                    Err(error) => {
                        trace.reason("error");
                        let _ = tx.send(error_value(error.0, error.1)).await;
                        return;
                    }
                }
            }
            if let Some(error) = generation_error(&failed, "message") {
                let _ = tx.send(error_value(error.0, error.1)).await;
            }
        });
        let events = tokio_stream::wrappers::ReceiverStream::new(rx)
            .map(|value| Ok::<_, Infallible>(SseEvent::default().event(value["type"].as_str().expect("Messages event type")).data(value.to_string())));
        return Ok((headers, Sse::new(events).keep_alive(KeepAlive::default())).into_response());
    }
    let mut response = MessagesResponse::new(id, model, now(), 0, 0);
    let mut chunks = std::pin::pin!(chunks);
    while let Some(chunk) = chunks.next().await {
        response.append(response_chunk(chunk, &failed, "message")?);
    }
    if let Some(error) = generation_error(&failed, "message") {
        return Err(error.into());
    }
    log_reason(&trace, response.stop_reason.as_ref());
    Ok((headers, axum::Json(response)).into_response())
}
