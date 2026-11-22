# Fleet gateway

`deepseek-ultrafast-gateway` routes Chat Completions, Completions, Responses, Messages, and Models requests to private replica origins. It builds separately from the CUDA server.

```sh
cargo build --locked --release -p deepseek-ultrafast-gateway
target/release/deepseek-ultrafast-gateway gateway.json
```

Mount a config and provide its named credentials through the environment. Keep replica listeners and `/internal/admission` private. Terminate public TLS at the ingress that fronts the gateway.

```json
{
  "listen": "0.0.0.0:8080",
  "max_active": 512,
  "max_pending": 128,
  "max_body_bytes": 67108864,
  "max_buffered_bytes": 536870912,
  "queue_timeout_ms": 2000,
  "upstream_idle_ms": 120000,
  "drain_timeout_ms": 55000,
  "standard_max_output_tokens": 8192,
  "tenants": [
    {
      "id": "coding",
      "token_env": "DSUF_CODING_TOKEN",
      "max_active": 512,
      "max_pending": 128,
      "max_buffered_bytes": 536870912,
      "max_reserved_output_tokens": 4194304
    }
  ],
  "backends": [
    {
      "id": "standard-0",
      "url": "http://standard-0.internal:8000",
      "pool": "standard",
      "cohort": "packed-v1",
      "max_inflight": 32
    },
    {
      "id": "long-0",
      "url": "http://long-0.internal:8000",
      "pool": "long",
      "cohort": "dspark-v1",
      "max_inflight": 8
    }
  ]
}
```

Clients authenticate with `Authorization: Bearer …` or Anthropic `x-api-key`. Replica credentials can use an optional backend `token_env`. The gateway derives the upstream session ID from the authenticated tenant and the client's conversation branch, and returns the original session ID to the client. Credential rotation preserves the namespace when the tenant ID remains the same. Multi-turn requests require a session header. Use distinct branch IDs for parallel agents in a conversation.

The standard pool handles text and image turns within its configured envelope. Set `standard_accepts_images` to `false` for a text-only standard backend so image turns use the long pool. Larger output requests and general API output defaults also route to the long pool. The legacy Completions default of 16 stays eligible for standard routing. The serving tokenizer and renderer determine the exact context size and can reject a turn to the long pool before acceptance. Clients send complete request bodies and keep their requested output limits.

Replicas in the same pool and cohort share session affinity. Configure separate cohorts for incompatible checkpoint fingerprints or backends. A cohort is selected consistently from the session ID even when replicas restart. Among ready replicas in that cohort, routing uses local stream occupancy, reported load and capacity, and affinity to the previous owner. The gateway probes `/readyz` and `/internal/admission`. For an upstream without private capacity telemetry, set `probe_capacity` to `false` and configure `readiness_path` as needed. A vLLM backend can use `/health`. Backend `max_inflight` caps the gateway's dispatches even between probes.

Ultrafast admission reports can include `work.prefill_ms`, `work.decode_ms`, and
`work.restore_ms`. The gateway uses known stage estimates and KV headroom to choose
a replica within the dispatch deadline. An upstream with omitted or partial work
telemetry uses request counts for the missing information.
Packed replicas also expose `work.retirement_bytes` and `work.retirement_ms`
for completion-copy backlog. `kv_data_plane_failures` is the cumulative
count of peer registration, RDMA GET, and snapshot staging failures since
engine startup. A donor outage can still serve via cold recompute, so compare
counter deltas when qualifying cross-host cache reuse.

For a private vLLM text backend, set `vllm_cache_salt` on the backend configuration
to replace a caller-supplied prefix-cache salt with a tenant- and branch-scoped
value. Restrict direct access to the upstream to trusted infrastructure. The
pinned B300 comparison uses Chat Completions, and tenant credential rotation
changes its prefix-cache salt.

Each gateway process bounds stream count, pending requests, and retained request bytes across its own traffic and per tenant. A tenant can also cap outstanding output work through `max_reserved_output_tokens`. This reserves each turn's requested output allowance or model default until its response body completes or disconnects. Replica tokenization and admission enforce the final input and KV limits. The ingest and queue deadline covers the body read and each routing attempt. `upstream_idle_ms` controls the timeout for reading from a replica, including time to headers for a non-streamed response.

The gateway retries connection setup failures and explicit replica admission rejections within the dispatch deadline and attempt budget. It relays ambiguous upstream failures to the client. Accepted streams stay on that replica until the body ends. Overload returns `429`, and absence of a ready replica in the selected cohort returns `503`. On shutdown, readiness drops and admission closes while accepted streams drain within `drain_timeout_ms`.

Use `tools/bench_coding.py` against the gateway to measure the service at the workload and latency targets in the [continuous batching spec](CONTINUOUS_BATCHING.md).
