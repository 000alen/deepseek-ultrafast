import json
from pathlib import Path
import tempfile
import unittest

from bench_coding import endpoint_identity, load_inventory
from collect_fleet_inventory import collect


class FleetInventoryTests(unittest.TestCase):
    def test_records_the_gpu_assignment_epoch_and_resolved_width(self):
        pod = {
            "metadata": {"name": "dsuf-standard-0", "uid": "pod-id"},
            "spec": {"nodeName": "gpu-host", "containers": [{
                "name": "serve",
                "args": ["--gpus", "0,1", "--max-num-seqs", "64", "--max-num-decode-seqs", "32"],
                "resources": {"limits": {"nvidia.com/gpu": "2"}},
            }]},
            "status": {"containerStatuses": [{
                "name": "serve", "ready": True,
                "imageID": "repo/image@sha256:" + "a" * 64,
            }]},
        }
        startup = json.dumps({
            "event": "packed_decode_ready", "max_residents": 64, "max_decode_width": 16,
            "slot_bytes_per_rank": 100, "kv_bytes_per_rank": 10000,
        })
        commands = {
            ("get", "pods", "-l", "app=dsuf-engine", "-o", "json"): json.dumps({"items": [pod]}),
            ("exec", "dsuf-standard-0", "-c", "serve", "--", "nvidia-smi",
             "--query-gpu=uuid,name", "--format=csv,noheader"):
                "GPU-00000000-0000-0000-0000-000000000001, NVIDIA B300\n"
                "GPU-00000000-0000-0000-0000-000000000002, NVIDIA B300\n",
            ("get", "--raw", "/api/v1/namespaces/test/pods/dsuf-standard-0:8000/proxy/internal/admission"):
                '{"epoch":"actual-epoch"}',
            ("logs", "dsuf-standard-0", "-c", "serve"): startup,
        }
        inventory = collect(lambda *command: commands[command], "test",
                            "https://bench.example/v1/chat/completions", "native", "a" * 40, "B300")
        replica = inventory["replicas"][0]
        self.assertEqual(replica["topology"], {"backend": "native", "stage": "combined", "tp": 2, "dp": 1, "ep": 1})
        self.assertEqual(replica["epoch"], "actual-epoch")
        self.assertEqual(replica["decode_width"], 16)
        self.assertEqual(replica["host"], "gpu-host")
        self.assertEqual(len(replica["gpu_uuids"]), 2)

    def test_vllm_collector_output_is_accepted_by_production_replay(self):
        pod = {
            "metadata": {"name": "dsuf-standard-0", "uid": "pod-id"},
            "spec": {"nodeName": "gpu-host", "containers": [{
                "name": "serve", "args": ["--data-parallel-size", "2", "--enable-expert-parallel"],
                "resources": {"limits": {"nvidia.com/gpu": "2"}},
            }]},
            "status": {"containerStatuses": [{
                "name": "serve", "ready": True, "restartCount": 3,
                "imageID": "repo/image@sha256:" + "a" * 64,
            }]},
        }
        commands = {
            ("get", "pods", "-l", "app=dsuf-engine", "-o", "json"): json.dumps({"items": [pod]}),
            ("exec", "dsuf-standard-0", "-c", "serve", "--", "nvidia-smi",
             "--query-gpu=uuid,name", "--format=csv,noheader"):
                "GPU-00000000-0000-0000-0000-000000000001, NVIDIA B300\n"
                "GPU-00000000-0000-0000-0000-000000000002, NVIDIA B300\n",
        }
        endpoint = "https://bench.example/v1/chat/completions"
        inventory = collect(lambda *command: commands[command], "test", endpoint,
                            "vllm", "a" * 40, "B300")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inventory.json"
            path.write_text(json.dumps(inventory))
            loaded = load_inventory(path, endpoint_identity(endpoint))
        self.assertEqual(loaded["gpu_count"], 2)
        self.assertIsNone(loaded["replicas"][0]["decode_width"])
        self.assertEqual(loaded["replicas"][0]["epoch"], "pod-id:3")


if __name__ == "__main__":
    unittest.main()
