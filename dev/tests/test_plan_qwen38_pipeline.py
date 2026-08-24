import json
import struct

from dev.tools.plan_qwen38_pipeline import MAGIC, PREFIX, plan


def test_plan_balances_weights_and_kv(tmp_path):
    objects = [
        {"name": "text/token_embedding", "kind": "tensor", "bytes": 100},
        {"name": "text/layers/0/gdn/in", "kind": "tensor", "bytes": 60},
        {"name": "text/layers/1/attention/q", "kind": "tensor", "bytes": 40},
        {"name": "text/layers/2/gdn/in", "kind": "tensor", "bytes": 60},
        {"name": "text/layers/3/attention/q", "kind": "tensor", "bytes": 40},
        {"name": "text/final_norm", "kind": "tensor", "bytes": 10},
        {"name": "text/output_head", "kind": "tensor", "bytes": 90},
        {"name": "frontend/tokenizer.json", "kind": "resource", "bytes": 12},
    ]
    directory = json.dumps(
        {"identity": {"model_id": "test", "weights_id": "test"}, "objects": objects}
    ).encode()
    artifact = tmp_path / "tiny.ninfer"
    artifact.write_bytes(PREFIX.pack(MAGIC, len(directory)) + directory)

    result = plan(artifact, context=64, kv_group=64)

    assert result["full_attention_layers"] == [1, 3]
    assert result["kv_bytes_per_full_attention_layer_token"] == 2112
    assert result["recommended"]["stage0_full_attention_layers"] == 1
    assert result["recommended"]["stage1_full_attention_layers"] == 1
    assert result["runtime_performance_default"]["stage0_layers"] == 2
    assert result["runtime_performance_default"]["stage1_layers"] == 2
