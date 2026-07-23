"""Remote policies adopt the SERVED checkpoint's observation contract.

The region-trim incident: a 39-token trimmed server behind ``remote:`` received
87-token default-layout encodes and 400'd every forward — collectors adopted
spec/masks only from local ``neural:`` specs. Contract under test:
  1. /config exposes the served model_config (and keeps it fresh across /reload);
  2. env_config_with_policy_spec_masks(remote:URL) == the same call with
     neural:<ckpt> for the checkpoint the server loaded (symmetric oracle);
  3. /reload refuses a token-shape mismatch (window guard's twin).
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

try:
    import torch  # noqa: F401

    TORCH = True
except Exception:  # pragma: no cover
    TORCH = False

if TORCH:
    from pokezero.collection import env_config_with_policy_spec_masks
    from pokezero.inference_service import fetch_remote_config, serve_inference
    from pokezero.local_showdown import LocalShowdownConfig
    from pokezero.neural_policy import (
        TransformerPolicyConfig,
        TransformerTrainingConfig,
        save_transformer_checkpoint,
        train_transformer_policy,
    )
    from tests.test_neural_policy import rollout_record
    from pokezero.collection import write_rollout_record


def _train_checkpoint(temp_dir: Path, name: str, **config_overrides) -> Path:
    data = temp_dir / f"{name}.jsonl"
    with data.open("w", encoding="utf-8") as handle:
        write_rollout_record(handle, rollout_record())
    values = dict(
        category_vocab=tuple(range(1, 17)), category_oov_buckets=4, policy_id=name,
        window_size=2, token_type_vocab_size=8, categorical_feature_count=1,
        numeric_feature_count=1, embedding_dim=16, transformer_layers=1,
        attention_heads=4, feedforward_dim=32, dropout=0.0,
    )
    values.update(config_overrides)
    model_config = TransformerPolicyConfig.compact_category(**values)
    model, result = train_transformer_policy(
        data, model_config=model_config,
        training_config=TransformerTrainingConfig(batch_size=2, epochs=1, window_size=2, max_batches=1, device="cpu"),
    )
    ckpt = temp_dir / f"{name}.pt"
    save_transformer_checkpoint(ckpt, model, result=result)
    return ckpt


@unittest.skipUnless(TORCH, "requires torch")
class RemoteSelfDescribeTests(unittest.TestCase):
    def _serve(self, ckpt: Path):
        server = serve_inference(str(ckpt), host="127.0.0.1", port=0, device="cpu")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def test_config_exposes_served_model_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt = _train_checkpoint(Path(temp_dir), "served")
            url = self._serve(ckpt)
            payload = fetch_remote_config(url)
            self.assertIn("model_config", payload)
            served = TransformerPolicyConfig.from_dict(payload["model_config"])
            self.assertEqual(served.token_count, TransformerPolicyConfig.from_dict(payload["model_config"]).token_count)
            self.assertEqual(served.policy_id, "served")

    def test_remote_spec_adopts_like_neural_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt = _train_checkpoint(Path(temp_dir), "served")
            url = self._serve(ckpt)
            base = LocalShowdownConfig()
            via_neural = env_config_with_policy_spec_masks(base, [f"neural:{ckpt}"], context="test")
            via_remote = env_config_with_policy_spec_masks(base, [f"remote:{url}"], context="test")
            self.assertEqual(via_neural.observation_spec, via_remote.observation_spec)
            self.assertEqual(via_neural.feature_masks, via_remote.feature_masks)

    def test_reload_refuses_token_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt = _train_checkpoint(Path(temp_dir), "served")
            other = _train_checkpoint(Path(temp_dir), "wider", categorical_feature_count=2)
            url = self._serve(ckpt)
            body = json.dumps({"checkpoint_path": str(other)}).encode()
            request = Request(url + "/reload", data=body, headers={"Content-Type": "application/json"})
            with self.assertRaises(HTTPError) as caught:
                urlopen(request, timeout=30)
            self.assertEqual(caught.exception.code, 400)
            self.assertIn("observation contract", caught.exception.read().decode())
            # still serving the original policy
            self.assertEqual(fetch_remote_config(url)["policy_id"], "served")


if __name__ == "__main__":
    unittest.main()
