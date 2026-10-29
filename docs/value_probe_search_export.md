# Frozen-trunk value-head search export

`export_value_probe_search.py` creates two matched FP32 TorchScript leaf models:
original champion value and the A1 validation-selected outcome head. Both keep
the champion's policy logits and opponent-action logits. The source checkpoint
is never changed; root and opponent policies still load that original checkpoint.

The exporter requires the exact champion checksum and completed head-summary
checksum. Architecture selection uses minimum validation MSE, with the fixed A1
kind order for ties. Test metrics do not enter selection. It checks the head
artifact checksum, selected epoch, feature-manifest binding and complete source
checkpoint contract before loading. The actual model is strictly loaded from an
immutable byte snapshot whose checksum matches the contract; missing value-head
parameters refuse instead of producing a fresh head. Nonidentity calibration refuses,
because the native evaluator does not apply it.

The adapter calls the production embedding, encoder, history pool and action
heads. It replaces only the value computation, using the encoder output and the
same attention mask used by the A1 frozen-feature extractor. Validation compares
both eager and reloaded TorchScript outputs against the production forward and
independent encoder-hook features. It covers distinct batch sizes and masked
tokens, checks all three outputs and rejects nonfinite outputs. A supplied saved
observation bundle adds real-input parity, without running a game or search.

```sh
python scripts/export_value_probe_search.py \
  --checkpoint /remote/champion.pt --checkpoint-sha256 CHECKSUM \
  --summary /remote/heads/head-refit-summary.json --summary-sha256 CHECKSUM \
  --device cuda --real-inputs /remote/saved-training-observations.npz \
  --output /remote/new-export
```

The optional observation bundle must contain exactly `categorical_ids`,
`numeric_features`, `token_type_ids`, `attention_mask`, and `history_mask`, in
expanded one-frame window form, with at least 20 observations. The caller must
prospectively freeze its nonholdout membership and input checksum; the exporter
records the checksum of the exact immutable bytes it deserializes. It does not
establish that membership independently.

The five positional inputs and three output tensors match `NativeLeafModel`.
CPU and CUDA traces are separate artifacts; use the artifact on its traced
device. BF16 policy agreement alone does not qualify value precision. TF32 is
disabled for this matched FP32 export.

The exporter disables `torch.backends.mha` fastpath while tracing and validating
both heads. This keeps the fused Transformer/MHA operators out of the saved
graphs, following a bounded CPU/CUDA control that isolated a value discrepancy
in that path. The manifest records `torch_mha_fastpath: false`. A `try/finally`
restores the caller's fastpath setting on success or failure. Existing artifacts
are never overwritten, and eager root/opponent policies need the same
prospectively admitted numerical mode when used with these leaves.

This is a candidate export repair, not CUDA qualification. Both repaired heads
still require the original value tolerance and complete native batch-cardinality
qualification; the earlier failed artifacts and result remain unchanged.

Export completion is preparation only. Actual native/CUDA compatibility, exact
runtime and encoder tables, failure-capture admission, saved-root decision cost
under intended contention, complete roster identity and independent prospective
review are still required before any A4 games. No old results, consumed roots,
partial panels or interrupted cells are resumed by this exporter. Artifacts use
a fresh output directory and record source, model and selection hashes.
