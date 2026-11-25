# Single-node text SFT and model publication

The launcher takes a Ray job JSON whose entrypoint runs
`python -m relax.entrypoints.train`. Keep the model, training settings and
runtime environment in that JSON. The entrypoint needs `--save` and
`--save-hf`. A new run writes both outputs under its own run directory.

On a QS node, start with `--dry-run` to review the paths and the target model.
The launcher uses the installed `model_tools` uploader. By default it reads
`MODEL_STORE_USER` and `MODEL_STORE_TOKEN`. Use `--publisher-user-env` and
`--publisher-token-env` to refer to the credential variables already present
in your environment.

```bash
python scripts/training/sft/relax_sft_train_and_publish.py \
  --job-spec "$JOB_SPEC" \
  --run-dir "$RUN_DIR" \
  --prepare-dir "$LOCAL_CACHE_DIR" \
  --model-name "$DESTINATION_MODEL" \
  --dry-run
```

Remove `--dry-run` to run the workflow. With `--prepare-dir`, the job spec
must point `--prompt-data` at an absolute JSONL path and `--hf-checkpoint`
at a local Hugging Face model directory. Use plain OpenAI Chat `messages`,
Megatron THD, TP/PP/CP of 1, one epoch, and `--max-tokens-per-gpu 16384`.
The CPU producer prepares batches while training consumes them. Use a
node-local cache with room for the source, indices and prepared batches.

Omit `--prepare-dir` to use the data settings already present in the job spec.
Omit `--run-dir` to wait for the existing job ID from the spec and publish its
final HF export. The launcher checks the final checkpoint tracker and all
referenced weight shards before publishing.

To reproduce the tested training settings, use `--global-batch-size 8192`
and `--manual-gc --manual-gc-interval 1000`. Omit the activation recompute,
loss recompute and offload options.

For a separate TransferQueue comparison, use the normal data path and set
`--sft-compact-tq-text`. Supported plain-text batches use contiguous tensors
with SimpleStorage. Other batches use the regular representation.
