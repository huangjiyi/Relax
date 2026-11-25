# Single-node text SFT and model publication

The Qwen3-0.6B recipe is a repository training script with model, data,
checkpoint, optimizer, and performance flags in Bash arrays. It builds the
Ray submission from those arguments.

```bash
MODEL_DIR="$MODELS" DATA_DIR="$DATASETS" SAVE_DIR="$CHECKPOINTS" \
  bash scripts/training/sft/run-qwen3-0.6B-8xgpu-prepared-sft.sh --dry-run
```

Set `MODEL_DIR` to the HF model directory or its parent with a
`Qwen3-0.6B` subdirectory. Put the OpenAI Chat JSONL in
`$DATA_DIR/train.jsonl`. Use `DATA_FILE` to select a different file under
`DATA_DIR`. Use `LOCAL_CACHE_DIR` to choose node-local storage with room for
the source, indices, and prepared batches. The default uses the system
temporary directory. Remove `--dry-run` to launch training.

The recipe saves checkpoints and HF weights under `$SAVE_DIR/$EXP_NAME`.
For a new experiment, choose a new `EXP_NAME`. Once a previous attempt has
ended, set a new `ATTEMPT_ID` with the same save directory and experiment
name to start another Ray job with the same native checkpoint load path.
Set `DATA_PIPELINE=transfer-queue` to select the optional compact
TransferQueue comparison.

Set `PUBLISH_MODEL_NAME` on a QS node to publish the final HF export. The
launcher uses the installed `model_tools` uploader and defaults to
`MODEL_STORE_USER` and `MODEL_STORE_TOKEN`. Set `PUBLISH_USER_ENV` and
`PUBLISH_TOKEN_ENV` to select other credential variables.

## Launch or publish from a saved Ray spec

The Python helper also accepts a Ray job JSON whose entrypoint runs
`python -m relax.entrypoints.train` with `--save` and `--save-hf`. Set
`--publisher-user-env` and `--publisher-token-env` if you invoke it directly
with a custom authentication environment.

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
