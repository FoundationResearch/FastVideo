# Exploration Log: DCP Checkpoint To Diffusers Export

## Status: draft

## Context
Convert a training checkpoint in DCP format into a diffusers-style model
directory for inference/reuse. Task instance:
`/mnt/data/world_model/zelda_runs/wangame_self_forcing_zelda/checkpoint-best-step-5600`.

## Progress
- [x] Step 1: Inspect checkpoint layout and confirm it contains `metadata.json`
  plus `dcp/` shards.
- [x] Step 2: Locate existing exporter in
  `fastvideo/train/entrypoint/dcp_to_diffusers.py`.
- [x] Step 3: Read checkpoint metadata to determine the base model and export
  role.
- [x] Step 4: Run exporter on one GPU and write a sibling diffusers directory.
- [x] Step 5: Verify the output contains `model_index.json` and a rewritten
  `transformer/model.safetensors`.

## Findings
- The correct exporter is:
  `python -m fastvideo.train.entrypoint.dcp_to_diffusers`.
- The exporter can reconstruct config from `metadata.json`; no explicit YAML was
  needed for this checkpoint.
- For the Zelda self-forcing checkpoint, the correct base model came from:
  `/mnt/data/world_model/SFWanGame-2.1-0326-2k-steps`.
- Successful command:
  `CUDA_VISIBLE_DEVICES=0 python -m fastvideo.train.entrypoint.dcp_to_diffusers --checkpoint /mnt/data/world_model/zelda_runs/wangame_self_forcing_zelda/checkpoint-best-step-5600 --output-dir /mnt/data/world_model/zelda_runs/wangame_self_forcing_zelda/checkpoint-best-step-5600-diffusers`
- The output directory was a full diffusers-style repo with
  `model_index.json`, copied tokenizer/encoders/VAE assets, and a rewritten
  `transformer/model.safetensors`.

## Mistakes / Dead Ends
- Running the exporter inside the sandbox failed because Triton/torch could not
  find an active CUDA driver (`RuntimeError: 0 active drivers ([])`).
- `python -m fastvideo.train.entrypoint.dcp_to_diffusers --help` imports enough
  of `fastvideo` to hit the same CUDA/Triton issue in the sandbox, so validation
  should use an unsandboxed run when GPU-backed imports are required.

## Proposed Standardization
Add a workflow document for "export DCP checkpoint to diffusers" covering:
- Required checkpoint files.
- How role selection works.
- Recommended output naming convention.
- The need for an unsandboxed GPU-visible environment.
