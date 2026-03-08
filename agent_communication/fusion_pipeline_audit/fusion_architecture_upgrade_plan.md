# Fusion Architecture Upgrade Plan

## Objective

Upgrade the fusion pipeline from a minimal 21-parameter probability-level fusion to a proper multi-strategy fusion architecture, then search over fusion strategies and parameters to find the best approach. All changes are in the fusion audit script only — no modifications to production code (`src/`).

## Critical Finding

The current `create_multimodal_model` (builders.py:464-476) for the 3-modality case (metadata + 2 images) does:
1. Collapse depth_rgb features (128-dim) + thermal_map features (32-dim) → concat → `image_classifier` Dense(3, softmax) → 3 probs
2. Concat RF probs (3) + image probs (3) → `output` Dense(3, softmax) → final output

**Total trainable fusion params: 21 (6×3 + 3 biases)**

`create_fusion_layer` with cross-modal attention queries is defined but **never called** in the 3-modality path. The patched attention parameters (R3-R5 search rounds) therefore have zero effect.

## Environment

- Python venv: `/venv/multimodal/bin/python`
- Working directory: `/workspace/DFUMultiClassification`
- GPU: multi-GPU via `setup_device_strategy(mode='multi')`
- Run command: `/venv/multimodal/bin/python agent_communication/fusion_pipeline_audit/fusion_hparam_search.py --fresh`

## Files to Modify

1. **`agent_communication/fusion_pipeline_audit/fusion_hparam_search.py`** — the only file modified

Production code (`src/models/builders.py`, etc.) is NOT touched. The fusion search script already monkey-patches `create_multimodal_model` via `build_fusion_model()`. All new fusion strategies are implemented inside this function.

## Feature Dimensions (from standalone configs)

- metadata (RF probs): 3
- depth_rgb projection head output: 128 (head_units=[128])
- thermal_map projection head output: 32 (head_units=[128, 32], last layer = 32)

## Architecture Changes

### Change A: Replace `build_fusion_model` with strategy-based builder

Add a `fusion_strategy` field to `FusionSearchConfig` with values:
- `"prob_concat"` — current behavior (baseline): image features → Dense(3) → concat with RF → Dense(3). 21 params.
- `"feature_concat"` — fuse at feature level: concat [rf_probs(3) + depth_features(128) + thermal_features(32)] → fusion head → Dense(3).
- `"feature_concat_attn"` — feature concat + cross-modal attention (activates the attention mechanism that was never used).
- `"gated"` — learned per-class gate between RF and image predictions.
- `"hybrid"` — concat [rf_probs(3) + image_features(160) + image_probs(3)] → fusion head → Dense(3).

The `build_fusion_model` function is rewritten to:
1. Build image branches and metadata branch using the existing `create_multimodal_model` internals (via `create_image_branch`, `create_metadata_branch`)
2. Instead of calling `create_multimodal_model`, wire the branches together according to `fusion_strategy`
3. No monkey-patching needed — build directly using Keras functional API

### Change B: Fusion head configuration

Add to `FusionSearchConfig`:
- `fusion_head_units: list` — Dense layer sizes between concat and output. Default `[]` (direct Dense(3) like current). Examples: `[32]`, `[64, 32]`.
- `fusion_head_dropout: float` — dropout rate in fusion head. Default `0.3`.
- `fusion_head_bn: bool` — whether to use BatchNorm in fusion head. Default `True`.

### Change C: Temperature scaling

Add to `FusionSearchConfig`:
- `image_temperature: float` — temperature for image softmax (1.0 = normal). Applied before fusion in strategies that use image probabilities.

### Change D: Cross-modal attention (for `feature_concat_attn` strategy)

Re-use the existing attention config fields that were previously unused:
- `fusion_query_dim` — query projection dimension
- `fusion_query_l2` — L2 regularization on query projection
- `meta_query_scale` — scaling factor when metadata queries images
- `image_query_scale` — scaling factor when images query metadata

These fields already exist in `FusionSearchConfig` and were already searched in R3-R5 but had no effect. Now they'll actually be wired in.

### Change E: Gated fusion

For `fusion_strategy="gated"`:
- `gate_hidden_dim: int` — hidden dimension for gate network. Default `32`.

Gate mechanism: concat all features → Dense(gate_hidden_dim, relu) → Dense(3, sigmoid) → per-class gate.
Output: gate * rf_probs + (1 - gate) * image_probs

## New Search Rounds

Replace R3-R5 (which had no effect) and add new rounds. Keep R1 (fusion LR), R2 (epochs), R6 (RF params), R7 (stage2 fine-tuning) as-is.

### New R3: Fusion Strategy

Test all 5 fusion strategies with empty fusion head (minimal params per strategy):
- `prob_concat` — baseline (current)
- `feature_concat` — feature-level, head=[]
- `feature_concat_attn` — feature-level + attention, head=[]
- `gated` — gated fusion
- `hybrid` — all signals combined, head=[]

### New R4: Fusion Head Architecture

Using the best strategy from R3, test fusion head depth/width:
- `head_none` — no hidden layers (direct to Dense(3))
- `head_32` — [32]
- `head_64` — [64]
- `head_64_32` — [64, 32]
- `head_128_64` — [128, 64]

Also test dropout (0.1, 0.3, 0.5) and BN (on/off) for the best head size.

### New R5: Fusion-Specific Parameters

Depends on which strategy won R3:
- If `feature_concat_attn`: search attention params (query_dim, L2, asymmetry) — same as old R4-R5 but now actually effective
- If `gated`: search gate_hidden_dim (16, 32, 64)
- If `prob_concat`/`hybrid`: search temperature scaling (0.5, 1.0, 1.5, 2.0)
- All strategies: search metadata confidence scaling (old R3 variants)

### R6 and R7: Keep as-is

RF hyperparameters and Stage 2 fine-tuning remain unchanged.

## Implementation Steps

### Step 1: Update `FusionSearchConfig` dataclass

Add new fields: `fusion_strategy`, `fusion_head_units`, `fusion_head_dropout`, `fusion_head_bn`, `image_temperature`, `gate_hidden_dim`.

### Step 2: Rewrite `build_fusion_model`

Replace the current monkey-patching approach with direct Keras functional API construction:
1. Use `create_image_branch` and `create_metadata_branch` from `src/models/builders` to build branches (with `patched_get_modality_config` for standalone configs)
2. Switch on `cfg.fusion_strategy` to wire branches together
3. Build fusion head from `cfg.fusion_head_units`
4. Return the complete Model

### Step 3: Update `pick_best` to reconstruct new fields

Add the new config fields to the `pick_best` function's `FusionSearchConfig` reconstruction.

### Step 4: Update result dict in `train_single_config`

Add new fields to the result dict saved to CSV.

### Step 5: Update `load_completed_results` type conversion

Add new fields to the float/int/bool/string conversion lists.

### Step 6: Rewrite R3, R4, R5 search round functions

Replace the old ineffective rounds with the new strategy/head/parameter searches.

### Step 7: Update `main()` round wiring

Wire R3→R4→R5 to use the new round functions. R1, R2, R6, R7 unchanged.

### Step 8: Update `print_config_summary` in main()

Add new fields to the summary printout so results are readable.

### Step 9: Syntax check and verify

Run `py_compile` on the modified script. Verify all new `FusionSearchConfig` fields have defaults so existing CSV resume still works.

## Level of Care

- **Variable names**: All new config fields use snake_case matching existing conventions. `fusion_strategy`, not `fusionStrategy`.
- **Backward compatibility**: All new fields have defaults matching current behavior (`fusion_strategy="prob_concat"`, `fusion_head_units=[]`, etc.) so old CSV results can still be loaded and old configs still work.
- **No omissions**: Every new field must appear in: (1) `FusionSearchConfig` dataclass, (2) `build_fusion_model`, (3) result dict, (4) `pick_best` reconstruction, (5) `load_completed_results` type conversion, (6) `print_config_summary`.
- **No production code changes**: Everything is self-contained in the fusion audit script.
- **Determinism**: Seed management unchanged — the existing seed reset before pre-training is preserved.
- **Weight loading**: The standalone weight loading added earlier is strategy-agnostic (loads backbone weights regardless of fusion strategy). No changes needed.
