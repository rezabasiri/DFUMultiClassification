# Generative Augmentation Improvement - Action Plans
## Ranked by Effectiveness (High → Low)

Generated: 2026-01-19

---

## 📊 Current Status
- **Baseline Kappa**: 0.2867
- **With Gen Aug**: 0.2569 (-10.4% worse)
- **Issue**: Generated images are degrading model performance

---

## 🎯 Action Plans (Ranked by Expected Impact)

### **Tier 1: Critical - Do These First (Highest ROI)**

#### Plan 1.1: Visual Quality Inspection & Validation ⭐⭐⭐⭐⭐
**Expected Impact**: Critical diagnostic step
**Effort**: Low (2-4 hours)
**Prerequisites**: None

**Objective**: Understand what's being generated and why it's failing

**Steps**:
1. Generate 100 images per phase (I, P, R) using current models
2. Save to `agent_communication/generative_augmentation/generated_samples/`
3. Manually inspect for:
   - Visual realism (do they look like real wounds?)
   - Phase-appropriate features (can you distinguish I vs P vs R?)
   - Artifacts (blurriness, unrealistic colors, distortions)
   - Diversity (are they all similar or varied?)
4. Document findings in `VISUAL_INSPECTION_REPORT.md`

**Success Criteria**:
- Clear understanding of generation quality issues
- Identified specific failure modes
- Decision on whether to fix current models or re-train

**Files to Create**:
- `scripts/generate_sample_images.py` - Generate test images
- `VISUAL_INSPECTION_REPORT.md` - Document findings

---

#### Plan 1.2: Measure Generation Quality Metrics ⭐⭐⭐⭐⭐
**Expected Impact**: Quantify quality issues
**Effort**: Medium (4-6 hours)
**Prerequisites**: Plan 1.1

**Objective**: Get objective quality measurements

**Steps**:
1. Implement quality metrics:
   - **FID** (Fréchet Inception Distance) - Lower is better, <50 is good
   - **SSIM** (Structural Similarity) - Higher is better, >0.7 is good
   - **LPIPS** (Perceptual similarity) - Lower is better
   - **Diversity Score** - Measure feature space coverage
2. Compare generated images vs real images per phase
3. Identify if issues are:
   - Low quality (high FID, low SSIM)
   - Low diversity (mode collapse)
   - Wrong features (LPIPS shows different perceptual features)

**Success Criteria**:
- FID < 50 (currently likely > 100)
- SSIM > 0.7 with real images
- Diversity score shows good spread

**Files to Create**:
- `scripts/measure_quality_metrics.py` - Compute all metrics
- `QUALITY_METRICS_REPORT.md` - Document baseline metrics

---

#### Plan 1.3: Re-train Models with LoRA at 128×128 ⭐⭐⭐⭐⭐
**Expected Impact**: Very High (addresses root cause)
**Effort**: High (16-24 hours including compute time)
**Prerequisites**: Plans 1.1, 1.2

**Objective**: Create high-quality generative models for Phase I and R

**Key Improvements Over Current**:
1. **LoRA** instead of full fine-tuning (less overfitting)
2. **128×128** instead of 64×64 (better detail)
3. **Stable Diffusion v2.1** instead of v1.5 (better base model)
4. **Perceptual loss** + diffusion loss (better quality)
5. **Quality-aware training** (filter low-quality generations during training)
6. **Multi-GPU** training (faster)

**Training Configuration**:
```yaml
Model: Stable Diffusion v2.1 base
Training Method: LoRA (rank=16, alpha=32)
Resolution: 128×128
Batch Size: 8 per GPU (16 total)
Learning Rate: 1e-5 → 1e-6 (cosine schedule)
Epochs: 100 (early stopping patience=10)
Inference Steps: 100 (high quality)
Guidance Scale: 12.0 (strong prompt adherence)
Mixed Precision: FP16
```

**Success Criteria**:
- FID < 50 (vs current likely >100)
- SSIM > 0.75 with real wounds
- Generated images visually indistinguishable from real (by human inspection)

**Files to Create**:
- `scripts/train_lora_model.py` - Main training script
- `configs/training_config.yaml` - All hyperparameters
- `scripts/evaluate_model.py` - Quality evaluation
- `TRAINING_REPORT.md` - Document training progress

**Estimated Time**: 8-12 hours training per phase on 2× RTX 4090

---

#### Plan 1.4: Implement Quality Filtering During Inference ⭐⭐⭐⭐⭐
**Expected Impact**: Very High (prevents bad data from hurting model)
**Effort**: Medium (6-8 hours)
**Prerequisites**: Plan 1.2

**Objective**: Only use high-quality generated images

**Implementation**:
1. Create quality filter class that checks:
   - SSIM > threshold (e.g., 0.7)
   - LPIPS < threshold (e.g., 0.3)
   - Perceptual quality score
2. During training, reject generated images that fail quality check
3. Log rejection rate (if >50% rejected, model quality is poor)
4. Make thresholds configurable

**Success Criteria**:
- Rejection rate < 20% (means most generated images are good)
- Only high-quality images used in training
- Improved training performance

**Files to Create**:
- `src/data/quality_filter.py` - Quality filtering class
- Update `generative_augmentation_v2.py` to use filter
- `configs/quality_thresholds.yaml` - Configurable thresholds

---

### **Tier 2: High Impact - Do These Next**

#### Plan 2.1: Focus on Minority Classes Only ⭐⭐⭐⭐
**Expected Impact**: High (addresses class imbalance)
**Effort**: Low (1-2 hours)
**Prerequisites**: None

**Objective**: Generate only for underrepresented classes

**Implementation**:
```python
# Only generate for Phase I (35%) and Phase R (13%)
# Skip Phase P (52% - already well-represented)
GENERATIVE_AUG_PHASES = ['I', 'R']
```

**Rationale**: Generative augmentation is most effective for minority classes where real data is limited.

**Success Criteria**:
- Better class balance in training data
- Improved performance on Phase R (hardest class)

**Files to Update**:
- `src/utils/production_config.py` - Set GENERATIVE_AUG_PHASES = ['I', 'R']

---

#### Plan 2.2: Increase Guidance Scale ⭐⭐⭐⭐
**Expected Impact**: High (better prompt adherence)
**Effort**: Low (1 hour)
**Prerequisites**: None

**Objective**: Make generated images follow prompts more strictly

**Implementation**:
```python
GENERATIVE_AUG_GUIDANCE_SCALE = 12.0  # Up from 7.5 default

# In pipeline call:
pipeline(..., guidance_scale=GENERATIVE_AUG_GUIDANCE_SCALE)
```

**Rationale**: Higher guidance scale (10-15) forces model to generate more realistic, prompt-aligned wound images.

**Success Criteria**:
- Generated images have more distinct phase-specific features
- Visual inspection shows improvement

**Files to Update**:
- `src/utils/production_config.py` - Add GENERATIVE_AUG_GUIDANCE_SCALE
- `src/data/generative_augmentation_v2.py` - Use in pipeline call

---

#### Plan 2.3: Progressive Augmentation ⭐⭐⭐⭐
**Expected Impact**: High (reduces early training confusion)
**Effort**: Medium (3-4 hours)
**Prerequisites**: None

**Objective**: Gradually introduce synthetic data during training

**Implementation**:
Start with 0% synthetic data, ramp up to target probability over first 25% of epochs.

```python
def get_dynamic_aug_prob(current_epoch, total_epochs):
    ramp_epochs = int(total_epochs * 0.25)  # Ramp over first 25%
    if current_epoch < ramp_epochs:
        return GENERATIVE_AUG_PROB * (current_epoch / ramp_epochs)
    return GENERATIVE_AUG_PROB
```

**Rationale**: Let model learn real data patterns first, then introduce augmentation gradually.

**Success Criteria**:
- Smoother training curves
- Better final performance

**Files to Update**:
- `src/data/generative_augmentation_v2.py` - Implement progressive augmentation

---

#### Plan 2.4: Increase Inference Steps ⭐⭐⭐
**Expected Impact**: Medium-High (better quality)
**Effort**: Low (10 minutes)
**Prerequisites**: None

**Objective**: Generate higher quality images

**Implementation**:
```python
GENERATIVE_AUG_INFERENCE_STEPS = 100  # Up from 50
```

**Rationale**: More denoising steps = higher quality. Cost is minimal since gen aug only applies to small % of batches.

**Success Criteria**:
- Visually improved image quality
- Higher SSIM scores

**Files to Update**:
- `src/utils/production_config.py` - Set to 100

---

### **Tier 3: Medium Impact - Consider These**

#### Plan 3.1: Class-Aware Mixing ⭐⭐⭐
**Expected Impact**: Medium (preserves class distribution)
**Effort**: Medium (4-6 hours)
**Prerequisites**: None

**Objective**: Replace samples only within same class

**Implementation**:
Instead of random batch replacement, only replace Phase I samples with generated Phase I images, etc.

**Rationale**: Prevents accidentally replacing minority class samples with majority class generated images.

**Success Criteria**:
- Class distribution unchanged after augmentation
- No negative impact on class balance

**Files to Update**:
- `src/data/generative_augmentation_v2.py` - Implement class-aware mixing

---

#### Plan 3.2: Increase Mix Ratio ⭐⭐⭐
**Expected Impact**: Medium (if quality is good)
**Effort**: Low (10 minutes)
**Prerequisites**: Plans 1.3, 1.4 (need good quality first)

**Objective**: Use more synthetic data per batch

**Implementation**:
```python
GENERATIVE_AUG_MIX_RATIO = (0.15, 0.25)  # Up from (0.01, 0.05)
```

**Rationale**: If generated images are high quality, using more of them helps model generalize.

**WARNING**: Only do this AFTER verifying generation quality is good!

**Success Criteria**:
- Improved model performance
- No degradation in validation metrics

**Files to Update**:
- `src/utils/production_config.py` - Increase after quality validation

---

#### Plan 3.3: Improve Negative Prompts ⭐⭐
**Expected Impact**: Medium (prevents common failures)
**Effort**: Low (1 hour)
**Prerequisites**: None

**Objective**: Better negative prompts to avoid artifacts

**Implementation**:
```python
negative_prompt = (
    "blurry, out of focus, low quality, jpeg artifacts, "
    "oversaturated, unrealistic colors, smooth skin, healthy tissue, "
    "artificial, cartoon, drawing, painting, text, watermark, "
    "limbs, body parts, face, person, background clutter, "
    "unrealistic wound edges, unnatural textures"
)
```

**Success Criteria**:
- Fewer generation artifacts
- More realistic wound boundaries

**Files to Update**:
- `src/data/generative_augmentation_v2.py` - Update prompt generator

---

### **Tier 4: Lower Impact - Nice to Have**

#### Plan 4.1: Add Data Augmentation to SD Training ⭐⭐
**Expected Impact**: Low-Medium
**Effort**: Medium (4 hours)
**Prerequisites**: Plan 1.3

**Objective**: Increase effective training data diversity

**Implementation**:
Add random flips, rotations, color jitter during SD model training.

**Success Criteria**:
- Better model generalization
- More diverse generated samples

---

#### Plan 4.2: Train at Even Higher Resolution (256×256) ⭐⭐
**Expected Impact**: Low-Medium (diminishing returns)
**Effort**: High (compute intensive)
**Prerequisites**: Plan 1.3 success at 128×128

**Objective**: Capture even finer details

**Warning**: Requires 4× more compute and memory. Only do if 128×128 results are good but need improvement.

---

#### Plan 4.3: Add Mixup Instead of Hard Replacement ⭐
**Expected Impact**: Low
**Effort**: Medium (3-4 hours)
**Prerequisites**: None

**Objective**: Blend real and synthetic images

**Implementation**:
```python
alpha = random.uniform(0.3, 0.7)
mixed = alpha * real + (1 - alpha) * generated
```

**Rationale**: Preserves real image features while adding variation.

---

## 🎬 Recommended Execution Order

### **Week 1: Diagnosis**
1. ✅ Plan 1.1 - Visual Inspection (CRITICAL)
2. ✅ Plan 1.2 - Quality Metrics (CRITICAL)
3. Based on findings, decide: fix current models or re-train?

**Decision Point**: If FID > 100 or SSIM < 0.6 → Re-train (Plan 1.3)

---

### **Week 2: Re-training (If Needed)**
1. ✅ Plan 1.3 - Train LoRA models for Phase I at 128×128
2. ✅ Plan 1.3 - Train LoRA models for Phase R at 128×128
3. ✅ Plan 1.2 - Re-measure quality metrics on new models

**Decision Point**: If FID < 50 and SSIM > 0.7 → Proceed to deployment

---

### **Week 3: Deployment & Optimization**
1. ✅ Plan 1.4 - Implement quality filtering
2. ✅ Plan 2.1 - Focus on minority classes only
3. ✅ Plan 2.2 - Add guidance scale control
4. ✅ Plan 2.3 - Progressive augmentation
5. Test on full training pipeline

---

### **Week 4: Fine-tuning**
1. Adjust hyperparameters based on results
2. Consider Tier 3 plans if needed
3. Run final validation experiments

---

## 📈 Success Metrics

### **Phase 1 (Diagnosis) Success**:
- [ ] Generated 100+ sample images per phase
- [ ] Computed FID, SSIM, LPIPS, diversity metrics
- [ ] Identified root causes of poor performance
- [ ] Clear action plan for improvement

### **Phase 2 (Re-training) Success**:
- [ ] FID < 50 (vs baseline FID > 100)
- [ ] SSIM > 0.75 with real images
- [ ] Visual quality indistinguishable from real (human eval)
- [ ] Diversity score shows good coverage
- [ ] Training completed for Phase I and R

### **Phase 3 (Deployment) Success**:
- [ ] Quality filtering rejects < 20% of generated images
- [ ] Training kappa with gen aug ≥ baseline kappa
- [ ] Ideally: Training kappa > baseline by 5-10%
- [ ] Model performance on Phase R (minority) improved

### **Final Success**:
- [ ] **Kappa improvement > 0% (currently -10.4%)**
- [ ] **Target: +5% to +15% improvement**
- [ ] Generated images pass visual Turing test
- [ ] Reproducible training pipeline

---

## 🔧 Infrastructure Requirements

### **Compute**:
- 2× RTX 4090 GPUs (available ✓)
- ~12-16 GB VRAM per GPU for 128×128 training
- ~64 GB system RAM
- ~500 GB storage for checkpoints and generated samples

### **Software**:
- Python 3.10+
- PyTorch 2.0+
- Diffusers 0.25+
- Accelerate (multi-GPU)
- Transformers
- PEFT (for LoRA)
- torchmetrics (for FID, SSIM)
- lpips (for perceptual similarity)

### **Time Estimates**:
- Phase I training: 8-12 hours
- Phase R training: 8-12 hours
- Quality evaluation: 2-4 hours
- Deployment: 4-6 hours

**Total: ~1-2 weeks calendar time**

---

## 📝 Notes

- **Start with diagnosis** (Plans 1.1, 1.2) before spending time on fixes
- **Re-training** (Plan 1.3) is likely necessary given current poor results
- **Quality filtering** (Plan 1.4) is critical - don't use bad synthetic data
- **Focus on minority classes** (Plan 2.1) is low-effort, high-impact
- **Track everything** - save all metrics, samples, configs for reproducibility

---

## 🚀 Next Steps

1. Review these plans and select priority actions
2. Run Plan 1.1 (visual inspection) immediately
3. Run Plan 1.2 (quality metrics) to quantify issues
4. Based on results, decide on re-training strategy
5. Execute selected plans in priority order

---

**Document Version**: 1.0
**Last Updated**: 2026-01-19
**Status**: Ready for execution
