# Diagram Descriptions for Paper Figures

Instructions for an AI diagram generation agent. Each figure description below includes the layout, components, connections, labels, and styling. The diagrams should be clean, professional, suitable for an academic journal. Use a white background, consistent font sizes, and muted professional colors (blues, greens, oranges). Avoid excessive decoration. All text should be legible at print size.

---

## Figure 1: DFU-MFNet Framework Overview

**Purpose:** High level system diagram showing the complete pipeline from raw data to final prediction. This is the first figure the reader sees, so it should communicate the overall approach at a glance.

**Layout:** Horizontal flow, left to right, with three major blocks.

### Block 1 (Left): Data Inputs

Four parallel input streams entering from the left, stacked vertically:

1. **Clinical Metadata** (blue rounded rectangle)
   - Icon or small table symbol
   - Label: "Clinical Metadata (72 features)"
   - Subtitle: "Demographics, wound scores, temperatures"

2. **RGB Images** (orange rounded rectangle)
   - Small sample wound image placeholder (or camera icon)
   - Label: "Depth RGB (1280x720)"
   - Subtitle: "Bounding box crop, resize to 128x128"

3. **Thermal Images** (green rounded rectangle)
   - Small thermal colormap placeholder (or thermometer icon)
   - Label: "Thermal Map (1080x1440)"
   - Subtitle: "Bounding box crop, resize to 128x128"

4. **Depth Images** (purple rounded rectangle)
   - Small depth colormap placeholder
   - Label: "Depth Map (1280x720)"
   - Subtitle: "Bounding box crop, resize to 128x128"

A bracket or brace on the left labeled "4 Modalities" groups these.

### Block 2 (Center): Processing Branches

Each input connects to its processing branch via an arrow:

1. **Metadata Branch** (blue)
   - Box 1: "Random Forest (300 trees)"
   - Box 2: "OOF Probabilities (3 dim)"
   - Arrow from Clinical Metadata to Box 1 to Box 2
   - Small label: "No trainable neural parameters"

2. **RGB Branch** (orange)
   - Box 1: "DenseNet121 (ImageNet)"
   - Box 2: "Projection Head [128, 32]"
   - Arrow from RGB Images to Box 1 to Box 2
   - Small label: "Two stage training"

3. **Thermal Branch** (green)
   - Box 1: "DenseNet121 (ImageNet)"
   - Box 2: "Projection Head [128]"
   - Arrow from Thermal Images to Box 1 to Box 2

4. **Depth Branch** (purple)
   - Box 1: "DenseNet121 (ImageNet)"
   - Box 2: "Projection Head [128]"
   - Arrow from Depth Images to Box 1 to Box 2

Between the RGB branch input and its DenseNet121, add a small dashed box labeled "SDXL Generative Augmentation (15%)" with a dashed arrow injecting into the RGB data stream. This indicates synthetic images are mixed in during training only.

### Block 3 (Right): Fusion and Ensemble

All four branch outputs converge into:

1. **Feature Concatenation** (large rounded rectangle, light gray)
   - Label: "Feature Concatenation"
   - Shows the four feature vectors being concatenated
   - Annotation: "RF probs (3) + Image features (variable)"

2. **Dense Output** (small rectangle)
   - Label: "Dense(3, softmax)"
   - Arrow to three class outputs: I, P, R

3. **Below the fusion block**, show the ensemble:
   - A bracket grouping "15 Modality Combinations" (with a note: "all subsets of 4 modalities")
   - Arrow from all combinations to:
   - **Simple Average Ensemble** (rounded rectangle)
   - Label: "Average predictions from metadata containing combinations"
   - Arrow to final output: "Final Prediction (I / P / R)"

### Additional Elements

- A horizontal dashed line separates "Per Combination Training" (top) from "Ensemble" (bottom)
- Title at top: "DFU-MFNet: Multimodal Fusion Network for DFU Healing Phase Classification"
- Small legend in corner showing: Blue = Metadata, Orange = RGB, Green = Thermal, Purple = Depth

---

## Figure 3: DFU-MFNet Detailed Architecture

**Purpose:** Detailed technical diagram showing the internal architecture of a single multimodal combination (specifically metadata + RGB + thermal, the best performing combination). This complements Figure 1 by showing tensor shapes and layer details.

**Layout:** Vertical flow, top to bottom, with three parallel columns merging at the bottom.

### Column 1 (Left): Metadata Branch

Vertical stack of boxes, top to bottom:

1. "Clinical Metadata Input" (shape: batch x 72)
2. Arrow down
3. "Random Forest Classifier" (300 trees, depth 10, 80 MI selected features)
4. Arrow down
5. "OOF Probability Vector" (shape: batch x 3)
6. Label on side: "0 trainable parameters"

### Column 2 (Center): RGB Branch

Vertical stack:

1. "RGB Image Input" (shape: batch x 128 x 128 x 3)
2. Small dashed box to the side: "SDXL GenAug (15% prob, 1-5% mix)" with dashed arrow into input
3. Arrow down
4. "DenseNet121 Backbone (ImageNet)" with annotation "Stage 1: Frozen, Stage 2: Top 5% unfrozen"
5. Arrow down
6. "Global Average Pooling" (shape: batch x 1024)
7. Arrow down
8. "Dense(128) + BatchNorm + Dropout(0.3)"
9. Arrow down
10. "Dense(32) + BatchNorm + Dropout(0.3)" (shape: batch x 32)

### Column 3 (Right): Thermal Branch

Vertical stack (similar to RGB but different head):

1. "Thermal Map Input" (shape: batch x 128 x 128 x 3)
2. Arrow down
3. "DenseNet121 Backbone (ImageNet)" with same Stage 1/2 annotation
4. Arrow down
5. "Global Average Pooling" (shape: batch x 1024)
6. Arrow down
7. "Dense(128) + BatchNorm + Dropout(0.3)" (shape: batch x 128)

### Fusion Section (Bottom, spans all columns)

All three branch outputs converge:

1. Three arrows merge into a wide horizontal bar labeled "Concatenation"
   - Show dimensions: "[3] + [32] + [128] = [163]"
2. Arrow down
3. "Dense(3, softmax)"
4. Arrow down
5. "Output: P(I), P(P), P(R)"

### Training Details (Side Panel or Box)

A boxed annotation on the right side:

```
Training Protocol:
  Stage 1: Backbones frozen
    - LR: 1.72e-3
    - Epochs: up to 500 (early stop, patience 20)
    - Loss: Focal (gamma=2.0, class weighted)

  Stage 2: Top 5% backbone layers unfrozen
    - LR: 5e-6
    - Epochs: 30
    - BatchNorm: Frozen
    - Revert to Stage 1 if no improvement
```

### Tensor Shape Annotations

At each arrow, show the tensor shape in small gray text:
- After backbone: (B, 1024)
- After projection: (B, 32) for RGB, (B, 128) for thermal
- After concat: (B, 163)
- Output: (B, 3)

### Color Scheme

- Metadata path: Blue boxes and arrows
- RGB path: Orange boxes and arrows
- Thermal path: Green boxes and arrows
- Fusion layers: Gray boxes
- GenAug: Dashed orange box
- Frozen layers: Indicated with a small lock icon or hatching pattern
- Trainable layers: Solid fill

### Style Notes

- Keep boxes uniform width within each column
- Use rounded rectangles for neural network layers
- Use sharp rectangles for data/tensors
- Arrows should be clean, no decorative arrowheads
- Font: sans serif, 9-10pt for labels, 7-8pt for annotations
- Total figure width should fit a single column journal format (~3.5 inches) or double column (~7 inches)
