# Raw Data Analysis: Why Depth Map Underperforms

## 1. Image Modality Pixel Characteristics

Analysis of 50 randomly sampled images per modality, both full-frame and within wound bounding boxes.

### 1.1 Full-Frame Image Statistics

| Metric | Depth RGB | Depth Map | Thermal RGB | Thermal Map |
|--------|-----------|-----------|-------------|-------------|
| Resolution | 1280x720 | 1280x720 | 1080x1440 | 1080x1440 |
| Mean pixel | 121.9 | 84.7 | 104.2 | 48.4 |
| Std pixel | 63.2 | 49.8 | 60.4 | 61.7 |
| **Entropy (bits)** | **7.55** | **6.07** | **7.46** | **4.26** |
| **Channel correlation** | **0.951** | **-0.102** | **0.967** | **0.635** |
| Zero fraction | 0.07% | 0.00% | 0.12% | **51.6%** |

### 1.2 Within-Wound Bounding Box Statistics

| Metric | Depth RGB | Depth Map | Ratio |
|--------|-----------|-----------|-------|
| Mean pixel | 80.2 | 75.2 | 1.07x |
| Std pixel | 33.4 | 36.4 | 0.92x |
| **Entropy (bits)** | **6.72** | **3.55** | **1.89x** |
| Dynamic range | 166 | 128 | 1.30x |
| **Gradient magnitude** | **6.98** | **0.95** | **7.35x** |

## 2. Root Causes of Depth Map Underperformance

### 2.1 Extremely Low Information Content in Wound Regions

Within the wound bounding box, depth map entropy is **3.55 bits** vs depth RGB's **6.72 bits** — the depth map carries roughly half the information. This is because the depth sensor captures elevation at limited precision over the small wound area. A typical DFU wound is 1-3 cm across; at the sensor's depth resolution, this produces only a handful of distinct depth values across the wound, resulting in large flat regions with near-uniform intensity.

### 2.2 Near-Zero Texture (Gradient Magnitude 7.3x Lower)

The gradient magnitude within wound regions is **0.95 for depth map** vs **6.98 for depth RGB** — a 7.3x difference. CNNs rely heavily on edges and texture gradients for feature extraction. The depth map's wound region is essentially a smooth gradient with minimal texture, giving the CNN almost nothing to extract. Depth RGB, being a photograph from the depth camera's RGB sensor, captures wound surface texture, color variation, and tissue boundaries.

### 2.3 Anti-Correlated Channels Incompatible with ImageNet Features

Depth map inter-channel correlation is **-0.102** (R, G, B encode different depth ranges via a colormap). All ImageNet-pretrained networks expect natural images where channels are highly correlated (>0.9). The first convolutional layers of DenseNet121 learn filters optimized for correlated-channel inputs. Applying these to anti-correlated channels produces feature maps that are meaningless for the task.

In contrast:
- Depth RGB: channel correlation 0.951 (natural photo — compatible with ImageNet features)
- Thermal RGB: channel correlation 0.967 (natural photo — compatible)
- Thermal Map: channel correlation 0.635 (moderate — partially compatible)
- Depth Map: channel correlation -0.102 (anti-correlated — incompatible)

### 2.4 Sensor Precision Limitations at Wound Scale

Consumer-grade depth cameras (Intel RealSense, Azure Kinect) have depth precision of 1-5 mm at 50 cm working distance. DFU wounds are typically 1-30 mm deep with gradual slopes. At this scale, the depth sensor captures at most 5-15 distinct depth levels across the wound — insufficient to distinguish inflammatory tissue swelling from proliferative granulation tissue or remodeling scar flattening.

The temperature sensor (thermal camera) does not have this limitation: wound surface temperature varies by 2-8 degrees C between healing phases, well within thermal camera precision (0.05 degrees C typical NETD). This explains thermal_map's substantially better performance (kappa 0.501) vs depth_map (kappa 0.191).

## 3. Bounding Box and Scale Analysis

### 3.1 Wound Region Sizes

| Statistic | Depth BB | Thermal BB |
|-----------|----------|------------|
| Mean width | 69 px | 79 px |
| Mean height | 70 px | 99 px |
| Median width | 55 px | 63 px |
| Median height | 56 px | 82 px |
| Min width | 16 px | 16 px |
| Min height | 18 px | 29 px |
| Mean area | 6,723 px2 | 10,430 px2 |

### 3.2 Wound Coverage of Full Image

| Modality | Mean Coverage | Median Coverage | Min Coverage |
|----------|--------------|-----------------|--------------|
| Depth | 0.73% | 0.34% | 0.047% |
| Thermal | 0.67% | 0.34% | 0.044% |

The wound occupies less than 1% of the full image on average. After bounding box cropping, the typical wound region is 55-82 pixels across — smaller than the 128x128 input used by the model.

### 3.3 Upsampling Factors (to 128x128 model input)

| Statistic | Depth | Thermal |
|-----------|-------|---------|
| Mean upsampling | 1.67x | 1.26x |
| Median upsampling | 2.06x | 1.52x |
| Worst case | 6.1x | 4.4x |
| Tiny BBs (<30px) | 317 (9.7%) | 174 (5.5%) |

For depth_map specifically, the 2x mean upsampling applied to a 3.55-bit entropy image means interpolated pixels carry even less information — the CNN is learning from heavily smoothed, low-entropy depth colormaps.

## 4. Metadata Feature Analysis

### 4.1 Feature Completeness

72 clinical features, 60 of which have some missing values. Key missing data:
- Pain Type 2 (80.4% missing — optional secondary coding, not used)
- Exudate Appearance (21.7% missing)
- Temperature features (4.6-5.1% missing — from thermal camera)
- Most other features: <3% missing

### 4.2 Label Quality

Phase labels were assigned with a mean confidence of **78.7%** (std 6.9%). Only 0.5% of labels have confidence below 60%. The R-class has the highest labeling confidence (83.4%), yet is the hardest to classify — suggesting the difficulty is inherent to the imaging modalities, not label noise.

### 4.3 Temperature Features in Metadata

The metadata includes wound temperature measurements:
- Wound Centre: mean 29.5 C, range [12.4, 39.9]
- Peri-Ulcer: mean 29.9 C, range [14.1, 37.2]
- Normalized wound-peri-ulcer differential: mean -0.02 C, range [-0.3, 0.4]

These temperature features (derived from the thermal camera) are already captured in the metadata and contribute to the RF classifier's performance. The thermal_map images provide spatial temperature distribution that the single-point metadata measurement cannot capture — which is why thermal_map adds value beyond metadata, while depth_map (which has no metadata equivalent) adds only noise.

## 5. Summary: Depth Map vs Other Modalities

| Factor | Depth RGB | Thermal Map | Depth Map |
|--------|-----------|-------------|-----------|
| Standalone kappa | 0.511 | 0.501 | **0.191** |
| Entropy (wound region) | 6.72 bits | 4.26 bits | **3.55 bits** |
| Gradient magnitude | 6.98 | N/A | **0.95** |
| Channel correlation | 0.951 | 0.635 | **-0.102** |
| ImageNet compatibility | High | Moderate | **Low** |
| Sensor precision at wound scale | N/A (photo) | High (0.05 C) | **Low (1-5 mm)** |
| Zero pixel fraction | 0.07% | 51.6% | 0.0% |
| Adds value to fusion? | Yes (+0.08 kappa) | Yes (+0.02 kappa) | **No (-0.01 to -0.07)** |

The depth map modality fails because it attempts to encode millimeter-scale wound depth variations into a colormap representation that: (1) contains very low information per pixel, (2) has near-zero texture for CNN feature extraction, and (3) uses anti-correlated RGB channels incompatible with ImageNet-pretrained features. The depth RGB camera simultaneously captures a natural photograph with rich texture, colour, and edge information — making the raw depth map redundant and harmful to fusion performance.
