# 🎨 U-Net Image Segmentation Model

> **Advanced Deep Learning Solution for Vehicle Image Masking Challenge**

## 📋 Overview

This project implements a **U-Net architecture** for semantic image segmentation, specifically designed for the Carvana Image Masking Challenge. The model performs pixel-level binary segmentation to identify and mask vehicles from high-resolution images.


## 🏗️ Architecture

### Model Components

**Encoder Blocks:**
- Convolutional layers with ReLU activation
- Max pooling for progressive downsampling
- Feature map extraction: 64 → 128 → 256 channels

**Decoder Blocks:**
- Transposed convolution for upsampling
- Skip connections for feature preservation
- Channel concatenation for refined predictions

**Output:**
- Sigmoid activation for binary classification
- Pixel-level mask generation (0-1 probability)

```
Input (1, 128, 128)
        ↓
    Encoder 1 (64 ch) → skip1
        ↓
    Encoder 2 (128 ch) → skip2
        ↓
    Encoder 3 (256 ch)
        ↓
    Decoder 1 (128 ch) + skip2
        ↓
    Decoder 2 (64 ch) + skip1
        ↓
Output (1, 1, 128, 128)
```

---

## 📊 Dataset

**Carvana Image Masking Challenge**
- **Train Images**: Vehicle photos at various angles and lighting conditions
- **Input Size**: 128×128 grayscale images
- **Output**: Binary segmentation masks
- **Format**: PNG/JPG with normalized pixel values


## 🎯 Results

### Model Performance Visualization

#### Input vs Output Comparison

| Type | IMAGE_1 | IMAGE_2 |
|------|---------|---------|
| **Input** | ![Input 1](RESULTS/IMAGE/IMAGE_1.png) | ![Input 2](RESULTS/IMAGE/IMAGE_2.png) |
| **Output** | ![Output 1](RESULTS/OUTPUT/IMAGE_1.png) | ![Output 2](RESULTS/OUTPUT/IMAGE_2.png) |

### Training Configuration
- **Optimizer**: Adam (lr=1e-4)
- **Loss Function**: BCEWithLogitsLoss
- **Epochs**: 10
- **Batch Size**: Optimized for GPU memory
- **Device**: CUDA GPU (with CPU fallback)




## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Input Resolution | 128×128 px |
| Model Parameters | ~2.1M |
| Training Loss | BCEWithLogitsLoss |
| Learning Rate | 1e-4 |
| Optimizer | Adam |
| GPU Memory | Optimized |
| Inference Speed | Real-time |

---

## 🔗 Dependencies

- `torch>=1.9.0` - Deep learning framework
- `torchvision>=0.10.0` - Computer vision utilities
- `pillow>=8.0` - Image processing
- `numpy>=1.20.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `matplotlib>=3.3.0` - Visualization
- `onnx` - Model format
- `onnxruntime` - ONNX inference engine

---

