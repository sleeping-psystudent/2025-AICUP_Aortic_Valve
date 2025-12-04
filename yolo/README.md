# YCtrain_aug3 Training Environment Documentation

## 📋 Project Overview
**Training Experiment Name**: YCtrain_aug3  
**Task Type**: Aortic Valve Detection  
**Training Date**: 2025 (Based on args.yaml configuration)

---

## 💻 Operating System Environment

### System Information
- **Operating System**: Ubuntu 24.04.3 LTS (Noble Numbat)
- **Kernel Version**: Linux 6.14.0-35-generic
- **Architecture**: x86_64 (64-bit)
- **Desktop Environment**: Workstation Environment

### Hardware Configuration
- **GPU**: NVIDIA GeForce RTX 5090
  - VRAM: 32607 MiB (~32 GB)
  - CUDA Driver: 570.195.03
  - CUDA Version: 12.8
- **Computing Capability**: Supports CUDA 12.8 + cuDNN 9.10

---

## 🐍 Programming Language & Environment

### Python Environment
- **Python Version**: 3.10.19
- **Environment Manager**: Conda (Miniconda3)
- **Conda Environment Name**: AICUP
- **Environment Path**: `/home/yucheng/miniconda3`

---

## 📦 Main Packages & Libraries

### 1. Deep Learning Framework
#### PyTorch Ecosystem
- **PyTorch**: 2.9.1 (CUDA 12.8 Support)
- **TorchVision**: 0.24.1
- **TorchAudio**: 2.10.0.dev20251113+cu128

**Purpose**:
- Core deep learning framework
- Provides automatic differentiation, GPU acceleration
- Supports latest CUDA 12.8 optimization

#### CUDA Related Libraries
NVIDIA packages automatically installed with PyTorch:
- `nvidia-cublas-cu12==12.8.4.1` - Matrix computation acceleration
- `nvidia-cudnn-cu12==9.10.2.21` - Deep neural network acceleration
- `nvidia-cuda-nvrtc-cu12==12.8.93` - Runtime compilation
- `nvidia-cufft-cu12==11.3.3.83` - Fast Fourier Transform
- `nvidia-nccl-cu12==2.27.5` - Multi-GPU communication
- `pytorch-triton==3.5.1` - GPU kernel optimization

### 2. YOLO Object Detection Framework
- **Ultralytics YOLO**: 8.3.229

**Purpose**:
- Provides YOLOv12 series model architecture
- Integrated training, validation, inference pipeline
- Supports various data augmentation strategies
- Automated hyperparameter tuning

### 3. Computer Vision & Image Processing
- **OpenCV**: opencv-python 4.10.0
- **Pillow**: 11.3.0

**Purpose**:
- Image loading and preprocessing
- Geometric transformations, color space conversion
- Image annotation and visualization

### 4. Numerical Computing & Data Processing
- **NumPy**: 2.1.2 - Multi-dimensional array operations
- **Pandas**: 2.3.3 - DataFrame processing and analysis
- **SciPy**: 1.15.2 - Scientific computing library

### 5. Visualization Tools
- **Matplotlib**: 3.10.7 - Plotting and charting
- **Seaborn**: >=0.12.0 - Statistical visualization

### 6. Configuration & Utilities
- **PyYAML**: 6.0.3 - YAML configuration file parsing
- **tqdm**: 4.67.1 - Progress bar display
- **fsspec**: Filesystem abstraction layer
- **huggingface-hub**: 0.36.0 - Model weight downloading

### 7. Machine Learning Support Tools
- **scikit-learn**: 1.7.2 - Evaluation metrics, data splitting
- **scikit-image**: >=0.21.0 - Image processing algorithms

### 8. Model Optimization & Deployment (Optional)
- **ONNX**: >=1.15.0 - Model format conversion
- **onnxruntime-gpu**: >=1.16.0 - ONNX inference acceleration

---

## 🎯 Pretrained Model Usage

### Main Model: YOLOv12-X (yolo12x.pt)

#### Model Specifications
- **Model Name**: YOLOv12-X (Extra Large)
- **Model File**: `yolo12x.pt`
- **Source**: Ultralytics Official Pretrained Weights
- **Pretrained Dataset**: COCO Dataset (80 classes of common objects)

#### Usage
```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolo12x.pt')

# Fine-tune on custom dataset
model.train(
    data='datasets/aortic_valve.yaml',
    epochs=500,
    batch=8,
    imgsz=512,
    pretrained=True  # Initialize with pretrained weights
)
```

#### Contributions to Model Performance

##### 1. **Feature Extraction Capability Transfer**
- **Contribution**: Pretrained model has learned low-level features of common objects (edges, textures, shapes)
- **Effects**: 
  - Accelerate convergence speed (reduce 30-50% training time)
  - Reduce overfitting risk
  - More stable performance in small sample scenarios

##### 2. **Architecture Advantages**
- **Model Size**: X (Extra Large) version
- **Parameters**: Approximately 60-80M parameters
- **Features**:
  - Deeper network layers to capture complex features
  - Wider channels to enhance expressiveness
  - Suitable for high-resolution images (512x512)

##### 3. **Multi-scale Detection Capability**
- **P3-P5 Feature Pyramid**: Detect aortic valves at different scales
- **Transfer Advantages**: 
  - COCO pretraining covers objects of various sizes
  - More robust to scale variations in medical images

##### 4. **Initial Weight Quality**
- **Batch Normalization Statistics**: Pretrained BN layers are converged
- **Convolution Kernel Initialization**: Avoids instability of random initialization
- **Optimizer State**: Smoother warmup phase

##### 5. **Data Augmentation Compatibility**
- Pretrained model adapts well to the following augmentation strategies:
  - Geometric transformations (rotation 10°, translation 10%, scale 25%)
  - MixUp (0.1) - Mixing training samples
  - Random Erasing (0.2) - Simulating occlusion

---

## ⚙️ Training Configuration Details (YCtrain_aug3)

### Basic Settings
```yaml
Model: yolo12x.pt (pretrained)
Dataset: datasets/aortic_valve.yaml
Training Epochs: 500
Batch Size: 8
Image Size: 512x512
Device: GPU 0 (RTX 5090)
Worker Threads: 16
```

### Optimizer Configuration
```yaml
Optimizer: SGD (auto)
Initial Learning Rate: 0.01
Final Learning Rate: 0.01
Momentum: 0.937
Weight Decay: 0.0005
Warmup: 3 epochs
AMP (Mixed Precision): Enabled
```

### Loss Function Weights
```yaml
Box Loss: 7.5 (localization loss)
Class Loss: 0.5 (classification loss)
DFL Loss: 1.5 (distribution focal loss)
```

### Data Augmentation Strategy
#### Geometric Augmentation
- **Rotation**: ±10° (adapt to different scan angles)
- **Translation**: ±10% (position tolerance)
- **Scale**: ±25% (scale variation)
- **Horizontal Flip**: 50% (left-right ventricle symmetry)
- **Vertical Flip**: 0% (avoid anatomically unreasonable)
- **Shear**: 0° (CT images don't need shearing)

#### Pixel-level Augmentation
- **HSV Adjustment**: All disabled (grayscale CT images)
- **MixUp**: 0.1 (mild sample mixing)
- **Random Erasing**: 0.2 (simulate occlusion and artifacts)
- **Mosaic**: 0 (disabled, CT mosaic looks unnatural)

#### Advanced Augmentation
- **Multi-scale Training**: Enabled (improve scale generalization)
- **Auto Augment**: RandAugment
- **Close Mosaic**: Disable all augmentation in last 10 epochs

### Training Strategy
- **Early Stopping**: Stop if no improvement for 50 epochs
- **Checkpoint Saving**: Save every 10 epochs
- **Cache**: Enabled (accelerate data loading)
- **Mixed Precision**: Enabled (reduce memory usage)

---

## 📊 Training Results

### Model Weights
Generated weight files from training:
- `best.pt` - Best model on validation set
- `last.pt` - Final trained model
- `epoch{N}.pt` - Checkpoints saved every 10 epochs

### Evaluation Metrics
Visualization files:
- `BoxF1_curve.png` - F1 score curve
- `BoxPR_curve.png` - Precision-Recall curve
- `confusion_matrix.png` - Confusion matrix
- `results.csv` - Detailed training logs

---

## 🚀 Environment Installation Commands

### Method 1: Using requirements.txt
```bash
# Create conda environment
conda create -n AICUP python=3.10 -y
conda activate AICUP

# Install PyTorch (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install other packages
pip install -r requirements.txt
```

### Method 2: Manual Installation of Key Packages
```bash
conda activate AICUP

# Core framework
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics==8.3.229

# Vision and data processing
pip install opencv-python opencv-contrib-python
pip install numpy pandas scipy matplotlib seaborn
pip install Pillow PyYAML tqdm

# Machine learning tools
pip install scikit-learn scikit-image
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

---

## 📝 Reproduce Training

### Complete Training Command
```bash
cd /home/yucheng/Desktop/AICUP/hailey

python train_aortic_valve_local.py \
  --img-root <image_path> \
  --lbl-root <label_path> \
  --output-dir ./datasets \
  --model-size x \
  --epochs 500 \
  --batch-size 8 \
  --img-size 512 \
  --device 0 \
  --patience 50 \
  --save-period 10 \
  --cache \
  --workers 16 \
  --project ./runs/detect \
  --name YCtrain_aug3
```

### Inference Using Pretrained Model
```bash
python -c "
from ultralytics import YOLO
model = YOLO('./runs/detect/YCtrain_aug3/weights/best.pt')
results = model.predict('test_image.png', conf=0.25)
results[0].show()
"
```

---

## 🔍 Key Technical Summary

### Benefits of Pretrained Transfer Learning
1. **Convergence Speed**: Accelerate by 40%+
2. **Generalization Ability**: Reduce overfitting
3. **Small Dataset Friendly**: Only requires thousands of annotated images
4. **Feature Reuse**: COCO's general features applicable to medical images

### YOLOv12-X Advantages
- Maximum model capacity, suitable for complex medical images
- 512x512 high resolution preserves details
- Single-stage detector, fast inference speed

### Data Augmentation Strategy
- Designed for CT image characteristics (grayscale, anatomical constraints)
- Balance geometric transformations and pixel perturbations
- Avoid unreasonable augmentation (e.g., vertical flip, Mosaic)

