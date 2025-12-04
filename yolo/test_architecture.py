#!/usr/bin/env python3
"""
YOLOv12-BDA Simple Test Script

Quick test to verify the YOLOv12-BDA model can be created and used.
"""

import torch
from pathlib import Path

def test_yolov12_bda():
    """Test YOLOv12-BDA model creation and forward pass"""
    
    print("=" * 80)
    print("YOLOv12-BDA Quick Test")
    print("=" * 80)
    
    # Step 1: Register custom modules
    print("\n1. Registering custom modules...")
    try:
        import register_modules  # This auto-registers on import
        print("   ✅ Modules registered")
    except Exception as e:
        print(f"   ❌ Registration failed: {e}")
        return False
    
    # Step 2: Load model
    print("\n2. Loading YOLOv12-BDA model...")
    try:
        from ultralytics import YOLO
        
        yaml_path = Path('yolov12_bda.yaml')
        if not yaml_path.exists():
            print(f"   ❌ YAML file not found: {yaml_path}")
            print("   Please ensure yolov12_bda.yaml is in the current directory")
            return False
        
        model = YOLO(str(yaml_path))
        print(f"   ✅ Model loaded from {yaml_path}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params/1e6:.2f}M")
        print(f"   Trainable parameters: {trainable_params/1e6:.2f}M")
        
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Test forward pass
    print("\n3. Testing forward pass...")
    try:
        model.model.eval()
        
        # Create dummy input
        batch_size = 2
        img_size = 640
        x = torch.randn(batch_size, 3, img_size, img_size)
        
        print(f"   Input shape: {x.shape}")
        
        with torch.no_grad():
            outputs = model.model(x)
        
        print(f"   ✅ Forward pass successful!")
        
        # Display output shapes
        if isinstance(outputs, (list, tuple)):
            print(f"   Number of detection heads: {len(outputs)}")
            for i, out in enumerate(outputs):
                print(f"     Head {i+1}: {out.shape}")
        else:
            print(f"   Output shape: {outputs.shape}")
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Model summary
    print("\n4. Model architecture summary...")
    try:
        print(model.model)
    except:
        print("   (Model summary not available)")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed! YOLOv12-BDA is working correctly.")
    print("=" * 80)
    
    print("\n📝 Expected behavior:")
    print("   - Dual backbone architecture (HGBlock + DGCS)")
    print("   - DLU fusion at multiple scales")
    print("   - DASI neck for adaptive multi-scale fusion")
    print("   - Standard YOLO detection head")
    
    print("\n🚀 Next steps:")
    print("   1. Prepare your dataset in YOLO format:")
    print("      datasets/")
    print("        ├── train/")
    print("        │   ├── images/")
    print("        │   └── labels/")
    print("        └── val/")
    print("            ├── images/")
    print("            └── labels/")
    
    print("\n   2. Create dataset YAML (e.g., dataset.yaml):")
    print("      path: ./datasets")
    print("      train: train/images")
    print("      val: val/images")
    print("      nc: 1  # number of classes")
    print("      names: ['your_class']")
    
    print("\n   3. Train the model:")
    print("      python train.py --data dataset.yaml --model yolov12_bda.yaml --epochs 300")
    
    print("\n   Or use the provided training script:")
    print("      python train_aortic_valve_bda.py --img-root ... --lbl-root ...")
    
    return True


if __name__ == "__main__":
    success = test_yolov12_bda()
    exit(0 if success else 1)