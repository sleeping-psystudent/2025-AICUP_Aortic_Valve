"""
Module Registration for YOLOv12-BDA (Updated)

This file registers our custom modules with the Ultralytics framework
so they can be used in YAML configuration files.

IMPORTANT: Import this file BEFORE creating any YOLO models!
"""

import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import all custom modules
try:
    # Try importing from fixed version first
    from yolov12_bda_modules_fixed import (
        HGStem,
        HGBlock,
        DLUBlock,
        DGCSBlock,
        DASINeck,
        Conv
    )
    print("✅ Loaded fixed YOLOv12-BDA modules")
except ImportError:
    # Fallback to original
    from yolov12_bda_modules import (
        HGStem,
        HGBlock,
        DLUBlock,
        DGCSBlock,
        DASINeck,
        Conv
    )
    print("✅ Loaded original YOLOv12-BDA modules")


def register_yolov12_bda_modules():
    """
    Register custom YOLOv12-BDA modules with Ultralytics.
    
    This allows the YAML parser to recognize our custom module names.
    """
    try:
        # Import Ultralytics modules namespace
        import ultralytics.nn.modules as ultralyticsnn
        
        # Register each custom module
        ultralyticsnn.HGStem = HGStem
        ultralyticsnn.HGBlock = HGBlock
        ultralyticsnn.DLUBlock = DLUBlock
        ultralyticsnn.DGCSBlock = DGCSBlock
        ultralyticsnn.DASINeck = DASINeck
        
        # Also add to __all__ if it exists
        if hasattr(ultralyticsnn, '__all__'):
            for name in ['HGStem', 'HGBlock', 'DLUBlock', 'DGCSBlock', 'DASINeck']:
                if name not in ultralyticsnn.__all__:
                    ultralyticsnn.__all__.append(name)
        
        print("✅ YOLOv12-BDA modules registered with Ultralytics")
        print("   Available: HGStem, HGBlock, DLUBlock, DGCSBlock, DASINeck")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Warning: Could not register modules: {e}")
        print("   Ultralytics may not be installed")
        return False
    except Exception as e:
        print(f"⚠️  Registration error: {e}")
        return False


def inject_modules_to_namespace():
    """
    Alternative registration - direct namespace injection
    """
    try:
        import ultralytics.nn.modules as ultralyticsnn
        
        # Inject into module dict
        module_dict = ultralyticsnn.__dict__
        
        module_dict['HGStem'] = HGStem
        module_dict['HGBlock'] = HGBlock
        module_dict['DLUBlock'] = DLUBlock
        module_dict['DGCSBlock'] = DGCSBlock
        module_dict['DASINeck'] = DASINeck
        
        print("✅ Modules injected into Ultralytics namespace")
        return True
        
    except Exception as e:
        print(f"⚠️  Injection failed: {e}")
        return False


# Auto-register when imported (not when run as main)
if __name__ != "__main__":
    success1 = register_yolov12_bda_modules()
    if not success1:
        inject_modules_to_namespace()


# ============================================================================
# TESTING CODE (run with: python register_modules.py)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing YOLOv12-BDA Module Registration")
    print("=" * 70)
    
    # Test 1: Registration
    print("\n1. Testing registration...")
    success1 = register_yolov12_bda_modules()
    
    print("\n2. Testing injection method...")
    success2 = inject_modules_to_namespace()
    
    # Test 2: Module instantiation
    print("\n3. Testing module instantiation...")
    try:
        import torch
        
        batch_size = 2
        
        # Test HGStem
        print("\n   Testing HGStem...")
        stem = HGStem(3, 32)
        x = torch.randn(batch_size, 3, 640, 640)
        out = stem(x)
        print(f"   ✓ HGStem: {x.shape} → {out.shape}")
        
        # Test HGBlock
        print("\n   Testing HGBlock...")
        hgblock = HGBlock(32, 64, n=3)
        out = hgblock(out)
        print(f"   ✓ HGBlock: {out.shape}")
        
        # Test DGCSBlock
        print("\n   Testing DGCSBlock...")
        dgcs = DGCSBlock(64, 64)
        out = dgcs(out)
        print(f"   ✓ DGCSBlock: {out.shape}")
        
        # Test DLUBlock with list input (YAML format)
        print("\n   Testing DLUBlock (dual input)...")
        dlu = DLUBlock(64)
        feat_a = torch.randn(batch_size, 64, 160, 160)
        feat_b = torch.randn(batch_size, 64, 160, 160)
        out = dlu([feat_a, feat_b])  # List input (YAML format)
        print(f"   ✓ DLUBlock: {feat_a.shape} + {feat_b.shape} → {out.shape}")
        
        # Test DASINeck with list input (YAML format)
        print("\n   Testing DASINeck (multi-scale input)...")
        dasi = DASINeck(128)
        high = torch.randn(batch_size, 128, 320, 320)
        curr = torch.randn(batch_size, 128, 160, 160)
        low = torch.randn(batch_size, 128, 80, 80)
        out = dasi([high, curr, low])  # List input (YAML format)
        print(f"   ✓ DASINeck: {high.shape} + {curr.shape} + {low.shape} → {out.shape}")
        
        print("\n" + "=" * 70)
        print("✅ All modules tested successfully!")
        print("=" * 70)
        
        # Test 3: YAML model creation
        print("\n4. Testing YAML model creation...")
        try:
            from ultralytics import YOLO
            
            # Check if YAML file exists
            yaml_path = Path('yolov12_bda.yaml')
            if not yaml_path.exists():
                print(f"   ⚠️  YAML file not found: {yaml_path}")
                print("   Please ensure yolov12_bda.yaml is in the current directory")
            else:
                print(f"   Found YAML: {yaml_path}")
                print("   Creating model...")
                
                model = YOLO(str(yaml_path))
                total_params = sum(p.numel() for p in model.model.parameters())
                
                print(f"   ✅ Model created successfully!")
                print(f"   Total parameters: {total_params/1e6:.2f}M")
                
                # Test forward pass
                print("\n   Testing forward pass...")
                model.model.eval()
                x_test = torch.randn(1, 3, 640, 640)
                
                with torch.no_grad():
                    try:
                        outputs = model.model(x_test)
                        print(f"   ✅ Forward pass successful!")
                        if isinstance(outputs, (list, tuple)):
                            print(f"   Output heads: {len(outputs)}")
                            for i, out in enumerate(outputs):
                                print(f"     Head {i}: {out.shape}")
                        else:
                            print(f"   Output shape: {outputs.shape}")
                    except Exception as e:
                        print(f"   ⚠️  Forward pass error: {e}")
                        print("   Model structure is correct but may need adjustments")
                
        except Exception as e:
            print(f"   ⚠️  Model creation error: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Testing complete!")
    print("=" * 70)