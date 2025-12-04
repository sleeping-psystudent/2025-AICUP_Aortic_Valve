#!/usr/bin/env python3
"""
Improved Aortic Valve Detection Training Script
Adapted for local execution with enhanced features
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import Tuple, Optional
import yaml
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetOrganizer:
    """Handles dataset organization and validation"""
    
    def __init__(self, img_root: str, lbl_root: str, output_dir: str):
        self.img_root = Path(img_root)
        self.lbl_root = Path(lbl_root)
        self.output_dir = Path(output_dir)
        
    def find_patient_root(self, root: Path) -> Path:
        """Find the directory containing patient folders"""
        for dirpath, dirnames, _ in os.walk(root):
            if any(d.startswith("patient") for d in dirnames):
                return Path(dirpath)
        return root
    
    def validate_dataset(self, img_dir: Path, lbl_dir: Path) -> int:
        """Validate that images and labels match"""
        if not img_dir.exists() or not lbl_dir.exists():
            logger.error(f"Directory not found: {img_dir} or {lbl_dir}")
            return 0
            
        img_files = {f.stem for f in img_dir.glob("*.png")}
        lbl_files = {f.stem for f in lbl_dir.glob("*.txt")}
        
        missing_labels = img_files - lbl_files
        missing_images = lbl_files - img_files
        
        if missing_labels:
            logger.warning(f"Images without labels: {len(missing_labels)}")
        if missing_images:
            logger.warning(f"Labels without images: {len(missing_images)}")
        
        matched = len(img_files & lbl_files)
        logger.info(f"✓ Matched pairs: {matched}")
        return matched
    
    def ensure_clean_dir(self, path: Path):
        """Create or clean directory"""
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    
    def organize_data(self, 
                     train_start: int, 
                     train_end: int, 
                     val_start: int, 
                     val_end: int) -> Tuple[int, int]:
        """Organize data into train/val splits"""
        
        # Find actual patient roots
        img_root = self.find_patient_root(self.img_root)
        lbl_root = self.find_patient_root(self.lbl_root)
        
        logger.info(f"Image root: {img_root}")
        logger.info(f"Label root: {lbl_root}")
        
        # Create directories
        train_img = self.output_dir / "train" / "images"
        train_lbl = self.output_dir / "train" / "labels"
        val_img = self.output_dir / "val" / "images"
        val_lbl = self.output_dir / "val" / "labels"
        
        for dir_path in [train_img, train_lbl, val_img, val_lbl]:
            self.ensure_clean_dir(dir_path)
        
        # Move files
        def move_patients(start: int, end: int, split: str) -> int:
            count = 0
            for i in range(start, end + 1):
                patient = f"patient{i:04d}"
                img_dir = img_root / patient
                lbl_dir = lbl_root / patient
                
                if not lbl_dir.exists():
                    logger.warning(f"Label directory not found: {lbl_dir}")
                    continue
                
                for lbl_file in lbl_dir.glob("*.txt"):
                    base_name = lbl_file.stem
                    img_file = img_dir / f"{base_name}.png"
                    
                    if not img_file.exists():
                        logger.warning(f"Image not found: {img_file}")
                        continue
                    
                    # Copy files
                    try:
                        shutil.copy2(img_file, self.output_dir / split / "images" / img_file.name)
                        shutil.copy2(lbl_file, self.output_dir / split / "labels" / lbl_file.name)
                        count += 1
                    except Exception as e:
                        logger.error(f"Error copying {img_file}: {e}")
            
            return count
        
        # Organize train and val
        logger.info(f"Organizing training data (patients {train_start}-{train_end})...")
        train_count = move_patients(train_start, train_end, "train")
        
        logger.info(f"Organizing validation data (patients {val_start}-{val_end})...")
        val_count = move_patients(val_start, val_end, "val")
        
        # Validate
        logger.info("\nValidating training set:")
        self.validate_dataset(train_img, train_lbl)
        
        logger.info("\nValidating validation set:")
        self.validate_dataset(val_img, val_lbl)
        
        return train_count, val_count


def create_yaml_config(dataset_path: str, num_classes: int = 1, class_names: list = None):
    """Create YOLO dataset configuration file"""
    
    if class_names is None:
        class_names = ['aortic_valve']
    
    config = {
        'path': str(Path(dataset_path).absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': num_classes,
        'names': class_names
    }
    
    yaml_path = Path(dataset_path) / 'aortic_valve.yaml'
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"✓ Created config file: {yaml_path}")
    return yaml_path


def train_model(
    model_size: str = 'n',
    data_yaml: str = None,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 512,
    device: str = '0',
    pretrained_weights: Optional[str] = None,
    project: str = './runs/detect',
    name: str = 'train',
    resume: bool = False,
    patience: int = 50,
    save_period: int = 10,
    cache: bool = False,
    workers: int = 8
):
    """Train YOLO model with specified parameters"""
    
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    
    # Determine model
    if pretrained_weights and Path(pretrained_weights).exists():
        logger.info(f"Loading pretrained weights: {pretrained_weights}")
        model = YOLO(pretrained_weights)
    else:
        model_name = f'yolo12{model_size}.pt'
        logger.info(f"Using official YOLO model: {model_name}")
        model = YOLO(model_name)
    
    # Training arguments
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'project': project,
        'name': name,
        'patience': patience,
        'save_period': save_period,
        'cache': cache,
        'workers': workers,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 0,
        'deterministic': False,
        'single_cls': False,
        'rect': False,

        # -----------------------------
        # ⚡ Learning Rate / Optimizer
        # -----------------------------
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,

        # -----------------------------
        # ⚡ Loss Weights
        # -----------------------------
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,

        # -----------------------------
        # ⚡ Model / Behavior
        # -----------------------------
        'close_mosaic': 10,
        'resume': resume,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,

        # -----------------------------
        # 🔥 BEST Augmentation for CT
        # -----------------------------
        # geometry
        'degrees': 10,          # rotation
        'translate': 0.10,      # shift
        'scale': 0.25,          # zoom
        'shear': 0.0,           # NO shear for CT

        # flip
        'flipud': 0.0,          # do NOT flip upside-down
        'fliplr': 0.5,          # LEFT/RIGHT flip improves generalization

        # CT is grayscale → no color augment
        'hsv_h': 0.0,
        'hsv_s': 0.0,
        'hsv_v': 0.0,

        # Composition augment
        'mosaic': 0.0,          # CT mosaic looks very unnatural → turn off
        'mixup': 0.1,           # small mixup is effective for CT
        'cutmix': 0.0,          
        'copy_paste': 0.0,
        'copy_paste_mode': 'flip',

        # erasing
        'erasing': 0.2,         # do not set too high

        # -----------------------------
        # 🟦 Multi-scale Training
        # -----------------------------
        'multi_scale': True,    # VERY important for YOLO generalization
    }
    
    # train_args = {
    #     'data': data_yaml,
    #     'epochs': epochs,
    #     'batch': batch_size,
    #     'imgsz': img_size,
    #     'device': device,
    #     'project': project,
    #     'name': name,
    #     'patience': patience,
    #     'save_period': save_period,
    #     'cache': cache,
    #     'workers': workers,

    #     # General
    #     'exist_ok': True,
    #     'pretrained': True,
    #     'optimizer': 'auto',
    #     'verbose': True,
    #     'seed': 0,
    #     'deterministic': False,
    #     'single_cls': False,
    #     'rect': False,
    #     'cos_lr': False,
    #     'close_mosaic': 10,
    #     'resume': resume,
    #     'amp': True,  # Automatic Mixed Precision
    #     'fraction': 1.0,
    #     'profile': False,
    #     'freeze': None,

    #     # Learning Rate / Optimizer
    #     'lr0': 0.01,
    #     'lrf': 0.01,
    #     'momentum': 0.937,
    #     'weight_decay': 0.0005,
    #     'warmup_epochs': 3.0,
    #     'warmup_momentum': 0.8,
    #     'warmup_bias_lr': 0.1,

    #     # Loss Weights
    #     'box': 7.5,
    #     'cls': 0.5,
    #     'dfl': 1.5,
    #     'pose': 12.0,
    #     'kobj': 1.0,
    #     'label_smoothing': 0.0,
    #     'nbs': 64,

    #     # Mask / Misc
    #     'overlap_mask': True,
    #     'mask_ratio': 4,
    #     'dropout': 0.0,

    #     # Validation
    #     'val': True,
    # }

    logger.info("\n" + "="*80)
    logger.info("Starting Training")
    logger.info("="*80)
    logger.info(f"Model: YOLO12{model_size.upper()}")
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Image Size: {img_size}")
    logger.info(f"Device: {device}")
    logger.info("="*80 + "\n")
    
    # Train
    results = model.train(**train_args)
    
    logger.info("\n" + "="*80)
    logger.info("Training Complete!")
    logger.info("="*80)
    logger.info(f"Best weights: {Path(project) / name / 'weights' / 'best.pt'}")
    logger.info(f"Last weights: {Path(project) / name / 'weights' / 'last.pt'}")
    logger.info("="*80 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLO model for aortic valve detection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data organization arguments
    parser.add_argument('--img-root', type=str, required=True,
                       help='Path to training images root directory')
    parser.add_argument('--lbl-root', type=str, required=True,
                       help='Path to training labels root directory')
    parser.add_argument('--output-dir', type=str, default='./datasets',
                       help='Output directory for organized dataset')
    parser.add_argument('--train-start', type=int, default=1,
                       help='First patient number for training')
    parser.add_argument('--train-end', type=int, default=30,
                       help='Last patient number for training')
    parser.add_argument('--val-start', type=int, default=31,
                       help='First patient number for validation')
    parser.add_argument('--val-end', type=int, default=50,
                       help='Last patient number for validation')
    
    # Training arguments
    parser.add_argument('--model-size', type=str, default='x', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=512,
                       help='Image size')
    parser.add_argument('--device', type=str, default='0',
                       help='CUDA device (0, 1, 2, etc.) or cpu')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to pretrained weights (optional)')
    parser.add_argument('--project', type=str, default='./runs/detect',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='train',
                       help='Experiment name')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    parser.add_argument('--save-period', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--cache', action='store_true',
                       help='Cache images for faster training')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of worker threads')
    
    # Workflow control
    parser.add_argument('--skip-organize', action='store_true',
                       help='Skip data organization (use existing)')
    parser.add_argument('--organize-only', action='store_true',
                       help='Only organize data, do not train')
    
    args = parser.parse_args()
    
    # Step 1: Organize data
    if not args.skip_organize:
        logger.info("="*80)
        logger.info("STEP 1: Organizing Dataset")
        logger.info("="*80)
        
        organizer = DatasetOrganizer(
            args.img_root,
            args.lbl_root,
            args.output_dir
        )
        
        train_count, val_count = organizer.organize_data(
            args.train_start,
            args.train_end,
            args.val_start,
            args.val_end
        )
        
        logger.info(f"\n✓ Training samples: {train_count}")
        logger.info(f"✓ Validation samples: {val_count}")
        
        # Create YAML config
        yaml_path = create_yaml_config(args.output_dir)
    else:
        yaml_path = Path(args.output_dir) / 'aortic_valve.yaml'
        if not yaml_path.exists():
            logger.error(f"Config file not found: {yaml_path}")
            logger.error("Run without --skip-organize first")
            sys.exit(1)
    
    if args.organize_only:
        logger.info("\n✓ Data organization complete!")
        return
    
    # Step 2: Train model
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Training Model")
    logger.info("="*80)
    
    train_model(
        model_size=args.model_size,
        data_yaml=str(yaml_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
        pretrained_weights=args.weights,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=args.patience,
        save_period=args.save_period,
        cache=args.cache,
        workers=args.workers
    )
    
    logger.info("\n✓ All steps complete!")


if __name__ == '__main__':
    main()