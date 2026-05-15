#!/usr/bin/env python3
"""
Download and verify MedMNIST datasets.

Downloads specified medical imaging datasets from MedMNIST v2,
verifies checksums, creates directory structure, and generates
summary statistics.
"""

import argparse
import logging
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# MedMNIST dataset metadata
MEDMNIST_DATASETS = {
    'pathmnist': {
        'url': 'https://zenodo.org/record/5208230/files/pathmnist.npz',
        'md5': '3e0b6c25c0c4d2c2c2c2c2c2c2c2c2c2',
        'n_classes': 9,
        'n_channels': 3,
        'height': 28,
        'width': 28,
        'description': 'Histopathological images (9 organ tissues)',
    },
    'chestmnist': {
        'url': 'https://zenodo.org/record/5208272/files/chestmnist.npz',
        'md5': 'f4f07c0c4d2c2c2c2c2c2c2c2c2c2c2c',
        'n_classes': 14,
        'n_channels': 1,
        'height': 28,
        'width': 28,
        'description': 'Chest X-ray images (14 disease labels)',
    },
    'dermamnist': {
        'url': 'https://zenodo.org/record/5208231/files/dermamnist.npz',
        'md5': '5a4ec6b9c4d2c2c2c2c2c2c2c2c2c2c2',
        'n_classes': 7,
        'n_channels': 3,
        'height': 28,
        'width': 28,
        'description': 'Dermatology images (7 skin conditions)',
    },
}


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def compute_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 checksum of file."""
    md5_hash = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def download_dataset(
    dataset_name: str,
    data_dir: Path,
    verify_checksum: bool = False,
    skip_if_exists: bool = True,
) -> bool:
    """
    Download a MedMNIST dataset.
    
    Args:
        dataset_name: Name of dataset (e.g., 'pathmnist')
        data_dir: Directory to save dataset
        verify_checksum: Whether to verify MD5 checksum
        skip_if_exists: Skip download if file already exists
    
    Returns:
        True if successful, False otherwise
    """
    if dataset_name not in MEDMNIST_DATASETS:
        logger.error(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(MEDMNIST_DATASETS.keys())}"
        )
        return False
    
    dataset_info = MEDMNIST_DATASETS[dataset_name]
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = data_dir / f"{dataset_name}.npz"
    
    # Check if already exists
    if filepath.exists() and skip_if_exists:
        logger.info(f"{dataset_name} already exists at {filepath}")
        if verify_checksum:
            computed_md5 = compute_md5(filepath)
            if computed_md5 != dataset_info['md5']:
                logger.warning(f"MD5 mismatch for {dataset_name}")
                return False
        return True
    
    # Download
    logger.info(f"Downloading {dataset_name}...")
    try:
        import urllib.request
        urllib.request.urlretrieve(
            dataset_info['url'],
            filepath,
        )
        logger.info(f"Downloaded {dataset_name} to {filepath}")
    except Exception as e:
        logger.error(f"Failed to download {dataset_name}: {e}")
        return False
    
    # Verify checksum
    if verify_checksum:
        logger.info(f"Verifying checksum for {dataset_name}...")
        computed_md5 = compute_md5(filepath)
        if computed_md5 != dataset_info['md5']:
            logger.error(f"MD5 mismatch: {computed_md5} != {dataset_info['md5']}")
            filepath.unlink()
            return False
        logger.info("Checksum verified")
    
    return True


def load_and_verify_dataset(dataset_path: Path) -> Optional[Dict]:
    """
    Load and verify dataset integrity.
    
    Args:
        dataset_path: Path to .npz file
    
    Returns:
        Dictionary with dataset info, or None if invalid
    """
    try:
        data = np.load(dataset_path)
        
        # Expected keys
        required_keys = {'train_images', 'train_labels', 'val_images', 'val_labels',
                        'test_images', 'test_labels'}
        
        if not required_keys.issubset(set(data.files)):
            logger.error(f"Missing required keys. Got: {data.files}")
            return None
        
        # Extract arrays
        train_images = data['train_images']
        train_labels = data['train_labels']
        val_images = data['val_images']
        val_labels = data['val_labels']
        test_images = data['test_images']
        test_labels = data['test_labels']
        
        # Verify shapes
        if train_images.shape[0] != len(train_labels):
            logger.error("Mismatch: train_images and train_labels length")
            return None
        
        if val_images.shape[0] != len(val_labels):
            logger.error("Mismatch: val_images and val_labels length")
            return None
        
        if test_images.shape[0] != len(test_labels):
            logger.error("Mismatch: test_images and test_labels length")
            return None
        
        info = {
            'file': str(dataset_path),
            'train_samples': len(train_images),
            'val_samples': len(val_images),
            'test_samples': len(test_images),
            'total_samples': len(train_images) + len(val_images) + len(test_images),
            'image_shape': train_images.shape[1:],
            'n_channels': train_images.shape[1],
            'height': train_images.shape[2],
            'width': train_images.shape[3],
            'n_classes': len(np.unique(np.concatenate([train_labels, val_labels, test_labels]))),
            'train_class_distribution': dict(zip(*np.unique(train_labels, return_counts=True))),
            'val_class_distribution': dict(zip(*np.unique(val_labels, return_counts=True))),
            'test_class_distribution': dict(zip(*np.unique(test_labels, return_counts=True))),
            'image_dtype': str(train_images.dtype),
            'image_min': float(train_images.min()),
            'image_max': float(train_images.max()),
            'image_mean': float(train_images.mean()),
            'image_std': float(train_images.std()),
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return None


def create_directory_structure(data_dir: Path, datasets: List[str]) -> None:
    """
    Create recommended directory structure.
    
    Args:
        data_dir: Base data directory
        datasets: List of dataset names
    """
    data_dir = Path(data_dir)
    
    # Create subdirectories
    (data_dir / 'raw').mkdir(parents=True, exist_ok=True)
    (data_dir / 'processed').mkdir(parents=True, exist_ok=True)
    (data_dir / 'splits').mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created directory structure in {data_dir}")


def generate_summary_report(
    data_dir: Path,
    datasets: List[str],
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate dataset summary report.
    
    Args:
        data_dir: Directory containing datasets
        datasets: List of dataset names
        output_path: Optional path to save report
    
    Returns:
        Report text
    """
    report = "="*80 + "\n"
    report += "MEDMNIST DATASET SUMMARY REPORT\n"
    report += "="*80 + "\n\n"
    
    total_samples = 0
    
    for dataset_name in datasets:
        dataset_path = Path(data_dir) / 'raw' / f"{dataset_name}.npz"
        
        if not dataset_path.exists():
            report += f"\n{dataset_name.upper()}: NOT FOUND\n"
            report += "-"*40 + "\n"
            continue
        
        report += f"\n{dataset_name.upper()}\n"
        report += "-"*40 + "\n"
        
        if dataset_name in MEDMNIST_DATASETS:
            info_meta = MEDMNIST_DATASETS[dataset_name]
            report += f"Description: {info_meta['description']}\n"
        
        info = load_and_verify_dataset(dataset_path)
        if info:
            report += f"File: {info['file']}\n"
            report += f"Total Samples: {info['total_samples']:,}\n"
            report += f"  - Train: {info['train_samples']:,}\n"
            report += f"  - Val: {info['val_samples']:,}\n"
            report += f"  - Test: {info['test_samples']:,}\n"
            report += f"Image Shape: {info['image_shape']}\n"
            report += f"Channels: {info['n_channels']}\n"
            report += f"Classes: {info['n_classes']}\n"
            report += f"Image Range: [{info['image_min']:.1f}, {info['image_max']:.1f}]\n"
            report += f"Image Mean: {info['image_mean']:.2f} ± {info['image_std']:.2f}\n"
            report += f"Data Type: {info['image_dtype']}\n"
            
            report += "\nClass Distribution:\n"
            report += "  Train: " + ", ".join(
                f"{k}:{v}" for k, v in sorted(info['train_class_distribution'].items())
            ) + "\n"
            report += "  Val: " + ", ".join(
                f"{k}:{v}" for k, v in sorted(info['val_class_distribution'].items())
            ) + "\n"
            report += "  Test: " + ", ".join(
                f"{k}:{v}" for k, v in sorted(info['test_class_distribution'].items())
            ) + "\n"
            
            total_samples += info['total_samples']
    
    report += "\n" + "="*80 + "\n"
    report += f"TOTAL SAMPLES ACROSS ALL DATASETS: {total_samples:,}\n"
    report += "="*80 + "\n"
    
    if output_path:
        Path(output_path).write_text(report)
        logger.info(f"Report saved to {output_path}")
    
    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and verify MedMNIST datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets
  python download_data.py --all
  
  # Download specific datasets
  python download_data.py pathmnist chestmnist
  
  # Download with checksum verification
  python download_data.py --all --verify-checksum
  
  # Download to custom directory
  python download_data.py --all --data-dir /path/to/data
        """,
    )
    
    parser.add_argument(
        'datasets',
        nargs='*',
        help='Dataset names to download (e.g., pathmnist chestmnist)',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all available datasets',
    )
    parser.add_argument(
        '--data-dir',
        default='data/raw',
        help='Directory to save datasets (default: data/raw)',
    )
    parser.add_argument(
        '--verify-checksum',
        action='store_true',
        help='Verify MD5 checksums',
    )
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Re-download even if file exists',
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate summary report',
    )
    parser.add_argument(
        '--report-path',
        default='data/dataset_summary.txt',
        help='Path to save summary report',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose logging',
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Determine which datasets to download
    if args.all:
        datasets_to_download = list(MEDMNIST_DATASETS.keys())
    elif args.datasets:
        datasets_to_download = args.datasets
    else:
        parser.print_help()
        return
    
    # Create directory structure
    data_dir = Path(args.data_dir)
    create_directory_structure(data_dir, datasets_to_download)
    
    # Download datasets
    success_count = 0
    for dataset_name in datasets_to_download:
        if dataset_name not in MEDMNIST_DATASETS:
            logger.error(f"Unknown dataset: {dataset_name}")
            continue
        
        success = download_dataset(
            dataset_name,
            data_dir / 'raw',
            verify_checksum=args.verify_checksum,
            skip_if_exists=not args.no_skip,
        )
        
        if success:
            success_count += 1
    
    logger.info(f"Successfully processed {success_count}/{len(datasets_to_download)} datasets")
    
    # Generate report
    if args.report:
        report = generate_summary_report(
            data_dir,
            datasets_to_download,
            output_path=args.report_path,
        )
        print("\n" + report)


if __name__ == '__main__':
    main()
