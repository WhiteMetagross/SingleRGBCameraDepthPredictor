"""
Setup script to download required model weights
"""
import os
import sys
from pathlib import Path
import urllib.request
import shutil


def download_depth_model(model_type='vits'):
    """
    Download DepthAnything V2 model weights
    
    Args:
        model_type: Model type ('vits', 'vitb', 'vitl', 'vitg')
    """
    checkpoint_dir = Path('data/models')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = checkpoint_dir / f'depth_anything_v2_{model_type}.pth'
    
    if model_path.exists():
        print(f"✓ Model already exists: {model_path}")
        return True
    
    print(f"\n{'='*80}")
    print(f"Downloading DepthAnything V2 ({model_type}) model...")
    print(f"{'='*80}\n")
    
    # Try using gdown first
    try:
        import gdown
        
        file_ids = {
            'vits': '1kSZWHpDvBH4sxKYu94-LKX9GfZVAxKaz',
            'vitb': '1mNtWnpAArnnAuvPyDcH_8YTOvUXQVDKJ',
            'vitl': '1nSscGBawMcKAw9sKYu94-LKX9GfZVAxKaz',
            'vitg': '1nSscGBawMcKAw9sKYu94-LKX9GfZVAxKaz',
        }
        
        file_id = file_ids.get(model_type)
        if file_id:
            url = f'https://drive.google.com/uc?id={file_id}'
            print(f"Using gdown to download from Google Drive...")
            gdown.download(url, str(model_path), quiet=False)
            print(f"✓ Model downloaded successfully!")
            return True
    except ImportError:
        print("gdown not installed, trying alternative method...")
    except Exception as e:
        print(f"gdown download failed: {e}")
    
    # Alternative: Try direct download from GitHub releases
    print("\nTrying GitHub releases...")
    github_urls = {
        'vits': 'https://github.com/DepthAnything/Depth-Anything-V2/releases/download/v1.0.0/depth_anything_v2_vits.pth',
        'vitb': 'https://github.com/DepthAnything/Depth-Anything-V2/releases/download/v1.0.0/depth_anything_v2_vitb.pth',
        'vitl': 'https://github.com/DepthAnything/Depth-Anything-V2/releases/download/v1.0.0/depth_anything_v2_vitl.pth',
        'vitg': 'https://github.com/DepthAnything/Depth-Anything-V2/releases/download/v1.0.0/depth_anything_v2_vitg.pth',
    }
    
    url = github_urls.get(model_type)
    if url:
        try:
            print(f"Downloading from: {url}")
            urllib.request.urlretrieve(url, str(model_path))
            print(f"✓ Model downloaded successfully!")
            return True
        except Exception as e:
            print(f"GitHub download failed: {e}")
    
    print(f"\n{'='*80}")
    print(f"✗ Failed to download model automatically")
    print(f"{'='*80}")
    print(f"\nPlease download manually from:")
    print(f"  https://huggingface.co/lemonaddie/Depth-Anything-V2/tree/main")
    print(f"\nAnd place the file at:")
    print(f"  {model_path}")
    print(f"\n{'='*80}\n")
    
    return False


def main():
    """Main setup function"""
    print("\n" + "="*80)
    print("YOLO + DepthAnything V2 - Model Setup")
    print("="*80 + "\n")
    
    # Check YOLO model
    yolo_model = Path('data/models/IDDDYOLO11m.pt')
    if yolo_model.exists():
        print(f"✓ YOLO model found: {yolo_model}")
    else:
        print(f"✗ YOLO model not found: {yolo_model}")
        print(f"  Please place IDDDYOLO11m.pt in data/models/")

    # Download depth models
    print("\nSetting up DepthAnything V2 models...")
    
    model_types = ['vits', 'vitb', 'vitl', 'vitg']
    success_count = 0
    
    for model_type in model_types:
        if download_depth_model(model_type):
            success_count += 1
        print()
    
    print("="*80)
    if success_count > 0:
        print(f"✓ Setup completed! ({success_count}/{len(model_types)} models ready)")
    else:
        print(f"✗ Setup incomplete. Please download models manually.")
    print("="*80 + "\n")
    
    return success_count > 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

