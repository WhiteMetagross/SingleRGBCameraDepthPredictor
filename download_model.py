"""
Download DepthAnything V2 model weights from HuggingFace
"""
import os
import sys
from pathlib import Path

def download_with_wget():
    """Try downloading using wget (if available on system)"""
    import subprocess
    
    os.makedirs('data/models', exist_ok=True)

    url = 'https://huggingface.co/lemonaddie/Depth-Anything-V2/resolve/main/depth_anything_v2_vits.pth'
    output_path = 'data/models/depth_anything_v2_vits.pth'

    print(f"Attempting to download using wget...")
    try:
        subprocess.run(['wget', url, '-O', output_path], check=True)
        print(f"✓ Downloaded successfully to {output_path}")
        return True
    except Exception as e:
        print(f"✗ wget download failed: {e}")
        return False

def download_with_curl():
    """Try downloading using curl (if available on system)"""
    import subprocess
    
    os.makedirs('data/models', exist_ok=True)

    url = 'https://huggingface.co/lemonaddie/Depth-Anything-V2/resolve/main/depth_anything_v2_vits.pth'
    output_path = 'data/models/depth_anything_v2_vits.pth'

    print(f"Attempting to download using curl...")
    try:
        subprocess.run(['curl', '-L', url, '-o', output_path], check=True)
        print(f"✓ Downloaded successfully to {output_path}")
        return True
    except Exception as e:
        print(f"✗ curl download failed: {e}")
        return False

def print_manual_instructions():
    """Print manual download instructions"""
    print("\n" + "=" * 80)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 80)
    print("\nThe automatic download failed. Please download manually:")
    print("\n1. Visit: https://huggingface.co/lemonaddie/Depth-Anything-V2/tree/main")
    print("\n2. Download the file: depth_anything_v2_vits.pth")
    print("\n3. Place it in: data/models/")
    print("\n4. The file should be at:")
    print("   data/models/depth_anything_v2_vits.pth")
    print("\n" + "=" * 80)

def main():
    """Main function"""
    print("=" * 80)
    print("DepthAnything V2 Model Download")
    print("=" * 80)
    
    os.makedirs('data/models', exist_ok=True)

    # Check if model already exists
    model_path = Path('data/models/depth_anything_v2_vits.pth')
    if model_path.exists():
        print(f"\n✓ Model already exists at: {model_path}")
        print(f"  Size: {model_path.stat().st_size / (1024**3):.2f} GB")
        return
    
    print("\nAttempting to download DepthAnything V2 (vits) model...")
    print("This is a large file (~350 MB), please be patient...\n")
    
    # Try different download methods
    if download_with_wget():
        return
    
    if download_with_curl():
        return
    
    # If all automatic methods fail, print manual instructions
    print_manual_instructions()

if __name__ == '__main__':
    main()

