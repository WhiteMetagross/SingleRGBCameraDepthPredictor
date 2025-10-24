# Installation and Setup Guide

This document provides comprehensive instructions for setting up the YOLO + DepthAnything V2 system on your machine.

## System Requirements

### Hardware Requirements

The system requires the following hardware for optimal performance:

- NVIDIA GPU with CUDA compute capability 7.0 or higher (RTX 2060 or newer recommended).
- Minimum 8 GB VRAM for core features (detection and depth estimation).
- Minimum 12 GB VRAM for all features including tracking and segmentation.
- 16 GB system RAM recommended.
- 50 GB free disk space for models and output videos.

### Software Requirements

The system requires the following software:

- Windows 10 or later, or Linux with CUDA support.
- Python 3.11 or later.
- CUDA 12.1 or later.
- cuDNN 8.x or later.
- Git for cloning repositories.

## Installation Steps

### Step 1: Verify Python Installation

Verify that Python 3.11 is installed and accessible from the command line.

```bash
python --version
```

If Python is not installed, download it from https://www.python.org/ and ensure it is added to your system PATH.

### Step 2: Verify CUDA Installation

Verify that CUDA 12.1 is installed and PyTorch can access it.

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
```

If CUDA is not available, install CUDA 12.1 from https://developer.nvidia.com/cuda-12-1-0-download-archive.

### Step 3: Clone or Navigate to Project Directory

Navigate to the project directory or clone it if not already present.

```bash
cd OneDrive/Documents/Programs/SingleRGBCameraDepthPredictor
```

### Step 4: Create Virtual Environment

Create a Python virtual environment to isolate project dependencies.

```bash
python -m venv venv
```

Activate the virtual environment:

On Windows:
```bash
venv\Scripts\activate
```

On Linux/Mac:
```bash
source venv/bin/activate
```

### Step 5: Upgrade pip

Upgrade pip to the latest version.

```bash
python -m pip install --upgrade pip
```

### Step 6: Install Dependencies

Install all required Python packages from the requirements file.

```bash
pip install -r requirements.txt
```

If you encounter issues with PyTorch, install it explicitly for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 6.5: Install PySide6 for Calibration GUI

The interactive calibration GUI requires PySide6 (Qt6 Python bindings). Install it explicitly:

```bash
pip install PySide6>=6.0.0
```

Verify the installation:

```bash
python -c "import PySide6; print(f'PySide6 version: {PySide6.__version__}')"
```
### Step 6.6: Install newFastReID (ReID Library)

Install the newFastReID library for appearance-based re-identification (imported as `fastreid`).

```bash
pip install git+https://github.com/WhiteMetagross/newFastReID.git
```

Verify installation:

```bash
python -c "import fastreid; print('fastreid OK')"
```


PySide6 is required for the interactive bonnet calibration window that opens before video processing. It provides a professional GUI for selecting the bonnet reference point with visual feedback and keyboard controls.

Additionally, prepare ReID resources:
- Copy VeRiWild checkpoint file (e.g., `veriwild_bot_R50-ibn.pth`) into `data/checkpoints/`.
- Copy VeRiWild config file (e.g., `veriwild_r50_ibn_config.yml`) into `data/configs/`.
- Ensure `paths.reid_model` and `paths.reid_config` in `config/config.yaml` point to these files.

### Step 7: Download Model Weights

Download the DepthAnything V2 model weights using the provided setup script.

```bash
python setup_models.py
```

Alternatively, download manually from HuggingFace:

- Visit https://huggingface.co/lemonaddie/Depth-Anything-V2/tree/main
- Download `depth_anything_v2_vits.pth` (or other model variants)
- Place the file in the `data/models/` directory.

### Step 8: Verify Installation

Verify that all components are installed correctly by running a test.

```bash
python -c "from pipeline.main_pipeline import Pipeline; print('Installation successful')"
```

### Step 9: Configure Paths

Edit `config/config.yaml` to set the correct paths for your system:

- `paths.reid_model` - Path to ReID model file.
- `paths.reid_config` - Path to ReID configuration file.
- `paths.input_video` - Path to input video file.
- `paths.output_dir` - Directory for output videos.

### Step 10: Configure Temporal Distance Smoothing

The system includes temporal smoothing for distance calculations to prevent unstable fluctuations. Configure it in `config/config.yaml`:

```yaml
distance_validation:
  temporal_smoothing:
    enabled: true
    smoothing_factor: 0.7
    window_size: 5
```

**Configuration Parameters:**

- `enabled` - Set to `true` to enable temporal smoothing, `false` to disable.
- `smoothing_factor` - Weight for current measurement (0.0-1.0). Higher values (0.7-0.9) give more weight to recent measurements. Lower values (0.3-0.5) provide more smoothing.
- `window_size` - Number of frames to maintain in history (typically 3-10). Larger values provide more smoothing but may lag behind actual changes.

**Recommended Settings:**

- **Stable Distances** - `smoothing_factor: 0.7, window_size: 5` (default, good balance).
- **More Responsive** - `smoothing_factor: 0.9, window_size: 3` (follows changes quickly).
- **More Stable** - `smoothing_factor: 0.5, window_size: 10` (smoother but slower to respond).

## Recommended: Automated Setup via setup.ps1

On Windows, the preferred way to install and prepare the project is with the automated PowerShell script:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Verbose
```

This script will:
- Create and initialize a virtual environment.
- Upgrade pip, setuptools, and wheel.
- Install dependencies from requirements.txt.
- Install newFastReID from GitHub.
- Create required folders: `data/models`, `data/checkpoints`, `data/configs`, `outputs`, `logs`.
- Copy VeRiWild ReID files from known locations if available.
- Verify model files and print next steps if any are missing.

After the script completes, activate the environment and run a quick check:

```powershell
.\SingleRGBCameraDepthPredictor_venv\Scripts\Activate.ps1
python --version
python main.py --help
```

## Automated Setup (PowerShell)

On Windows, you can use the provided PowerShell setup script for automated installation.

```powershell
.\setup.ps1
```

The script will:

1. Check Python installation.
2. Create a virtual environment.
3. Upgrade pip.
4. Install all dependencies.
5. Create required directories.
6. Verify model files.
7. Test Python imports.

## Troubleshooting

### Issue: Python Not Found

Ensure Python 3.11 is installed and added to your system PATH. Verify with:

```bash
python --version
```

If not found, reinstall Python from https://www.python.org/ and ensure "Add Python to PATH" is checked during installation.

### Issue: CUDA Not Available

Verify CUDA installation and PyTorch compatibility:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If False, install CUDA 12.1 and reinstall PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Out of Memory

If you encounter out-of-memory errors:

1. Reduce depth model size in `config/config.yaml`: `depth.model_type: "vits"` instead of `vitl`.
2. Disable tracking: `tracking.enabled: false`.
3. Reduce video resolution or process fewer frames.
4. Ensure no other GPU-intensive applications are running.

### Issue: Model Download Fails

If automatic model download fails:

1. Download manually from https://huggingface.co/lemonaddie/Depth-Anything-V2/tree/main.
2. Place the file in `data/models/` directory.
3. Verify file permissions and disk space.

### Issue: Import Errors

If you encounter import errors after installation:

1. Verify virtual environment is activated.
2. Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`.
3. Check that all required packages are listed in `requirements.txt`.

### Issue: Slow Processing

If processing is slower than expected:

1. Verify GPU is being used: Check NVIDIA GPU utilization with `nvidia-smi`.
2. Use a faster depth model: `depth.model_type: "vits"` instead of `vitl`.
3. Disable optional features: tracking, segmentation.
4. Ensure no other processes are using the GPU.

## Verification Checklist

After installation, verify the following:

- Python 3.11 is installed and accessible.
- CUDA 12.1 is installed and PyTorch can access it.
- All dependencies are installed: `pip list | grep -E "torch|opencv|ultralytics|PySide6"`.
- PySide6 is installed for the calibration GUI: `python -c "import PySide6"`.
- Model files are present in `data/models/`.
- Configuration file exists at `config/config.yaml`.
- Output directory can be created: `mkdir -p data/outputs`.
- Pipeline imports successfully: `python -c "from pipeline.main_pipeline import Pipeline"`.

## Next Steps

After successful installation:

1. Review the configuration in `config/config.yaml`.
2. Prepare your input video file.
3. Run the pipeline: `python main.py -i "video.mp4" -o "output_dir"`.
4. Check output videos in the specified output directory.
5. Review logs in `logs/pipeline.log` for processing details.

For usage examples and detailed documentation, see README.md and CodebaseIndex.md.

## Support

If you encounter issues not covered in this guide:

1. Check the troubleshooting section above.
2. Review logs in `logs/pipeline.log`.
3. Verify all system requirements are met.
4. Ensure all dependencies are correctly installed.
5. Try running with a smaller test video first.

---

For quick start instructions, see README.md. For technical architecture details, see CodebaseIndex.md.

