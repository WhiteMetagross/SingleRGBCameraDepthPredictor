<#
Setup script for SingleRGBCameraDepthPredictor
- Creates and activates a virtual environment
- Upgrades pip/setuptools/wheel
- Installs requirements.txt and newFastReID
- Creates data/ models/checkpoints/configs, outputs/, logs/
- Copies ReID files from known locations if available
- Verifies required model files and prints next steps

Run from project root in PowerShell:
  powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1 -Verbose
#>

[CmdletBinding()]
param(
  [string]$VenvName = 'SingleRGBCameraDepthPredictor_venv',
  [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info($m){ Write-Host $m -ForegroundColor Cyan }
function Write-Warn($m){ Write-Host $m -ForegroundColor Yellow }
function Write-Ok($m){ Write-Host $m -ForegroundColor Green }
function Write-Err($m){ Write-Host $m -ForegroundColor Red }

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
Write-Info "Working directory: $root"

# 1) Python detection
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) { Write-Err 'Python was not found on PATH. Please install Python 3.11+ and re-run.'; exit 1 }
$python = $pythonCmd.Source
Write-Ok "Python: $python"

# 2) Create virtual environment
$venvPath = Join-Path $root $VenvName
if ((Test-Path $venvPath) -and -not $Force) {
  Write-Warn "Virtual environment directory already exists: $venvPath (use -Force to recreate)."
} elseif (Test-Path $venvPath) {
  Write-Info "Removing existing venv (Force specified)..."
  Remove-Item -Recurse -Force $venvPath
}

if (-not (Test-Path $venvPath)) {
  Write-Info "Creating virtual environment at: $venvPath"
  & $python -m venv $venvPath
  Write-Ok "Virtual environment created."
}

$pyExe = Join-Path $venvPath 'Scripts/python.exe'
$pipExe = Join-Path $venvPath 'Scripts/pip.exe'
if (-not (Test-Path $pyExe)) { Write-Err "Python executable not found in venv: $pyExe"; exit 1 }

# 3) Upgrade pip and build tools
Write-Info 'Upgrading pip, setuptools, wheel...'
& $pyExe -m pip install --upgrade pip setuptools wheel

# 4) Install PyTorch with CUDA 12.1 support
Write-Info 'Installing PyTorch with CUDA 12.1 support...'
& $pyExe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
Write-Ok 'PyTorch with CUDA 12.1 installed.'

# 5) Install remaining dependencies from requirements.txt
Write-Info 'Installing remaining dependencies from requirements.txt...'
& $pyExe -m pip install -r requirements.txt
Write-Ok 'Base requirements installed.'

Write-Info 'Installing newFastReID from GitHub (fastreid)...'
& $pyExe -m pip install git+https://github.com/WhiteMetagross/newFastReID.git
Write-Ok 'newFastReID installed.'

# 6) Create directories
$dirs = @('data/models','data/checkpoints','data/configs','outputs','logs')
foreach($d in $dirs){ New-Item -ItemType Directory -Force -Path $d | Out-Null }
Write-Ok 'Project directories ensured.'

# 7) Copy ReID resources if present from known locations
$reidCheckpointSrc = 'C:\Users\Xeron\Videos\PrayagIntersection\veriwild_bot_R50-ibn.pth'
$reidConfigSrc     = 'C:\Users\Xeron\OneDrive\Documents\Programs\VehiclePathBoTSORTTracker\VehiclePathBoTSORTTrackerV2\veriwild_r50_ibn_config.yml'
$reidCheckpointDst = 'data/checkpoints/veriwild_bot_R50-ibn.pth'
$reidConfigDst     = 'data/configs/veriwild_r50_ibn_config.yml'

if (Test-Path $reidCheckpointSrc) { Copy-Item $reidCheckpointSrc -Destination $reidCheckpointDst -Force; Write-Ok "Copied ReID checkpoint to $reidCheckpointDst" } else { Write-Warn "ReID checkpoint not found at $reidCheckpointSrc (optional)." }
if (Test-Path $reidConfigSrc)     { Copy-Item $reidConfigSrc     -Destination $reidConfigDst     -Force; Write-Ok "Copied ReID config to $reidConfigDst" }     else { Write-Warn "ReID config not found at $reidConfigSrc (optional)." }

# 7) Verify required model files
$yoloModel   = 'data/models/IDDDYOLO11m.pt'
$depthModelL = 'data/models/depth_anything_v2_vitl.pth'
$yoloOk   = Test-Path $yoloModel
$depthOk  = Test-Path $depthModelL
$reidOk   = Test-Path $reidCheckpointDst
$reidCfgOk= Test-Path $reidConfigDst

Write-Info 'Model verification:'
if ($yoloOk) { Write-Ok   "✓ YOLO model: $yoloModel" } else { Write-Warn "✗ YOLO model missing: $yoloModel" }
if ($depthOk){ Write-Ok   "✓ DepthAnything V2 (vitl): $depthModelL" } else { Write-Warn "✗ Depth model missing: $depthModelL" }
if ($reidOk){ Write-Ok    "✓ ReID checkpoint: $reidCheckpointDst" } else { Write-Warn "⚠ Optional - ReID checkpoint missing: $reidCheckpointDst" }
if ($reidCfgOk){ Write-Ok  "✓ ReID config: $reidConfigDst" } else { Write-Warn "⚠ Optional - ReID config missing: $reidConfigDst" }

if (-not $depthOk) {
  Write-Info "Download DepthAnything V2 weights with:"
  Write-Host "  python setup_models.py" -ForegroundColor Magenta
}

# 8) Verify model files
$yoloModel   = 'data/models/IDDDYOLO11m.pt'
$depthModelL = 'data/models/depth_anything_v2_vitl.pth'
$yoloOk   = Test-Path $yoloModel
$depthOk  = Test-Path $depthModelL
$reidOk   = Test-Path $reidCheckpointDst
$reidCfgOk= Test-Path $reidConfigDst

Write-Info 'Model verification:'
if ($yoloOk) { Write-Ok   "✓ YOLO model: $yoloModel" } else { Write-Warn "✗ YOLO model missing: $yoloModel" }
if ($depthOk){ Write-Ok   "✓ DepthAnything V2 (vitl): $depthModelL" } else { Write-Warn "✗ Depth model missing: $depthModelL" }
if ($reidOk){ Write-Ok    "✓ ReID checkpoint: $reidCheckpointDst" } else { Write-Warn "⚠ Optional - ReID checkpoint missing: $reidCheckpointDst" }
if ($reidCfgOk){ Write-Ok  "✓ ReID config: $reidConfigDst" } else { Write-Warn "⚠ Optional - ReID config missing: $reidConfigDst" }

if (-not $depthOk) {
  Write-Info "Download DepthAnything V2 weights with:"
  Write-Host "  python setup_models.py" -ForegroundColor Magenta
}

# 9) Print activation and usage
Write-Host "`nActivation and quick test:" -ForegroundColor Cyan
Write-Host "  .\\$VenvName\\Scripts\\Activate.ps1" -ForegroundColor DarkCyan
Write-Host "  python --version" -ForegroundColor DarkCyan
Write-Host "  python main.py --help" -ForegroundColor DarkCyan

# 10) Summary
Write-Host "`nSummary:" -ForegroundColor Cyan
Write-Host "  Venv: $venvPath" -ForegroundColor Gray
Write-Host "  Directories ensured: $($dirs -join ', ')" -ForegroundColor Gray
Write-Host "  YOLO present: $yoloOk" -ForegroundColor Gray
Write-Host "  DepthAny (vitl) present: $depthOk" -ForegroundColor Gray
Write-Host "  ReID checkpoint present: $reidOk (optional)" -ForegroundColor Gray
Write-Host "  ReID config present: $reidCfgOk (optional)" -ForegroundColor Gray

Write-Ok "Setup complete. See README.md and Usage.md for next steps."
