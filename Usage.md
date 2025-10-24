# Usage Guide:

This guide shows how to run the SingleRGBCameraDepthPredictor pipeline, calibrate the system, configure options, and understand outputs.

## Basic Commands:

- Show help:

```bash
python main.py --help
```

- Process a video (default config):

```bash
python main.py -i "path/to/video.mp4"
```

- Limit frames for a quick test (e.g., first 150 frames):

```bash
python main.py -i "path/to/video.mp4" --max-frames 150
```

- Set output file prefix (videos saved in `outputs/`):

```bash
python main.py -i "path/to/video.mp4" --output-prefix "run1"
```

## Calibration Workflow (Bonnet-Based):

The system uses a bonnet-based calibration to convert DepthAnything V2 relative depth into metric distances.

1. When you start the pipeline, a PySide6 (Qt6) GUI opens showing the depth heatmap.
2. Click on the bonnet (hood) center to select the reference point. A black dot and crosshair indicate your selection.
3. Press `R` to edit the reference distance (default 1.35 meters) if needed.
4. Optionally press `B` to auto-detect the bonnet using YOLO and use its bounding box center.
5. Press `C` or Enter to confirm and start processing.
6. A single scale factor is computed once and applied for the entire run.

![Bonnet Calibration Interface](../visuals/BonnetCalibratiorWindow.png)

The calibration interface provides an interactive depth heatmap where users can precisely select the bonnet reference point. The system displays the selected coordinates and depth value in real-time for accurate calibration.

Notes:
- DepthAnything V2 outputs inverse depth (larger value = closer). The distance conversion uses `distance_m = scale_factor / depth_value`.
- Calibration results are shown in the on-screen information panel.

## Configuration (config/config.yaml):

Key sections:

- `paths`:
  - `yolo_model`: YOLO weights in `data/models/`.
  - `depth_model`: DepthAnything V2 weights in `data/models/`.
  - `reid_model`: VeRiWild checkpoint in `data/checkpoints/`.
  - `reid_config`: VeRiWild config in `data/configs/`.
  - `input_video`, `output_dir`.

- `yolo`:
  - `confidence_threshold`, `iou_threshold`, `device`.

- `depth`:
  - `model_type`: one of `vits`, `vitb`, `vitl`, `vitg`.
  - `device`.

- `calibration`:
  - Bonnet reference distance and related GUI options.

- `tracking`:
  - BoTSORT parameters and ReID usage.

- `distance_validation.temporal_smoothing`:
  - `enabled`, `smoothing_factor`, `window_size` for per-object EMA over distances.

## Output Files:

The pipeline writes two MP4 files to the `outputs/` directory using your `--output-prefix`:

- `<prefix>_standard.mp4`: RGB frames with bounding boxes, distances, and track IDs.
- `<prefix>_depth_visualization.mp4`: Depth heatmap overlay with the same annotations.

![Standard Output Example](../visuals/MainRoadVideoWithPredictionsOnRGBCam.gif)

The standard output displays detected vehicles with bounding boxes, class labels, and real-time distance measurements overlaid on the original RGB video frames.

![Depth Visualization Example](../visuals/MainRoadVideoWithDepthHeatMap.gif)

The depth visualization output shows the monocular depth estimation as a color heatmap with detected vehicles and distance measurements overlaid for visual analysis.

## Examples:

- High-confidence detection, smaller IOU threshold:

```bash
python main.py -i "video.mp4" --output-prefix "exp1" --conf-threshold 0.4 --iou-threshold 0.35
```

- Use a smaller depth model for speed:

```bash
python main.py -i "video.mp4" --output-prefix "fast" --depth-model-type vits
```

- Disable tracking (no BoTSORT/ReID):

```bash
python main.py -i "video.mp4" --output-prefix "no_track" --disable-tracking
```

- Run a short smoke test:

```bash
python main.py -i "video.mp4" --output-prefix "smoke" --max-frames 50
```

## Logging:

- Logs are written to `logs/pipeline.log` with INFO-level messages.
- Increase verbosity by editing the logging level in `config/config.yaml`.

## Tips:

- Ensure models are in `data/models/` and ReID resources in `data/checkpoints/` and `data/configs/`.
- If the GUI fails to start due to Qt plugin errors, re-run from an elevated terminal or ensure PySide6 is installed.
- Adjust smoothing if distances fluctuate (e.g., `smoothing_factor: 0.7`, `window_size: 5`).

## Troubleshooting:

- If YOLO fails to load: verify `paths.yolo_model` points to `data/models/IDDDYOLO11m.pt`.
- If DepthAnything V2 model is missing: download into `data/models/` or run `python setup_models.py`.
- If ReID is disabled due to missing files, either place the VeRiWild files or set tracking to not use ReID.

