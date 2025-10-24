## Data Folder Structure

Centralized model and resource storage used across the project.
- `data/models/` - YOLO and DepthAnything V2 model weights.
- `data/checkpoints/` - ReID checkpoints (e.g., VeRiWild .pth).
- `data/configs/` - ReID configuration files (YAML).

Update `config/config.yaml` under `paths` to point to these locations.

## VehicleTracker and ReID

- Tracking uses the BoTSORT algorithm with motion (Kalman filter) and appearance features.
- Appearance features are extracted via the newFastReID library (imported as `fastreid`).
- VeRiWild model is used for vehicle re-identification.
- Association combines IoU and cosine-similarity of ReID embeddings.

## Temporal Smoothing

Temporal smoothing is applied to per-object metric distances (not depth maps).
- Implemented in `models/distance_calculator.py` using an exponential moving average per track ID.
- Configured in `config/config.yaml` under `distance_validation.temporal_smoothing`.

# Codebase Index and Technical Architecture

This document provides a comprehensive overview of the project structure, module descriptions, and API reference for the YOLO + DepthAnything V2 system.

## Architecture Overview

The system follows a modular architecture with clear separation of concerns. The main pipeline orchestrates multiple specialized components that handle detection, depth estimation, distance calculation, tracking, and visualization.

### Core Components

The system consists of the following core components:

- **YOLODetector** - Wraps the YOLO11 model for vehicle detection.
- **DepthEstimator** - Wraps the DepthAnything V2 model for monocular depth estimation.
- **DistanceCalculator** - Converts depth maps to real-world distances with spatial validation.
- **BonnetCalibrator** - Uses vehicle bonnet as reference for metric calibration.
- **VehicleTracker** - Implements BoTSORT algorithm for multi-object tracking.
- **Visualizer** - Handles drawing detections, distances, and depth maps on frames.
- **VideoProcessor** - Manages video input and output operations.
- **Pipeline** - Orchestrates all components for end-to-end processing.

## Module Descriptions

### models/yolo_detector.py

The YOLODetector class wraps the Ultralytics YOLO model for vehicle detection. It loads the pre-trained model, runs inference on frames, and returns detections with bounding boxes and confidence scores.

**Key Methods:**

- `__init__(model_path, confidence_threshold, iou_threshold, device)` - Initializes the detector with model weights and detection parameters.
- `detect(frame)` - Detects objects in a frame and returns a list of detections.

### models/depth_estimator.py

The DepthEstimator class wraps the DepthAnything V2 model for monocular depth estimation. It supports multiple model sizes (vits, vitb, vitl, vitg) for different speed/accuracy tradeoffs.

**Key Methods:**

- `__init__(model_type, device, input_size, calibration_config)` - Initializes the depth estimator with model configuration.
- `estimate_depth(frame)` - Estimates depth map for a given frame.
- `_apply_calibration(depth_map, calibration_config)` - Applies calibration to convert relative depth to metric distances.

### models/distance_calculator.py

The DistanceCalculator class converts depth maps and bounding boxes to real-world distances. It includes perspective-aware angular correction, spatial consistency validation, and temporal smoothing for stable distance measurements.

**Key Methods:**

- `__init__(use_center, use_median, enable_spatial_validation, max_distance, min_distance, temporal_smoothing_enabled, temporal_window_size, temporal_smoothing_factor)` - Initializes the calculator with distance extraction and temporal smoothing parameters.
- `calculate_distance(depth_map, detection)` - Calculates raw depth value for a detected object.
- `apply_temporal_smoothing(distances, track_ids)` - Applies exponential moving average smoothing to distances using track IDs for per-object history.
- `_apply_perspective_correction(depth_value, obj_x, obj_y)` - Applies angular correction based on pixel offset from bonnet center.
- `validate_spatial_consistency(distances, detections)` - Validates that nearby objects have similar distances and detects outliers.

### models/bonnet_calibrator.py

The BonnetCalibrator class uses the vehicle's bonnet as a reference point for metric calibration. It detects the bonnet in each frame and maintains a temporal smoothing window for stable calibration factors.

**Key Methods:**

- `__init__(reference_distance, smoothing_window, min_bonnet_area)` - Initializes the calibrator with reference distance and smoothing parameters.
- `find_bonnet(detections, depth_map, frame_height)` - Identifies the bonnet in detections using position and size heuristics.
- `get_calibration_factor(bonnet_detection, depth_map)` - Calculates the calibration factor from bonnet depth.
- `set_bonnet_reference(bonnet_detection, depth_map)` - Updates the calibration factor when bonnet is detected.

### models/vehicle_tracker.py

The VehicleTracker class implements the BoTSORT algorithm for multi-object tracking. It uses Kalman filtering for motion prediction and optional ReID features for appearance-based association.

**Key Methods:**

- `__init__(max_disappeared, min_hits, iou_threshold, use_reid)` - Initializes the tracker with association parameters.
- `update(detections, frame)` - Updates tracks with new detections and returns tracked objects with persistent IDs.
- `_associate_detections(detections)` - Associates detections to existing tracks using IoU and optional ReID features.

### utils/logger.py

The logger module provides logging utilities for the pipeline. It sets up console and file handlers with consistent formatting.

**Key Functions:**

- `setup_logger(name, log_file, level)` - Creates and configures a logger instance with specified handlers and level.

### utils/video_processor.py

The VideoProcessor class handles video input operations including frame reading and property extraction.

**Key Methods:**

- `__init__(video_path)` - Opens the video file and extracts properties (FPS, resolution, frame count).
- `read_frame()` - Reads the next frame from the video.
- `get_frame_at(frame_number)` - Retrieves a frame at a specific position.
- `release()` - Closes the video file.

### utils/visualization.py

The Visualizer class handles drawing detections, distances, and depth maps on frames.

**Key Methods:**

- `__init__(font_scale, thickness, colormap)` - Initializes the visualizer with drawing parameters.
- `draw_detections(frame, detections, distances)` - Draws bounding boxes and distance labels.
- `draw_tracked_detections(frame, tracked_detections, distances)` - Draws tracked objects with track IDs.
- `draw_depth_map(depth_map)` - Converts depth map to colored visualization using specified colormap.
- `draw_info_panel(frame, info_dict, position)` - Draws information panel with status and statistics.

### pipeline/main_pipeline.py

The Pipeline class orchestrates all components for end-to-end video processing.

**Key Methods:**

- `__init__(config_path)` - Initializes the pipeline by loading configuration and setting up all models.
- `process_video(input_video, output_standard, output_depth)` - Processes a video file and generates two output videos.
- `_process_frame_dual(frame)` - Processes a single frame and returns both standard and depth visualization outputs.
- `_initialize_models()` - Loads and initializes all required models based on configuration.

## Configuration Parameters

The system is configured through `config/config.yaml`. Key configuration sections include:

- **paths** - File paths for models, input videos, and output directories.
- **yolo** - YOLO detection parameters including confidence and IOU thresholds.
- **depth** - Depth estimation model type and device selection.
- **calibration** - Bonnet calibration settings including reference distance and temporal smoothing.
- **tracking** - Vehicle tracking parameters including association thresholds.
- **distance_validation** - Spatial consistency validation settings.
- **visualization** - Drawing options and colormap selection.
- **logging** - Logging level and output file path.

## Data Flow

The processing pipeline follows this data flow:

1. Video frames are read from the input video file.
2. YOLO detector processes each frame and returns vehicle detections.
3. DepthAnything V2 estimates depth map for the frame.
4. BonnetCalibrator identifies bonnet and calculates calibration factor.
5. DistanceCalculator converts depth values to real-world distances.
6. VehicleTracker associates detections to existing tracks.
7. Temporal smoothing is applied to distance values using track IDs for per-object history.
8. Visualizer draws detections, distances, and depth information.
9. Processed frames are written to output video files.

## Temporal Distance Smoothing

The system implements temporal smoothing for final distance calculations to prevent unstable fluctuations between frames. This feature is particularly important for ADAS applications where sudden distance changes can trigger false alarms.

**How It Works:**

1. Each tracked object maintains a distance history based on its track ID.
2. When a new distance is calculated, it is added to the object's history.
3. Exponential moving average (EMA) is applied: `smoothed_distance = factor * current + (1-factor) * historical_average`.
4. The smoothing factor controls the weight of recent measurements (0.7 = 70% current, 30% historical).
5. A window size limits the history length to prevent stale data from affecting smoothing.
6. When a track is lost, its history is automatically cleaned up.

**Configuration:**

```yaml
distance_validation:
  temporal_smoothing:
    enabled: true
    smoothing_factor: 0.7
    window_size: 5
```

**Benefits:**

- Prevents sudden distance jumps (e.g., 3.5m → 13m) caused by depth estimation noise.
- Provides stable distance readings for ADAS decision-making.
- Per-object smoothing ensures each vehicle is tracked independently.
- Configurable parameters allow tuning for different use cases.

## Extension Points

The modular architecture provides several extension points for customization:

- Replace YOLODetector with alternative detection models.
- Implement different depth estimation models in DepthEstimator.
- Add alternative calibration methods in BonnetCalibrator.
- Extend VehicleTracker with different tracking algorithms.
- Add custom visualization methods in Visualizer.
- Implement alternative distance calculation methods in DistanceCalculator.

## Dependencies

The project depends on the following key libraries:

- PyTorch for deep learning model inference.
- Ultralytics YOLO for object detection.
- OpenCV for video processing and visualization.
- NumPy for numerical computations.
- SciPy for optimization and spatial operations.
- FilterPy for Kalman filtering in tracking.
- FastReID for appearance-based re-identification (optional).
- PySide6 for the interactive calibration GUI (Qt6 Python bindings).

## Performance Considerations

The system is optimized for real-time processing on GPU hardware. Key performance considerations include:

- Model inference is performed on GPU for speed.
- Depth maps are processed at reduced resolution for memory efficiency.
- Temporal smoothing reduces computational overhead for calibration.
- Spatial validation is performed only on nearby objects to reduce complexity.
- Optional features (tracking, segmentation) can be disabled to improve speed.

---

For detailed installation instructions, see InstallationAndSetup.md. For usage examples, see README.md.

