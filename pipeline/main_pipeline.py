# This program implements the main processing pipeline combining YOLO detection and DepthAnything V2 depth estimation.
# This program processes input video to produce two outputs: standard detections and depth visualization.

# Author: Mridankan Mandal.

import yaml
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from models import YOLODetector, DepthEstimator, DistanceCalculator, BonnetCalibrator, VehicleTracker
from utils import setup_logger, VideoProcessor, VideoWriter, Visualizer


class Pipeline:
    # This class orchestrates the main pipeline for YOLO + DepthAnything V2 processing with vehicle tracking and distance calibration.

    def __init__(self, config_path):
        # This method initializes the pipeline by loading configuration and setting up all required models and utilities.
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Setup logger
        self.logger = setup_logger(
            'Pipeline',
            log_file=self.config['logging'].get('log_file'),
            level=self.config['logging'].get('level', 'INFO')
        )

        self.logger.info("Initializing pipeline...")

        # Initialize models
        self.detector = None
        self.depth_estimator = None
        self.distance_calculator = None
        self.bonnet_calibrator = None
        self.visualizer = None
        self.vehicle_tracker = None

        self._initialize_models()

        self.logger.info("Pipeline initialized successfully.")

    def _load_config(self):
        # This method loads configuration from YAML file.
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _initialize_models(self):
        # This method initializes all models.
        # YOLO Detector
        yolo_config = self.config['yolo']
        yolo_model_path = Path(self.config['paths']['yolo_model'])

        self.logger.info(f"Loading YOLO model from {yolo_model_path}.")
        self.detector = YOLODetector(
            model_path=yolo_model_path,
            confidence_threshold=yolo_config['confidence_threshold'],
            iou_threshold=yolo_config['iou_threshold'],
            device=yolo_config['device']
        )
        self.logger.info("YOLO detector loaded.")

        # DepthAnything V2
        depth_config = self.config['depth']
        self.logger.info(f"Loading DepthAnything V2 ({depth_config['model_type']}).")
        try:
            # Get calibration config
            calibration_config = depth_config.get('calibration', {})

            self.depth_estimator = DepthEstimator(
                model_type=depth_config['model_type'],
                device=depth_config['device'],
                input_size=depth_config.get('input_size', 518),
                calibration_config=calibration_config
            )
            self.logger.info("DepthAnything V2 loaded.")
            self.logger.info(f"Depth calibration enabled: windshield_filter={calibration_config.get('windshield_filter_ratio', 0.15)}, min_depth={calibration_config.get('min_depth', 0.5)}m, max_depth={calibration_config.get('max_depth', 100.0)}m.")
        except FileNotFoundError as e:
            self.logger.error(f"Failed to load DepthAnywhere V2: {e}.")
            self.logger.error("Please download the model weights first.")
            raise

        # Distance Calculator with spatial consistency validation and temporal smoothing
        distance_config = self.config['distance']
        validation_config = self.config.get('distance_validation', {})
        temporal_config = validation_config.get('temporal_smoothing', {})
        self.distance_calculator = DistanceCalculator(
            use_center=distance_config['use_center'],
            use_median=distance_config['use_median'],
            enable_spatial_validation=validation_config.get('enabled', True),
            max_distance=validation_config.get('max_distance', 100.0),
            min_distance=validation_config.get('min_distance', 0.5),
            spatial_smoothing=validation_config.get('spatial_smoothing', True),
            outlier_threshold=validation_config.get('outlier_threshold', 2.0),
            temporal_smoothing_enabled=temporal_config.get('enabled', False),
            temporal_window_size=temporal_config.get('window_size', 5),
            temporal_smoothing_factor=temporal_config.get('smoothing_factor', 0.7)
        )
        self.logger.info("Distance calculator initialized with spatial validation.")
        if temporal_config.get('enabled', False):
            self.logger.info(f"Temporal smoothing enabled (window={temporal_config.get('window_size', 5)}, factor={temporal_config.get('smoothing_factor', 0.7)}).")

        # Bonnet Calibrator for metric distance conversion
        calibration_config = self.config.get('calibration', {})
        self.bonnet_calibrator = BonnetCalibrator(
            reference_distance=calibration_config.get('reference_distance', 1.35),
            bonnet_depth_reference=calibration_config.get('bonnet_depth_reference', None)
        )
        calib_info = self.bonnet_calibrator.get_calibration_info()
        if calib_info['calibration_complete']:
            self.logger.info(f"Bonnet calibrator initialized with scale factor: {calib_info['calibration_scale_factor']:.6f}.")
            self.logger.info(f"Method: {calib_info['calibration_method']}.")
        else:
            self.logger.warning("Bonnet calibrator not yet calibrated. Distances will be invalid.")

        # Visualizer
        vis_config = self.config['visualization']
        self.visualizer = Visualizer(
            font_scale=vis_config['font_scale'],
            thickness=vis_config['thickness'],
            colormap=vis_config['depth_colormap']
        )

        # Vehicle Tracker (optional)
        tracking_config = self.config.get('tracking', {})
        if tracking_config.get('enabled', False):
            try:
                reid_model = self.config['paths'].get('reid_model')
                reid_config = self.config['paths'].get('reid_config')

                if reid_model and reid_config:
                    self.vehicle_tracker = VehicleTracker(
                        reid_model_path=reid_model,
                        reid_config_path=reid_config,
                        max_disappeared=tracking_config.get('max_disappeared', 30),
                        min_hits=tracking_config.get('min_hits', 3),
                        iou_threshold=tracking_config.get('iou_threshold', 0.3),
                        device=yolo_config['device'],
                        use_reid=tracking_config.get('use_reid', True)
                    )
                    self.logger.info("Vehicle tracker initialized with ReID.")
                else:
                    self.vehicle_tracker = VehicleTracker(
                        max_disappeared=tracking_config.get('max_disappeared', 30),
                        min_hits=tracking_config.get('min_hits', 3),
                        iou_threshold=tracking_config.get('iou_threshold', 0.3),
                        device=yolo_config['device'],
                        use_reid=False
                    )
                    self.logger.info("Vehicle tracker initialized (IoU-only).")
            except Exception as e:
                self.logger.warning(f"Failed to initialize vehicle tracker: {e}.")
                self.vehicle_tracker = None

    def process_video(self, output_standard=None, output_depth=None):
        # This method processes video file and generates two output videos: standard detection output and depth visualization output.
        input_video = self.config['paths']['input_video']

        # Set default output paths
        output_dir = Path(self.config['paths']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_standard is None:
            output_standard = output_dir / 'output_standard.mp4'
        if output_depth is None:
            output_depth = output_dir / 'output_depth_visualization.mp4'

        output_standard = Path(output_standard)
        output_depth = Path(output_depth)

        self.logger.info(f"Processing video: {input_video}.")
        self.logger.info(f"Standard output: {output_standard}.")
        self.logger.info(f"Depth visualization output: {output_depth}.")

        # Open input video
        with VideoProcessor(input_video) as video_reader:
            video_info = video_reader.get_info()
            self.logger.info(f"Video info: {video_info}")

            video_config = self.config['video']
            output_fps = video_config.get('output_fps', video_info['fps'])

            # Create two output video writers
            with VideoWriter(
                output_standard,
                fps=output_fps,
                width=video_info['width'],
                height=video_info['height'],
                codec=video_config.get('output_codec', 'mp4v')
            ) as writer_standard, VideoWriter(
                output_depth,
                fps=output_fps,
                width=video_info['width'],
                height=video_info['height'],
                codec=video_config.get('output_codec', 'mp4v')
            ) as writer_depth:

                frame_idx = 0
                skip_frames = video_config.get('skip_frames', 1)
                max_frames = video_config.get('max_frames')

                # Process frames
                pbar = tqdm(total=video_info['total_frames'], desc="Processing")

                while True:
                    success, frame = video_reader.read_frame()
                    if not success:
                        break

                    # Skip frames if needed
                    if frame_idx % skip_frames != 0:
                        frame_idx += 1
                        pbar.update(1)
                        continue

                    # Check max frames
                    if max_frames and frame_idx >= max_frames:
                        break

                    # Process frame and get both outputs
                    frame_standard, frame_depth = self._process_frame_dual(frame)

                    # Write to both outputs
                    writer_standard.write_frame(frame_standard)
                    writer_depth.write_frame(frame_depth)

                    frame_idx += 1
                    pbar.update(1)

                pbar.close()

        self.logger.info(f"Standard output saved to: {output_standard}.")
        self.logger.info(f"Depth visualization output saved to: {output_depth}.")
        return output_standard, output_depth

    def _process_frame_dual(self, frame):
        # This method processes single frame and generates two outputs: standard detection output and depth visualization output.
        # Detect objects
        detections = self.detector.detect(frame)

        # Estimate depth
        depth_map = self.depth_estimator.estimate_depth(frame)

        # Check if calibration is complete
        calib_info = self.bonnet_calibrator.get_calibration_info()
        bonnet_detected = calib_info['calibration_complete']

        # Calculate raw depth distances (relative depth values from DepthAnything V2)
        raw_depths = []
        for det in detections:
            depth_value = self.distance_calculator.calculate_distance(depth_map, det)
            raw_depths.append(depth_value)

        # Convert raw depths to metric distances using pre-calculated scale factor
        # Formula: metric_distance = scale_factor / object_depth
        # where scale_factor = reference_distance * bonnet_depth_reference (from calibration)
        # This accounts for DepthAnything V2's inverse relationship: LARGER depth values = CLOSER objects
        metric_distances = []
        for raw_depth in raw_depths:
            metric_dist = self.bonnet_calibrator.convert_depth_to_meters(raw_depth)
            metric_distances.append(metric_dist)

        # Validate spatial consistency on METRIC distances (post-calibration)
        if len(detections) > 1:
            metric_distances, outlier_flags = self.distance_calculator.validate_spatial_consistency(
                metric_distances, detections
            )
            # Log outliers for debugging
            for i, is_outlier in enumerate(outlier_flags):
                if is_outlier:
                    self.logger.debug(f"Metric distance outlier for detection {i}: {metric_distances[i]:.2f}m")

        # Apply vehicle tracking if enabled
        tracked_detections = None
        tracked_class_names = None
        track_ids = []
        if self.vehicle_tracker is not None:
            # Convert detections to format expected by tracker
            det_array = []
            class_names = []
            for det in detections:
                det_array.append([det['x1'], det['y1'], det['x2'], det['y2'], det.get('confidence', 0.5)])

                class_names.append(det.get('class_name', 'Vehicle'))

            # Update tracker with class names
            tracked_detections, tracked_class_names = self.vehicle_tracker.update(det_array, frame, class_names)

            # Extract track IDs from tracked detections (last column)
            if tracked_detections is not None and len(tracked_detections) > 0:
                track_ids = [int(det[4]) for det in tracked_detections]

            # Apply temporal smoothing to metric distances using track IDs
            if self.distance_calculator.temporal_smoothing_enabled and len(track_ids) > 0:
                metric_distances = self.distance_calculator.apply_temporal_smoothing(
                    metric_distances, track_ids
                )

        # Create standard output (bboxes + metric distances, no depth overlay)
        frame_standard = frame.copy()

        # Draw tracked detections if available, otherwise draw regular detections
        if tracked_detections is not None and len(tracked_detections) > 0:
            frame_standard = self.visualizer.draw_tracked_detections(
                frame_standard, tracked_detections,
                distances=metric_distances,
                class_names=tracked_class_names,
                draw_distance=True,
                is_metric=True
            )
        else:
            frame_standard = self.visualizer.draw_detections(
                frame_standard, detections,
                distances=metric_distances,
                draw_distance=True,
                show_confidence=False,
                is_metric=True
            )

        # Add info panel to standard output
        calibration_status = "Calibrated" if bonnet_detected else "Not calibrated"
        tracking_status = f"{len(tracked_detections)}" if tracked_detections is not None else "Disabled"

        info_standard = {
            'Detections': len(detections),
            'Calibration': calibration_status,
            'Tracking': tracking_status
        }
        frame_standard = self.visualizer.draw_info_panel(
            frame_standard, info_standard, position='top-right'
        )

        # Create depth visualization output (depth heatmap + bboxes + metric distances)
        # Use depth heatmap as the base (not an overlay)
        frame_depth = self.visualizer.draw_depth_map(depth_map)

        # Resize to match frame dimensions if needed
        if frame_depth.shape[:2] != frame.shape[:2]:
            frame_depth = cv2.resize(frame_depth, (frame.shape[1], frame.shape[0]))

        # Draw tracked detections if available, otherwise draw regular detections
        if tracked_detections is not None and len(tracked_detections) > 0:
            frame_depth = self.visualizer.draw_tracked_detections(
                frame_depth, tracked_detections,
                distances=metric_distances,
                class_names=tracked_class_names,
                draw_distance=True,
                is_metric=True
            )
        else:
            frame_depth = self.visualizer.draw_detections(
                frame_depth, detections,
                distances=metric_distances,
                draw_distance=True,
                show_confidence=False,
                is_metric=True
            )

        # Add info panel to depth output
        info_depth = {
            'Detections': len(detections),
            'Calibration': calibration_status,
            'Tracking': tracking_status
        }
        frame_depth = self.visualizer.draw_info_panel(
            frame_depth, info_depth, position='top-right'
        )

        return frame_standard, frame_depth


    def _process_frame(self, frame):
        # This method processes single frame (legacy method for backward compatibility).
        # Detect objects
        detections = self.detector.detect(frame)

        # Estimate depth
        depth_map = self.depth_estimator.estimate_depth(frame)

        # Calculate distances
        distances = []
        for det in detections:
            distance = self.distance_calculator.calculate_distance(depth_map, det)
            distances.append(distance)

        # Visualization
        vis_config = self.config['visualization']

        # Draw detections
        if vis_config['draw_bbox']:
            frame = self.visualizer.draw_detections(
                frame, detections,
                distances=distances if vis_config['draw_distance'] else None,
                draw_distance=vis_config['draw_distance']
            )

        # Draw depth map overlay
        if vis_config['draw_depth_map']:
            frame = self.visualizer.overlay_depth_map(frame, depth_map, alpha=0.2)

        # Draw info panel
        info = {
            'Detections': len(detections),
            'FPS': 30  # Placeholder
        }
        frame = self.visualizer.draw_info_panel(frame, info, position='top-right')

        return frame

    def process_frame(self, frame):
        # This method processes single frame (for interactive use).
        # Detect objects
        detections = self.detector.detect(frame)

        # Estimate depth
        depth_map = self.depth_estimator.estimate_depth(frame)

        # Calculate distances
        distances = []
        for det in detections:
            distance = self.distance_calculator.calculate_distance(depth_map, det)
            distances.append(distance)

        # Visualization
        vis_config = self.config['visualization']

        # Draw detections
        frame_with_detections = frame.copy()
        if vis_config['draw_bbox']:
            frame_with_detections = self.visualizer.draw_detections(
                frame_with_detections, detections,
                distances=distances if vis_config['draw_distance'] else None,
                draw_distance=vis_config['draw_distance']
            )

        # Draw depth map overlay
        frame_with_depth = frame.copy()
        if vis_config['draw_depth_map']:
            frame_with_depth = self.visualizer.overlay_depth_map(frame_with_depth, depth_map, alpha=0.3)

        return {
            'detections': detections,
            'distances': distances,
            'depth_map': depth_map,
            'frame_with_detections': frame_with_detections,
            'frame_with_depth': frame_with_depth,
            'depth_colored': self.visualizer.draw_depth_map(depth_map)
        }

    def get_config(self):
        # This method gets current configuration.
        return self.config

