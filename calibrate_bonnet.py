#!/usr/bin/env python3
# Author: Mridankan Mandal.

# Interactive bonnet calibration tool for selecting the reference point (1.35m from camera)
import cv2
import yaml
import numpy as np
from pathlib import Path
import argparse
import sys
import os
from models import DepthEstimator, YOLODetector
from utils import setup_logger


def _setup_qt_environment():
    # Setup Qt environment variables before importing PySide6 GUI components
    try:
        import PySide6
        pyside6_path = os.path.dirname(PySide6.__file__)
        plugins_path = os.path.join(pyside6_path, 'plugins')

        if os.path.exists(plugins_path):
            os.environ['QT_PLUGIN_PATH'] = plugins_path
            # Also set QT_QPA_PLATFORM_PLUGIN_PATH for additional compatibility
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(plugins_path, 'platforms')
    except Exception as e:
        print(f"Warning: Could not setup Qt environment: {e}.")


# Setup Qt environment before any GUI imports
_setup_qt_environment()

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QLineEdit, QSpinBox, QMessageBox)
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen, QBrush
from PySide6.QtCore import Qt, QThread, Signal, QSize


class DepthEstimationThread(QThread):
    # Background thread for depth estimation to keep GUI responsive
    finished = Signal(np.ndarray)
    error = Signal(str)

    def __init__(self, depth_estimator, frame):
        super().__init__()
        self.depth_estimator = depth_estimator
        self.frame = frame

    def run(self):
        try:
            depth_map = self.depth_estimator.estimate_depth(self.frame)
            self.finished.emit(depth_map)
        except Exception as e:
            self.error.emit(str(e))


class CalibrationImageWidget(QWidget):
    # Custom widget for displaying depth heatmap and handling mouse clicks
    point_selected = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.pixmap = None
        self.selected_point = None
        self.frame_width = 0
        self.frame_height = 0

    def set_image(self, pixmap):
        self.pixmap = pixmap
        self.update()

    def set_frame_dimensions(self, width, height):
        self.frame_width = width
        self.frame_height = height

    def set_selected_point(self, point):
        self.selected_point = point
        self.update()

    def mousePressEvent(self, event):
        if self.pixmap is None:
            return

        # Convert widget coordinates to image coordinates
        x = int(event.pos().x() * self.frame_width / self.width())
        y = int(event.pos().y() * self.frame_height / self.height())

        # Validate coordinates
        if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
            self.selected_point = (x, y)
            self.point_selected.emit(x, y)
            self.update()

    def paintEvent(self, event):
        if self.pixmap is None:
            return

        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.width(), self.height(), self.pixmap)

        # Draw marker at selected point
        if self.selected_point is not None:
            x, y = self.selected_point
            # Scale coordinates to widget size
            widget_x = int(x * self.width() / self.frame_width)
            widget_y = int(y * self.height() / self.frame_height)

            # Draw black dot with white outline
            painter.setBrush(QBrush(QColor(0, 0, 0)))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawEllipse(widget_x - 8, widget_y - 8, 16, 16)

            # Draw crosshair
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawLine(widget_x - 25, widget_y, widget_x + 25, widget_y)
            painter.drawLine(widget_x, widget_y - 25, widget_x, widget_y + 25)

    def sizeHint(self):
        if self.pixmap:
            return QSize(self.pixmap.width(), self.pixmap.height())
        return QSize(1920, 1080)


class BonnetCalibrationTool(QMainWindow):
    # Interactive tool to select bonnet reference point by clicking on video frame

    def __init__(self, video_path, config_path, output_config_path=None, frame_number=10, reference_distance=1.35):
        super().__init__()
        # Initialize calibration tool with video and config
        self.video_path = Path(video_path)
        self.config_path = Path(config_path)
        self.output_config_path = Path(output_config_path) if output_config_path else self.config_path
        self.frame_number = max(1, frame_number)  # Ensure frame_number >= 1
        self.reference_distance = reference_distance

        self.logger = setup_logger('BonnetCalibration', level='INFO')

        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize depth estimator
        depth_config = self.config['depth']
        self.depth_estimator = DepthEstimator(
            model_type=depth_config['model_type'],
            device=depth_config['device'],
            input_size=depth_config.get('input_size', 518),
            calibration_config=depth_config.get('calibration', {})
        )

        # Initialize YOLO detector for bonnet detection
        detector_config = self.config.get('detector', {})
        yolo_model_path = self.config.get('paths', {}).get('yolo_model', detector_config.get('model_path', 'data/models/IDDDYOLO11m.pt'))
        self.detector = YOLODetector(
            model_path=yolo_model_path,
            device=detector_config.get('device', 'cuda'),
            confidence_threshold=detector_config.get('confidence_threshold', 0.25)
        )

        self.selected_point = None
        self.frame = None
        self.depth_map = None
        self.depth_normalized = None
        self.editing_distance = False
        self.distance_text = str(reference_distance)
        self.use_auto_bonnet = False  # Flag for automatic bonnet detection mode
        self.calibration_complete = False
        self.estimation_thread = None

        # Setup UI
        self.setup_ui()
        self.setWindowTitle("Bonnet Calibration Tool")
        self.setGeometry(100, 100, 1400, 900)

    def setup_ui(self):
        # Setup the user interface
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()

        # Left side: Image display
        left_layout = QVBoxLayout()
        self.image_widget = CalibrationImageWidget()
        self.image_widget.point_selected.connect(self.on_point_selected)
        left_layout.addWidget(self.image_widget)

        # Right side: Controls
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)

        # Title
        title_label = QLabel("Bonnet Calibration")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        right_layout.addWidget(title_label)

        # Instructions
        instructions = QLabel(
            "1. Click on the bonnet center in the depth heatmap\n"
            "2. Adjust reference distance if needed\n"
            "3. Click Confirm to save calibration\n\n"
            "Depth Heatmap: Red=Close | Blue=Far"
        )
        instructions.setWordWrap(True)
        right_layout.addWidget(instructions)

        # Reference distance
        dist_layout = QHBoxLayout()
        dist_layout.addWidget(QLabel("Reference Distance (m):"))
        self.distance_input = QLineEdit(str(self.reference_distance))
        self.distance_input.setMaximumWidth(100)
        dist_layout.addWidget(self.distance_input)
        right_layout.addLayout(dist_layout)

        # Selected point info
        self.point_info_label = QLabel("No point selected")
        right_layout.addWidget(self.point_info_label)

        # Depth value info
        self.depth_info_label = QLabel("Depth: N/A")
        right_layout.addWidget(self.depth_info_label)

        # Method info
        self.method_label = QLabel("Method: None")
        right_layout.addWidget(self.method_label)

        # Status label
        self.status_label = QLabel("Loading frame and estimating depth...")
        self.status_label.setStyleSheet("color: blue;")
        right_layout.addWidget(self.status_label)

        # Buttons
        button_layout = QVBoxLayout()

        self.auto_bonnet_btn = QPushButton("Auto Bonnet Detection (B)")
        self.auto_bonnet_btn.clicked.connect(self.on_auto_bonnet)
        button_layout.addWidget(self.auto_bonnet_btn)

        self.clear_btn = QPushButton("Clear Selection (D)")
        self.clear_btn.clicked.connect(self.on_clear)
        button_layout.addWidget(self.clear_btn)

        self.confirm_btn = QPushButton("Confirm Calibration (C)")
        self.confirm_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.confirm_btn.clicked.connect(self.on_confirm)
        button_layout.addWidget(self.confirm_btn)

        self.cancel_btn = QPushButton("Cancel (ESC)")
        self.cancel_btn.setStyleSheet("background-color: #f44336; color: white;")
        self.cancel_btn.clicked.connect(self.on_cancel)
        button_layout.addWidget(self.cancel_btn)

        right_layout.addLayout(button_layout)
        right_layout.addStretch()

        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 1)

        central_widget.setLayout(main_layout)

        # Load frame and estimate depth
        self.load_frame_and_estimate_depth()

    def load_frame_and_estimate_depth(self):
        # Load frame from video and estimate depth
        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {self.video_path}")
            QMessageBox.critical(self, "Error", f"Failed to open video: {self.video_path}")
            self.close()
            return

        # Read specified frame (frame_number is 1-indexed)
        frame_idx = 0
        ret = True
        frame = None
        while frame_idx < self.frame_number and ret:
            ret, frame = cap.read()
            frame_idx += 1

        cap.release()

        if not ret or frame is None:
            self.logger.error(f"Failed to read frame {self.frame_number}")
            QMessageBox.critical(self, "Error", f"Failed to read frame {self.frame_number}")
            self.close()
            return

        self.logger.info(f"Loaded frame {self.frame_number} for calibration")
        self.frame = frame
        h, w = frame.shape[:2]
        self.image_widget.set_frame_dimensions(w, h)

        # Start depth estimation in background thread
        self.estimation_thread = DepthEstimationThread(self.depth_estimator, frame)
        self.estimation_thread.finished.connect(self.on_depth_estimation_finished)
        self.estimation_thread.error.connect(self.on_depth_estimation_error)
        self.estimation_thread.start()

    def on_depth_estimation_finished(self, depth_map):
        # Handle depth estimation completion
        self.depth_map = depth_map
        self.logger.info("Depth estimation completed")

        # Normalize depth for visualization (0-1 range)
        depth_min = np.nanmin(self.depth_map)
        depth_max = np.nanmax(self.depth_map)
        if depth_max > depth_min:
            self.depth_normalized = (self.depth_map - depth_min) / (depth_max - depth_min)
        else:
            self.depth_normalized = np.ones_like(self.depth_map) * 0.5

        # Create depth heatmap visualization (TURBO colormap: red=close, blue=far)
        depth_heatmap = cv2.applyColorMap(
            (self.depth_normalized * 255).astype(np.uint8),
            cv2.COLORMAP_TURBO
        )

        # Convert BGR to RGB for Qt
        depth_heatmap_rgb = cv2.cvtColor(depth_heatmap, cv2.COLOR_BGR2RGB)

        # Convert to QPixmap
        h, w, ch = depth_heatmap_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(depth_heatmap_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        self.image_widget.set_image(pixmap)
        self.status_label.setText("Ready. Click on bonnet center to select reference point.")
        self.status_label.setStyleSheet("color: green;")

    def on_depth_estimation_error(self, error_msg):
        # Handle depth estimation error
        self.logger.error(f"Depth estimation error: {error_msg}")
        QMessageBox.critical(self, "Error", f"Depth estimation failed: {error_msg}")
        self.close()

    def on_point_selected(self, x, y):
        # Handle point selection from image widget
        self.selected_point = (x, y)
        self.use_auto_bonnet = False
        self.logger.info(f"Selected point: ({x}, {y})")

        # Update info labels
        self.point_info_label.setText(f"Selected Point: ({x}, {y})")
        if self.depth_map is not None:
            depth_val = self.depth_map[int(np.clip(y, 0, self.depth_map.shape[0]-1)),
                                       int(np.clip(x, 0, self.depth_map.shape[1]-1))]
            self.depth_info_label.setText(f"Depth: {depth_val:.4f}")
        self.method_label.setText("Method: Manual Point Selection")
        self.image_widget.set_selected_point((x, y))

    def on_auto_bonnet(self):
        # Handle auto bonnet detection
        self._detect_bonnet_auto()
        self.logger.info("Auto bonnet detection applied")
        self.method_label.setText("Method: Automatic Bonnet Detection")
        if self.depth_map is not None:
            x, y = self.selected_point
            depth_val = self.depth_map[int(np.clip(y, 0, self.depth_map.shape[0]-1)),
                                       int(np.clip(x, 0, self.depth_map.shape[1]-1))]
            self.depth_info_label.setText(f"Depth: {depth_val:.4f}")
        self.image_widget.set_selected_point(self.selected_point)

    def on_clear(self):
        # Handle clear selection
        self.selected_point = None
        self.use_auto_bonnet = False
        self.point_info_label.setText("No point selected")
        self.depth_info_label.setText("Depth: N/A")
        self.method_label.setText("Method: None")
        self.image_widget.set_selected_point(None)
        self.logger.info("Calibration point cleared")

    def on_confirm(self):
        # Handle confirm calibration
        if self.selected_point is None and not self.use_auto_bonnet:
            QMessageBox.warning(self, "Warning", "No point selected. Please click on the bonnet center or use auto detection.")
            return

        # Update reference distance from input
        try:
            self.reference_distance = float(self.distance_input.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid reference distance value.")
            return

        # Save calibration
        if self._save_calibration():
            self.logger.info("Calibration saved successfully")
            self.calibration_complete = True
            QMessageBox.information(self, "Success", "Calibration saved successfully!")
            self.close()
        else:
            QMessageBox.critical(self, "Error", "Failed to save calibration")

    def on_cancel(self):
        # Handle cancel
        self.logger.info("Calibration cancelled")
        self.close()

    def keyPressEvent(self, event):
        # Handle keyboard shortcuts
        if event.key() == Qt.Key_Escape:
            self.on_cancel()
        elif event.key() == Qt.Key_C:
            self.on_confirm()
        elif event.key() == Qt.Key_D:
            self.on_clear()
        elif event.key() == Qt.Key_B:
            self.on_auto_bonnet()
        elif event.key() == Qt.Key_R:
            # Focus on distance input for editing
            self.distance_input.selectAll()
            self.distance_input.setFocus()
        else:
            super().keyPressEvent(event)
    def _detect_bonnet_auto(self):
        # Automatic bonnet detection using YOLO to find bonnet bounding box center
        h, w = self.frame.shape[:2]

        # Detect objects in the frame
        detections = self.detector.detect(self.frame)

        # Filter for vehicle-related classes in the bottom 30% of frame (typical bonnet region)
        vehicle_classes = ['car', 'vehicle fallback', 'truck', 'bus']
        bonnet_region_top = int(h * 0.7)  # Bottom 30% of frame

        candidate_detections = []
        for det in detections:
            class_name = det.get('class_name', '').lower()
            y1 = det.get('y1', 0)

            # Check if it's a vehicle class
            if any(vc in class_name for vc in vehicle_classes):
                # Check if detection is in bottom portion of frame (likely bonnet)
                if y1 >= bonnet_region_top:
                    candidate_detections.append(det)

        if candidate_detections:
            # Select the detection closest to bottom-center (most likely the vehicle's own bonnet)
            best_det = None
            best_distance = float('inf')

            for det in candidate_detections:
                x1 = det.get('x1', 0)
                y1 = det.get('y1', 0)
                x2 = det.get('x2', 0)
                y2 = det.get('y2', 0)

                # Calculate bbox center
                bbox_center_x = (x1 + x2) / 2
                bbox_center_y = (y1 + y2) / 2

                # Distance from frame bottom-center
                frame_bottom_center_x = w / 2
                frame_bottom_center_y = h

                distance = np.sqrt(
                    (bbox_center_x - frame_bottom_center_x) ** 2 +
                    (bbox_center_y - frame_bottom_center_y) ** 2
                )

                if distance < best_distance:
                    best_distance = distance
                    best_det = det

            if best_det is not None:
                x1 = best_det.get('x1', 0)
                y1 = best_det.get('y1', 0)
                x2 = best_det.get('x2', 0)
                y2 = best_det.get('y2', 0)
                x = int((x1 + x2) / 2)
                y = int((y1 + y2) / 2)

                self.selected_point = (x, y)
                self.use_auto_bonnet = True
                self.point_info_label.setText(f"Auto Bonnet (YOLO): ({x}, {y})")
                self.logger.info(f"Bonnet detected by YOLO at bbox center: ({x}, {y})")
                return

        # Fallback to fixed-point approach if no bonnet detected
        x = w // 2
        y = int(h * 0.88)
        self.selected_point = (x, y)
        self.use_auto_bonnet = True
        self.point_info_label.setText(f"Auto Bonnet (Fallback): ({x}, {y})")
        self.logger.warning(f"No bonnet detected by YOLO, using fixed-point fallback: ({x}, {y})")

    def _save_calibration(self):
        # Save bonnet calibration (depth value and reference distance) to config
        if self.selected_point is None:
            return False

        x, y = self.selected_point
        h, w = self.frame.shape[:2]

        # Validate coordinates are within bounds
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))

        # Get depth value at selected point with bounds checking
        bonnet_depth = float(self.depth_map[y, x])

        # Update config with calibration data
        if 'calibration' not in self.config:
            self.config['calibration'] = {}

        # Save the bonnet depth reference (from calibration frame) and reference distance
        # These are used to calculate the scale factor: scale_factor = reference_distance / bonnet_depth
        self.config['calibration']['bonnet_depth_reference'] = bonnet_depth
        self.config['calibration']['reference_distance'] = float(self.reference_distance)

        # Record which method was used
        method = "automatic_bonnet_detection" if self.use_auto_bonnet else "manual_point_selection"
        self.config['calibration']['calibration_method'] = method

        # Save to file
        try:
            with open(self.output_config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)

            self.logger.info(f"Calibration saved to {self.output_config_path}")
            self.logger.info(f"  Method: {method}")
            self.logger.info(f"  Bonnet depth (from calibration frame): {bonnet_depth:.4f}")
            self.logger.info(f"  Reference distance: {self.reference_distance}m")
            self.logger.info(f"  Scale factor: {self.reference_distance / bonnet_depth:.6f}")

            return True
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            return False

    def run(self):
        # Run the calibration GUI (for backward compatibility with main.py)
        # This method blocks until the window is closed
        self.show()
        # Get the application instance
        app = QApplication.instance()
        if app is None:
            # If no app exists, create one (for standalone usage)
            app = QApplication(sys.argv)
        # Run the event loop
        app.exec()
        return self.calibration_complete


def main():
    parser = argparse.ArgumentParser(description="Interactive bonnet calibration tool")
    parser.add_argument('-i', '--input', required=True, help='Input video path')
    parser.add_argument('-c', '--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('-o', '--output', help='Output config file path (default: same as input)')
    parser.add_argument('-f', '--frame', type=int, default=10, help='Frame number to use for calibration (default: 10)')
    parser.add_argument('-d', '--distance', type=float, default=1.35, help='Reference distance in meters (default: 1.35)')

    args = parser.parse_args()

    # Create Qt application
    app = QApplication(sys.argv)

    # Create and show calibration tool
    tool = BonnetCalibrationTool(args.input, args.config, args.output, frame_number=args.frame, reference_distance=args.distance)
    tool.show()

    # Run the application
    exit_code = app.exec()

    # Return success if calibration was completed
    return 0 if tool.calibration_complete else 1


if __name__ == '__main__':
    exit(main())

