# This program provides visualization utilities for drawing detections, depth maps, and tracking information on video frames.
import cv2
import numpy as np
import torch


class Visualizer:
    # This class visualizes detections, depth information, and tracking data on video frames.

    def __init__(self, font_scale=0.7, thickness=2, colormap='turbo'):
        # This method initializes the visualizer with font settings and colormap configuration.
        self.font_scale = font_scale
        self.thickness = thickness
        self.colormap = self._get_colormap(colormap)
        
        # Color palette for different classes
        self.colors = {
            'vehicle': (0, 255, 0),      # Green
            'car': (0, 255, 0),
            'truck': (0, 165, 255),      # Orange
            'bus': (255, 0, 0),          # Blue
            'motorcycle': (255, 255, 0), # Cyan
            'default': (0, 255, 0)
        }
    
    def _get_colormap(self, colormap_name):
        # This method retrieves the OpenCV colormap by name for depth visualization.
        colormaps = {}

        # Add available colormaps
        if hasattr(cv2, 'COLORMAP_VIRIDIS'):
            colormaps['viridis'] = cv2.COLORMAP_VIRIDIS
        if hasattr(cv2, 'COLORMAP_TURBO'):
            colormaps['turbo'] = cv2.COLORMAP_TURBO
        if hasattr(cv2, 'COLORMAP_JET'):
            colormaps['jet'] = cv2.COLORMAP_JET
        if hasattr(cv2, 'COLORMAP_HOT'):
            colormaps['hot'] = cv2.COLORMAP_HOT
        if hasattr(cv2, 'COLORMAP_COOL'):
            colormaps['cool'] = cv2.COLORMAP_COOL
        if hasattr(cv2, 'COLORMAP_SPRING'):
            colormaps['spring'] = cv2.COLORMAP_SPRING
        if hasattr(cv2, 'COLORMAP_SUMMER'):
            colormaps['summer'] = cv2.COLORMAP_SUMMER
        if hasattr(cv2, 'COLORMAP_AUTUMN'):
            colormaps['autumn'] = cv2.COLORMAP_AUTUMN
        if hasattr(cv2, 'COLORMAP_WINTER'):
            colormaps['winter'] = cv2.COLORMAP_WINTER
        if hasattr(cv2, 'COLORMAP_BONE'):
            colormaps['bone'] = cv2.COLORMAP_BONE
        if hasattr(cv2, 'COLORMAP_PINK'):
            colormaps['pink'] = cv2.COLORMAP_PINK
        if hasattr(cv2, 'COLORMAP_HSV'):
            colormaps['hsv'] = cv2.COLORMAP_HSV

        # Default to first available or JET
        default = colormaps.get('turbo', colormaps.get('jet', cv2.COLORMAP_JET))
        return colormaps.get(colormap_name, default)
    
    def draw_detections(self, frame, detections, distances=None, draw_distance=True, show_confidence=False, is_metric=False):
        """
        Draw bounding boxes and labels on frame

        Args:
            frame: Input frame
            detections: List of detections
            distances: List of distances (optional)
            draw_distance: Whether to draw distance labels
            show_confidence: Whether to show confidence scores (default: False)
            is_metric: Whether distances are in meters (default: False)

        Returns:
            frame: Frame with drawn detections
        """
        frame = frame.copy()

        # Use 50% smaller font for labels
        label_font_scale = self.font_scale * 0.5

        for i, det in enumerate(detections):
            x1 = int(det['x1'])
            y1 = int(det['y1'])
            x2 = int(det['x2'])
            y2 = int(det['y2'])
            class_name = det['class_name']

            # Get color
            color = self.colors.get(class_name.lower(), self.colors['default'])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

            # Prepare label - class name only
            label = f"{class_name}"

            # Add distance if available
            if draw_distance and (distances is not None) and i < len(distances):
                distance = distances[i]
                # Guard against None and NaN
                if distance is not None and not (isinstance(distance, float) and np.isnan(distance)):
                    if is_metric:
                        # Display metric distance in meters
                        label += f", {float(distance):.1f}m"
                    else:
                        # Display relative depth (0-1)
                        label += f", D:{float(distance):.2f}"

            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                        label_font_scale, self.thickness)[0]
            label_y = max(y1 - 5, label_size[1] + 5)
            cv2.rectangle(frame, (x1, label_y - label_size[1] - 5),
                         (x1 + label_size[0] + 5, label_y + 5), color, -1)

            # Draw label text
            cv2.putText(frame, label, (x1 + 2, label_y - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (255, 255, 255),
                       self.thickness)

        return frame
    
    def draw_depth_map(self, depth_map, alpha=0.5):
        """
        Convert depth map to colored visualization
        
        Args:
            depth_map: Depth map (float32, 0-1 or 0-255)
            alpha: Transparency (0-1)
            
        Returns:
            colored_depth: Colored depth map (BGR format)
        """
        # Normalize to 0-255
        if depth_map.max() <= 1.0:
            depth_normalized = (depth_map * 255).astype(np.uint8)
        else:
            depth_normalized = np.clip(depth_map, 0, 255).astype(np.uint8)
        
        # Apply colormap
        colored_depth = cv2.applyColorMap(depth_normalized, self.colormap)
        
        return colored_depth
    
    def overlay_depth_map(self, frame, depth_map, alpha=0.3):
        """
        Overlay depth map on frame
        
        Args:
            frame: Input frame
            depth_map: Depth map
            alpha: Transparency (0-1)
            
        Returns:
            overlaid_frame: Frame with overlaid depth map
        """
        colored_depth = self.draw_depth_map(depth_map)
        
        # Resize depth map to match frame if needed
        if colored_depth.shape[:2] != frame.shape[:2]:
            colored_depth = cv2.resize(colored_depth, 
                                      (frame.shape[1], frame.shape[0]))
        
        # Blend
        overlaid = cv2.addWeighted(frame, 1 - alpha, colored_depth, alpha, 0)
        
        return overlaid
    
    def draw_info_panel(self, frame, info_dict, position='top-left'):
        """
        Draw information panel on frame
        
        Args:
            frame: Input frame
            info_dict: Dictionary with information to display
            position: Position ('top-left', 'top-right', 'bottom-left', 'bottom-right')
            
        Returns:
            frame: Frame with info panel
        """
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Prepare text
        lines = []
        for key, value in info_dict.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.2f}")
            else:
                lines.append(f"{key}: {value}")
        
        # Calculate panel size
        line_height = 25
        panel_height = len(lines) * line_height + 10
        panel_width = max([cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.6, 1)[0][0] for line in lines]) + 20
        
        # Determine position
        if position == 'top-left':
            x, y = 10, 10
        elif position == 'top-right':
            x, y = w - panel_width - 10, 10
        elif position == 'bottom-left':
            x, y = 10, h - panel_height - 10
        else:  # bottom-right
            x, y = w - panel_width - 10, h - panel_height - 10
        
        # Draw panel background
        cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height),
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height),
                     (255, 255, 255), 2)
        
        # Draw text
        for i, line in enumerate(lines):
            text_y = y + 20 + i * line_height
            cv2.putText(frame, line, (x + 10, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def draw_tracked_detections(self, frame, tracked_detections, distances=None, class_names=None, draw_distance=True, is_metric=False):
        # This method draws tracked bounding boxes with class labels and distances using track-specific colors.
        frame = frame.copy()
        label_font_scale = self.font_scale * 0.5

        for i, track in enumerate(tracked_detections):
            if len(track) >= 5:
                x1, y1, x2, y2, track_id = track[:5]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                track_id = int(track_id)

                # Get color based on track ID (darker shade for better visibility)
                base_color = self._get_track_color(track_id)
                # Darken the color by reducing each channel by 30%
                color = tuple(int(c * 0.7) for c in base_color)

                # Draw bounding box with darker color
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

                # Prepare label with class name and distance only (no track ID)
                label = "Vehicle"
                if class_names and i < len(class_names) and class_names[i]:
                    label = class_names[i]

                # Add distance if available
                if draw_distance and (distances is not None) and i < len(distances):
                    distance = distances[i]
                    if distance is not None and not (isinstance(distance, float) and np.isnan(distance)):
                        if is_metric:
                            label += f", DL: {float(distance):.1f}m"
                        else:
                            label += f", DL: {float(distance):.2f}"

                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                            label_font_scale, self.thickness)[0]
                label_y = max(y1 - 5, label_size[1] + 5)
                cv2.rectangle(frame, (x1, label_y - label_size[1] - 5),
                             (x1 + label_size[0] + 5, label_y + 5), color, -1)

                # Draw label text
                cv2.putText(frame, label, (x1 + 2, label_y - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (255, 255, 255),
                           self.thickness)

        return frame

    @staticmethod
    def _get_track_color(track_id):
        # This method returns a consistent color for each track ID from a diverse palette.
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
            (128, 0, 255), (0, 128, 255), (255, 255, 128), (255, 128, 255), (128, 255, 255),
            (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0), (192, 0, 192),
            (0, 192, 192), (128, 128, 0), (128, 0, 128), (0, 128, 128), (64, 64, 64),
            (255, 64, 64), (64, 255, 64), (64, 64, 255), (255, 192, 0), (192, 255, 0),
            (0, 255, 192), (192, 0, 255), (255, 0, 192), (0, 192, 255)
        ]
        return colors[int(track_id) % len(colors)]

    def create_side_by_side(self, frame1, frame2, label1="Frame 1", label2="Frame 2"):
        """
        Create side-by-side visualization

        Args:
            frame1: First frame
            frame2: Second frame
            label1: Label for first frame
            label2: Label for second frame

        Returns:
            combined: Side-by-side frames
        """
        # Resize to same height
        h = max(frame1.shape[0], frame2.shape[0])
        frame1_resized = cv2.resize(frame1, (int(frame1.shape[1] * h / frame1.shape[0]), h))
        frame2_resized = cv2.resize(frame2, (int(frame2.shape[1] * h / frame2.shape[0]), h))

        # Add labels
        cv2.putText(frame1_resized, label1, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame2_resized, label2, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Concatenate
        combined = np.hstack([frame1_resized, frame2_resized])

        return combined

