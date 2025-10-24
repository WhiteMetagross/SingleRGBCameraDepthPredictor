# This program provides video processing utilities for reading and writing video files.
import cv2
import numpy as np
from pathlib import Path


class VideoProcessor:
    # This class handles video input operations including frame reading and property extraction.

    def __init__(self, video_path):
        # This method initializes the video processor by opening the video file and extracting its properties.
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = 0
    
    def read_frame(self):
        # This method reads the next frame from the video file and returns it with a success flag.
        success, frame = self.cap.read()
        if success:
            self.frame_count += 1
        return success, frame
    
    def get_frame_at(self, frame_number):
        # This method retrieves a frame at a specific position in the video file.
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = self.cap.read()
        return frame if success else None

    def reset(self):
        # This method resets the video to the beginning.
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_count = 0

    def close(self):
        # This method closes the video file.
        self.cap.release()

    def get_info(self):
        # This method returns video information including fps, dimensions, and total frames.
        return {
            'path': str(self.video_path),
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'total_frames': self.total_frames,
            'duration_seconds': self.total_frames / self.fps if self.fps > 0 else 0
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class VideoWriter:
    # This class handles video output operations for writing processed frames to video files.

    def __init__(self, output_path, fps, width, height, codec='mp4v'):
        # This method initializes the video writer with output path, frame rate, dimensions, and codec.
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps,
            (width, height)
        )
        
        if not self.writer.isOpened():
            raise ValueError(f"Cannot create video writer: {self.output_path}")
        
        self.frame_count = 0
    
    def write_frame(self, frame):
        # This method writes a frame to the output video file.
        self.writer.write(frame)
        self.frame_count += 1

    def close(self):
        # This method closes the video writer and releases resources.
        self.writer.release()

    def get_info(self):
        # This method returns information about the written video including path and frame count.
        return {
            'path': str(self.output_path),
            'frames_written': self.frame_count
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

