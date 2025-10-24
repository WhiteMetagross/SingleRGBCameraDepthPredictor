# This program is the main entry point for the YOLO + DepthAnything V2 pipeline that generates two output videos: standard detection output with bounding boxes and distances, and depth visualization output with depth heatmap, bounding boxes, and distances.
# Author: Mridankan Mandal.

import argparse
import sys
import warnings
import os
from pathlib import Path


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

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*xFormers.*')
warnings.filterwarnings('ignore', message='.*triton.*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings if any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipeline import Pipeline
from calibrate_bonnet import BonnetCalibrationTool

# Import PySide6 for GUI
from PySide6.QtWidgets import QApplication


def main():
    # This function parses command-line arguments and orchestrates the video processing pipeline.
    parser = argparse.ArgumentParser(
        description='YOLO + DepthAnything V2: Precise Object Distances Without LiDAR'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to input video file (required)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='outputs',
        help='Directory to save output videos (default: outputs)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to process (optional, default: process entire video)'
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='output',
        help='Prefix for output filenames (default: output)'
    )
    parser.add_argument(
        '--skip-calibration',
        action='store_true',
        help='Skip the interactive calibration step (use existing calibration)'
    )
    parser.add_argument(
        '--calibration-frame',
        type=int,
        default=10,
        help='Frame number to use for calibration (default: 10)'
    )
    parser.add_argument(
        '--reference-distance',
        type=float,
        default=1.35,
        help='Reference distance in meters for bonnet calibration (default: 1.35)'
    )

    args = parser.parse_args()

    # Validate input video exists
    input_video = Path(args.input)
    if not input_video.exists():
        print(f"Error: Input video not found: {input_video}.")
        sys.exit(1)

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}.")
        sys.exit(1)

    # Initialize pipeline
    print("YOLO + DepthAnything V2 started.")
    print("Generating two output videos.")
    print("1. Standard detection output (bounding boxes and distances).")
    print("2. Depth visualization output (depth heatmap, bounding boxes, and distances).")

    # Run calibration if not skipped
    if not args.skip_calibration:
        print("Bonnet calibration.")
        print(f"Opening calibration tool for frame {args.calibration_frame}.")
        print(f"Reference distance: {args.reference_distance} m.")
        print("Calibration window controls:")
        print(" - Click: Select bonnet reference point.")
        print(" - R: Edit reference distance (type value, then press Enter).")
        print(" - B: Use automatic bonnet detection.")
        print(" - C or Enter: Confirm calibration and start processing.")
        print(" - D or Delete: Clear selection.")
        print(" - ESC: Cancel and exit.")
        print("Depth heatmap colors:")
        print(" - Red/Warm: Close objects (larger depth values).")
        print(" - Blue/Cool: Far objects (smaller depth values).")

        # Create QApplication for GUI (if not already created)
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        calibrator = BonnetCalibrationTool(
            str(input_video),
            str(config_path),
            str(config_path),
            frame_number=args.calibration_frame,
            reference_distance=args.reference_distance
        )

        if not calibrator.run():
            print("\nCalibration cancelled or failed. Exiting.")
            sys.exit(1)

        print("Calibration completed successfully.")

    pipeline = Pipeline(config_path)

    # Override input video
    pipeline.config['paths']['input_video'] = str(input_video)

    # Override output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.config['paths']['output_dir'] = str(output_dir)

    # Override max_frames if provided
    if args.max_frames:
        pipeline.config['video']['max_frames'] = args.max_frames

    # Generate output filenames with prefix
    output_standard = output_dir / f"{args.output_prefix}_standard.mp4"
    output_depth = output_dir / f"{args.output_prefix}_depth_visualization.mp4"

    # Process video
    try:
        print(f"\nInput video: {input_video}")
        print(f"Output directory: {output_dir}")
        print(f"Output prefix: {args.output_prefix}")
        if args.max_frames:
            print(f"Max frames: {args.max_frames}")
        print()

        output_standard, output_depth = pipeline.process_video(
            output_standard=str(output_standard),
            output_depth=str(output_depth)
        )
        print("Processing completed successfully.")
        print("Standard detection output:")
        print(f"  {output_standard}.")
        print("Depth visualization output:")
        print(f"  {output_depth}.")
    except Exception as e:
        print(f"\nError during processing: {e}.")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

