from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
RTSP = 'RTSP'


SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, RTSP]


# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'office_4.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'office_4_detected.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEO_1_PATH = VIDEO_DIR / 'video_1.mp4'
VIDEO_2_PATH = VIDEO_DIR / 'video_2.mp4'
VIDEO_3_PATH = VIDEO_DIR / 'video_3.mp4'
VIDEO_4_PATH = VIDEO_DIR / 'DCube_Office_Video.mp4'
VIDEO_5_PATH = VIDEO_DIR / 'Dr_Haider_Garage_Video.mp4'
VIDEO_6_PATH = VIDEO_DIR / 'Sample_Warehouse_Video.mp4'
VIDEOS_DICT = {
    'Test Video 1': VIDEO_1_PATH,
    'Test Video 2': VIDEO_2_PATH,
    'Test Video 3': VIDEO_3_PATH,
    'DCube Office': VIDEO_4_PATH,
    'Dr. Haider Garage': VIDEO_5_PATH,
    'Warehouse Footage': VIDEO_6_PATH,
}

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolov8n.pt'
SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'

# Webcam
WEBCAM_PATH = 0
