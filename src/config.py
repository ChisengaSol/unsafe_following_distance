import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, 'weights', 'best.pt')
VIDEO_PATH = os.path.join(PROJECT_ROOT, 'data', 'videos', 'test_video_00.mp4')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'videos', 'output_ttc.mp4')

PIXELS_PER_METER_Y = 30  
PIXELS_PER_METER_X = 20  
TTC_THRESHOLD = 2.0  
LANE_ALIGNMENT_THRESHOLD_X = 1.5