# object-tracking-cli

## Installation
```bash
pip install -e .
```

## Test
```bash
pytest .
```

## Usage

### Config
```yaml
video:
  desired_fps: 30. # upper bound - needed when processing is too fast
  output_width: 800 # resizing with fixed aspect ratio

detection:
  conf: 0.3 # confidence threshold 
  iou: 0.7 # NMS

trackers: # Trackers will be displayed side by side
  - NaiveTracker:
      use_kdtree: false
      max_missing_frames: 3
  
  - NaiveTracker:
      use_kdtree: true
      max_missing_frames: 3

```
```bash
object_tracking test_car.mp4 (--config)
```