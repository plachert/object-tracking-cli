# Object Tracking
This tool enables to develop trackers and compare their performance visually.

## Installation
```bash
pip install -e .
```

## Test
```bash
pytest .
```

## Usage
```bash
object_tracking test_car.mp4 (--config path_to_custom_config)
```

### Default Config
```yaml
video:
  desired_fps: 30
  output_width: 800

detection:
  conf: 0.5
  iou: 0.5

trackers:
  - Motion Agnostic:
      max_missing_frames: 10
      cost_matrix_func:
        iou_cost_matrix:
      assignment_func:
        hungarian_assignment:
          th: 0.9
      motion_model_cls:
        MotionAgnosticModel:

  - KFCentroidVelocityModel:
      max_missing_frames: 10
      cost_matrix_func:
        iou_cost_matrix:
      assignment_func:
        hungarian_assignment:
          th: 0.9
      motion_model_cls:
        KFCentroidVelocityModel:

```

## Comparing trackers
The trackers defined in the `yaml` config file will appear side by side:

![](https://github.com/plachert/object-tracking-cli/blob/develop/examples/compare.gif)

## Development
`MultiObjectTracker` is created in `Composition over inharitance` spirit. Developing tracking methods should be done by developing the components of MOT rather than subclassing. The components are:
- cost_matrix_func: function that for 2 sets of bounding boxes creates a normalized cost function.
- assignment_func: function that creates a matching between two sets of bounding boxes given their cost matrix
- motion_model: To be implemented. 


## Testing trackers

1. Add tracker to `test_object_tracking.py`:
```python
test_trackers = [
    (
        MultiObjectTracker,
        {
            "assignment_func": greedy_assignment,
            "cost_matrix_func": euclidean_cost_matrix,
        },
    ),
    (
        MultiObjectTracker,
        {
            "assignment_func": partial(hungarian_assignment, th=1.0),
            "cost_matrix_func": euclidean_cost_matrix,
        },
    ),
    (
        MultiObjectTracker,
        {"assignment_func": greedy_assignment, "cost_matrix_func": iou_cost_matrix},
    ),
    (
        MultiObjectTracker,
        {
            "assignment_func": partial(hungarian_assignment, th=1.0),
            "cost_matrix_func": iou_cost_matrix,
        },
    ),
]
```
2. Run tests: `pytest .`



