import pytest
from object_tracking_cli.object_tracking.trackers import NaiveTracker
import random 

test_trackers = [
    (NaiveTracker, {'use_kdtree': False}), 
    (NaiveTracker, {'use_kdtree': True}),
    ]


@pytest.fixture(scope="function")
def perfect_move():
    n_objects = 10
    def gen_bboxes():
        bboxes = [(0, 2*idx, 2, 2) for idx in range(n_objects)]
        while True:
            bboxes = [(bbox[0]+1, bbox[1], bbox[2]+1, bbox[3]) for bbox in bboxes]
            yield tuple(bboxes)
    return gen_bboxes(), n_objects

@pytest.fixture(scope="function", params=test_trackers)
def tracker_with_params(request):
    tracker_class, params = request.param
    return tracker_class, params


def test_naive_tracker_on_perfect_move(perfect_move, tracker_with_params):
    gen_bboxes, n_objects = perfect_move
    tracker_class, params = tracker_with_params
    tracker = tracker_class(**params)
    for _ in range(10):
        bboxes = next(gen_bboxes)
        tracker.update(bboxes)
        tracker.no_objects == n_objects


def test_below_missing_frames(perfect_move, tracker_with_params):
    max_missing_frames = random.randint(1, 10)
    gen_bboxes, n_objects = perfect_move
    tracker_class, params = tracker_with_params
    tracker = tracker_class(**params | {"max_missing_frames": max_missing_frames})
    for idx in range(100):
        bboxes = next(gen_bboxes)
        if idx % max_missing_frames:
            bboxes = []
        tracker.update(bboxes)
        assert tracker.no_objects == n_objects


def test_over_missing_frames(perfect_move, tracker_with_params):
    max_missing_frames = random.randint(1, 10)
    gen_bboxes, n_objects = perfect_move
    tracker_class, params = tracker_with_params
    tracker = tracker_class(**params | {"max_missing_frames": max_missing_frames})
    missing_frames = 0
    for idx in range(100):
        bboxes = next(gen_bboxes)
        if idx % (max_missing_frames + 2):
            bboxes = []
            missing_frames += 1
        else:
            missing_frames = 0
        tracker.update(bboxes)
        if missing_frames > max_missing_frames:
            assert tracker.no_objects == 0
        else:
            assert tracker.no_objects == n_objects


def test_perfect_assignment(perfect_move, tracker_with_params):
    gen_bboxes, _ = perfect_move
    tracker_class, params = tracker_with_params
    tracker = tracker_class(**params)
    init_bboxes = next(gen_bboxes)
    tracker.update(init_bboxes)
    initial_objects = tracker.objects
    for _ in range(100):
        bboxes = next(gen_bboxes)
        tracker.update(bboxes)
        updated_objects = tracker.objects
        for object_id, centroid in updated_objects.items():
            _, y = centroid
            assert y == initial_objects[object_id][1]
