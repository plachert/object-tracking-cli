from object_tracking_cli.object_tracking.trackers import NaiveTracker
import time


def perfect_move(n_objects: int = 10):
    bboxes = [(0, 2*idx, 2, 2) for idx in range(n_objects)]
    while True:
        bboxes = [(bbox[0]+1, bbox[1], bbox[2]+1, bbox[3]) for bbox in bboxes]
        yield bboxes


if __name__ == "__main__":
    tracker = NaiveTracker(use_kdtree=False)
    kd_tracker = NaiveTracker(use_kdtree=True)

    def profile_tracker(tracker, n_objects=500, n_updates=100):
        start = time.time()
        gen = perfect_move(n_objects)
        for _ in range(n_updates):
            bboxes = next(gen)
            tracker.update(bboxes)
        end = time.time()
        return end - start

    print(f"Regular tracker: {profile_tracker(tracker)}")
    print(f"KD-tree tracker: {profile_tracker(kd_tracker)}")
