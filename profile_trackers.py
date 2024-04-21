import time

from object_tracking_cli.object_tracking.trackers import MotionAgnosticTracker


def perfect_move(n_objects: int = 10):
    bboxes = [(0, 2 * idx, 2, 2, None, None) for idx in range(n_objects)]
    while True:
        bboxes = [
            (bbox[0] + 1, bbox[1], bbox[2] + 1, bbox[3], None, None) for bbox in bboxes
        ]
        yield bboxes


if __name__ == "__main__":
    naive_tracker = MotionAgnosticTracker(assignment_strategy="naive")
    kd_tracker = MotionAgnosticTracker(assignment_strategy="kd_tree")
    hungarian_tracker = MotionAgnosticTracker(assignment_strategy="hungarian")

    def profile_tracker(tracker, n_objects=500, n_updates=100):
        start = time.time()
        gen = perfect_move(n_objects)
        for _ in range(n_updates):
            bboxes = next(gen)
            tracker.update(bboxes)
        end = time.time()
        return end - start

    print(f"Naive tracker: {profile_tracker(naive_tracker)}")
    print(f"KD-tree tracker: {profile_tracker(kd_tracker)}")
    print(f"Hungarian tracker: {profile_tracker(hungarian_tracker)}")
