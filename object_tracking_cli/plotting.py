import cv2


def plot_bboxes(frame, bboxes, class_to_color_and_name):
    for x1, y1, x2, y2, class_ in bboxes:
        color, name = class_to_color_and_name[class_]
        cv2.putText(
            frame,
            name,
            (x1 - 10, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

def plot_tracking(frame, tracker, tracker_name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    color = (0, 0, 255)
    cv2.putText(frame, tracker_name, (10, 30), font, font_scale, color, thickness)
    for object_id, (x, y) in tracker.objects.items():
        text = f"ID {object_id}"
        cv2.putText(
            frame,
            text,
            (x - 10, y - 10),
            font,
            font_scale,
            color,
            thickness,
        )
        cv2.circle(frame, (x, y), 4, color, -1)