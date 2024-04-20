import cv2

def resize_with_aspect_ratio(image, target_width=None, target_height=None):
    height, width = image.shape[:2]

    if target_width is not None and target_height is not None:
        raise ValueError("Only one of target_width or target_height should be provided.")

    if target_width is not None:
        ratio = target_width / width
        resized_image = cv2.resize(image, (target_width, int(height * ratio)))

    elif target_height is not None:
        ratio = target_height / height
        resized_image = cv2.resize(image, (int(width * ratio), target_height))

    else:
        raise ValueError("Either target_width or target_height should be provided.")

    return resized_image