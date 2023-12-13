from typing import Tuple, List


def make_bbox_larger(coordinates: Tuple[int, int, int, int], percentage: float) -> Tuple[float, float, float, float]:
    """ Makes a bounding box larger. It receives as input a bounding box in the format (xmin, ymin, xmax, ymax)
    and makes it larger by a percentage factor given by the input.

    Args:
        coordinates: Tuple representing bounding box coordinates in the format (xmin, ymin, xmax, ymax).
        percentage: The percentage by which to enlarge the bounding box.

    Returns: The new bounding box coordinates in the format (xmin, ymin, xmax, ymax).
    """

    xmin, ymin, xmax, ymax = [float(i) for i in coordinates]
    width = xmax - xmin
    height = ymax - ymin
    xmin -= percentage * float(width)
    xmax += percentage * float(width)
    ymin -= percentage * float(height)
    ymax += percentage * float(height)

    return xmin, ymin, xmax, ymax


def clip_detections(detections: List[Tuple[int, int, int, int]], image_shape) -> List[Tuple[int, int, int, int]]:
    """ Clips the detections to the boundaries of the image. In the case that a detection is completely out of the image
    it removes it completely from the list.

    Args:
        detections: List made out of tuples representing the coordinates of a detection in the format:
                    (x, y, width, height).
        image_shape: Tuple with the shape of the image in the format (h, w, c).

    Returns: The list of detections clipped to the boundary of the image in the same format as the input.
    """

    clipped_detections: List[Tuple] = []
    for i in range(len(detections)):
        x, y, w, h = detections[i]
        x_end = x + w
        y_end = y + h
        img_height, img_width, _ = image_shape

        # The detection is completely outside or the image
        if x > img_width or y > img_height or x_end < 0 or y_end < 0:
            continue

        # If one of the sides of the detection is outside the image it clips it to its boundaries.
        clipped_detection = list(detections[i])
        if x < 0:
            clipped_detection[0] = 0
            # When clipping x the width of the detection should account for the change
            # The width should be reduced by the distance between x and 0 in order for the x_end to be in the same
            # place as before. Works similarly for the height.
            clipped_detection[2] += int(x)
            w = clipped_detection[2]
        if y < 0:
            clipped_detection[1] = 0
            # When clipping y the height of the detection should account for the change
            clipped_detection[3] += int(y)
            h = clipped_detection[3]

        if x + w > img_width:
            clipped_detection[2] = int(img_width - x)
        if y + h > img_height:
            clipped_detection[3] = int(img_height - y)

        clipped_detections.append(tuple(clipped_detection))

    return clipped_detections
