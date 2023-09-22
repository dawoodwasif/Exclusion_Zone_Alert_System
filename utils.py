# def calculate_iou(box, polygon):
#     """
#     Calculate IoU between a bounding box (xyxy format) and a polygon.

#     Args:
#     - box (list): Bounding box coordinates in xyxy format [x1, y1, x2, y2].
#     - polygon (list): List of coordinates [(x1, y1), (x2, y2), ...] representing the polygon.

#     Returns:
#     - IoU (float): Intersection over Union value.
#     """
#     # Calculate the coordinates of the intersection rectangle
#     x1, y1, x2, y2 = box
#     intersection_x1 = max(x1, min(polygon[0][0], polygon[1][0], polygon[2][0], polygon[3][0]))
#     intersection_y1 = max(y1, min(polygon[0][1], polygon[1][1], polygon[2][1], polygon[3][1]))
#     intersection_x2 = min(x2, max(polygon[0][0], polygon[1][0], polygon[2][0], polygon[3][0]))
#     intersection_y2 = min(y2, max(polygon[0][1], polygon[1][1], polygon[2][1], polygon[3][1]))
    
#     # Calculate the area of intersection
#     intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
    
#     # Calculate the area of bounding box and polygon
#     box_area = (x2 - x1) * (y2 - y1)
    
#     # Calculate the area of the polygon using the shoelace formula
#     polygon_area = 0.5 * abs(
#         (polygon[0][0] * polygon[1][1] + polygon[1][0] * polygon[2][1] + polygon[2][0] * polygon[3][1] + polygon[3][0] * polygon[0][1])
#         - (polygon[1][0] * polygon[0][1] + polygon[2][0] * polygon[1][1] + polygon[3][0] * polygon[2][1] + polygon[0][0] * polygon[3][1])
#     )
#     # Calculate IoU
#     iou = intersection_area / (box_area + polygon_area - intersection_area)
#     return iou



import torch

def calculate_iou(box, polygon, bottom_percent=0.1):
    """
    Calculate IoU between the bottom 10% of a bounding box (xyxy format) and a polygon.

    Args:
    - box (list): Bounding box coordinates in xyxy format [x1, y1, x2, y2].
    - polygon (list): List of coordinates [(x1, y1), (x2, y2), ...] representing the polygon.
    - bottom_percent (float): The percentage of the bounding box's height to use for calculation (default is 0.1).

    Returns:
    - IoU (float): Intersection over Union value.
    """
    # Calculate the height of the bottom 10% of the bounding box
    x1, y1, x2, y2 = box
    box_height = y2 - y1
    bottom_10_percent_height = bottom_percent * box_height

    # Calculate the coordinates of the intersection rectangle
    intersection_x1 = max(x1, min(polygon[0][0], polygon[1][0], polygon[2][0], polygon[3][0]))
    intersection_y1 = max(y1 + box_height - bottom_10_percent_height, min(polygon[0][1], polygon[1][1], polygon[2][1], polygon[3][1]))
    intersection_x2 = min(x2, max(polygon[0][0], polygon[1][0], polygon[2][0], polygon[3][0]))
    intersection_y2 = min(y2, max(polygon[0][1], polygon[1][1], polygon[2][1], polygon[3][1]))

    # Calculate the area of intersection
    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)

    # Calculate the area of the polygon using the shoelace formula
    polygon_area = 0.5 * abs(
        (polygon[0][0] * polygon[1][1] + polygon[1][0] * polygon[2][1] + polygon[2][0] * polygon[3][1] + polygon[3][0] * polygon[0][1])
        - (polygon[1][0] * polygon[0][1] + polygon[2][0] * polygon[1][1] + polygon[3][0] * polygon[2][1] + polygon[0][0] * polygon[3][1])
    )

    # Calculate IoU
    iou = intersection_area / (bottom_10_percent_height * (x2 - x1) + polygon_area - intersection_area)
    return iou
