import numpy as np

# Perorms non-max suppression, to merge any overlapping bounding boxes.
def non_max_suppression(boxes, weights, iou_threshold=0.5):
    keep = []
    keep_weights = []
    
    assert(len(boxes) == len(weights))
    while len(boxes) > 0:
        # Remove the bounding box with the hightest confidence value.
        s_i = np.argmax(weights)
        s_c = weights.pop(s_i)
        s = boxes.pop(s_i)
        keep.append(s)
        keep_weights.append(s_c)
        
        # Suppress any bounding boxes with a significant overlap.
        for i, (t, t_c) in enumerate(zip(boxes, weights)):
            iou = intersection_over_union(s, t)
            if iou > iou_threshold:
                boxes.pop(i)
                weights.pop(i)
    
        
    return keep, keep_weights

def intersection_over_union(s, t):
    s_x1, s_y1, s_x2, s_y2 = s
    t_x1, t_y1, t_x2, t_y2 = t
    
    # Find the area of each individual box.
    area_s = (s_x2 - s_x1) * (s_y2 - s_y1)
    area_t = (t_x2 - t_x1) * (t_y2 - t_y1)
    
    # Find the intersection box.
    x1 = max(s_x1, t_x1)
    y1 = max(s_y1, t_y1)
    x2 = min(s_x2, t_x2)
    y2 = min(s_y2, t_y2)
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = area_s + area_t - intersection
    
    return intersection / union
