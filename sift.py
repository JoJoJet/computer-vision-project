import cv2
from math import log2, floor, ceil
from itertools import product

def get_sift_matches(sift, image, template_image):
    kp1, des1 = sift.detectAndCompute(image, None)
    kp2, des2 = sift.detectAndCompute(template_image, None)
    
    # Exit early if the template has no keypoints to match on.
    if len(kp2) == 0:
        return []
    
    matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
    good_matches = []
    matchesMask = [[0,0] for i in range(len(matches))]
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i] = [1,0]
            good_matches.append((kp1[m.queryIdx], kp2[m.trainIdx]))
    
    img3 = cv2.drawMatchesKnn(image,kp1,template_image,kp2,matches,None,
                            matchColor=(0,255,0),
                            singlePointColor=(245,0,0),
                            matchesMask=matchesMask,
                            flags=cv2.DrawMatchesFlags_DEFAULT)
    cv2.imshow('sift',img3)
        
    if len(good_matches) == 0:
        return []
    
    # Rounds a transform into a set of bins representing the
    # two nearest bins in each dimension.
    # Theta should be measured in degrees.
    def into_bins(x, y, s, theta):
        x1 = floor(x / 100) * 100
        x2 = ceil(x / 100) * 100
        y1 = floor(y / 100) * 100
        y2 = ceil(y / 100) * 100
        s1 = 2**(floor(log2(s)))
        s2 = 2**(ceil(log2(s)))
        theta1 = floor(theta / 30) * 30
        theta2 = ceil(theta / 30) * 30
        return [x1, x2], [y1, y2], [s1, s2], [theta1, theta2]
    
    # For each matched feature, vote for its nearby hough bins.
    hough = {}
    for i,(m,n) in enumerate(good_matches):
        dx, dy, ds, d_theta = into_bins(n.pt[0]-m.pt[0], n.pt[1]-m.pt[1], n.size/m.size, n.angle-m.angle)
        for x in dx:
            for y in dy:
                for s in ds:
                    for theta in d_theta:
                        hough.setdefault((x,y,s,theta),[]).append(i)
    
    # Find the bin with the most votes.
    best_bin = max(hough, key = lambda k: len(hough[k]))
    matched_points = [good_matches[i][0] for i in hough[best_bin]]
    
    # The number of matches that need to fall into the maximum bin
    # in order for us to recognize an object.
    match_threshold = min(2, len(kp2))
    if len(matched_points) >= match_threshold:
        return matched_points
    else:
        return []
