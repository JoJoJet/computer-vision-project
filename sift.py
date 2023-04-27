import cv2
from math import log2, floor, ceil
from itertools import product

def get_sift_matches(sift, image, template_image):
    kp1, des1 = sift.detectAndCompute(image, None)
    kp2, des2 = sift.detectAndCompute(template_image, None)
    print(des1)
    
    # Exit early if the template has no keypoints to match on.
    if len(kp2) == 0:
        return []
    
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    
    good_matches = []
    matchesMask = [[0,0] for i in range(len(matches))]
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i] = [1,0]
            good_matches.append((kp1[m.queryIdx], kp2[m.trainIdx]))
    
    draw_params = dict(matchColor=(0,255,0),
                        singlePointColor=(245,0,0),
                        matchesMask=matchesMask,
                        flags=cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(image,kp1,template_image,kp2,matches,None,**draw_params)
    cv2.imshow('sift',img3)
    
    # Rounds a transform into a set of bins representing the
    # two nearest bins in each dimension (total of sixteen bins).
    # Theta should be measured in degrees.
    def into_bins(x, y, theta):
        x1 = floor(x * 100) / 100
        x2 = ceil(x * 100) / 100
        y1 = floor(y * 100) / 100
        y2 = ceil(y * 100) / 100
        theta1 = floor(theta * 30) / 30
        theta2 = ceil(theta * 30) / 30
        return [x1, x2], [y1, y2], [theta1, theta2]
        
    if len(good_matches) == 0:
        return []
    
    # For each matched feature, vote for its nearby hough bins.
    hough = {}
    for i,(m,n) in enumerate(good_matches):
        dx, dy, d_theta = into_bins(n.pt[0]-m.pt[0], n.pt[1]-m.pt[1], n.angle-m.angle)
        ds = 2**(n.octave-m.octave)
        for x in dx:
            for y in dy:
                for theta in d_theta:
                    hough.setdefault((x,y,ds,theta),[]).append(i)
    
    # Find the bin with the most votes.
    best_bin = max(hough, key = lambda k: len(hough[k]))
    matched_points = [good_matches[i][0] for i in hough[best_bin]]
    print('matched ', len(matched_points), ' points')
        
    return matched_points
