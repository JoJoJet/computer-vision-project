import cv2

def get_sift_matches(sift, image, template_image):
    kp1, des1 = sift.detectAndCompute(image, None)
    kp2, des2 = sift.detectAndCompute(template_image, None)
    
    # Exit early if the template has no keypoints to match on.
    if len(kp2) == 0:
        return False, []
    
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    
    matched_points = []
    matchesMask = [[0,0] for i in range(len(matches))]
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i] = [1,0]
            matched_points.append(kp1[m.queryIdx])
    
    draw_params = dict(matchColor=(0,255,0),
                        singlePointColor=(245,0,0),
                        matchesMask=matchesMask,
                        flags=cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(image,kp1,template_image,kp2,matches,None,**draw_params)
    cv2.imshow('sift',img3)
    
    is_good_match = len(matched_points) >= len(kp2) / 2
    
    return is_good_match, matched_points
