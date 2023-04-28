import numpy as np
import cv2
from datetime import datetime
from nms import non_max_suppression
from sift import get_sift_matches

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

sift = cv2.SIFT_create(400)

cv2.startWindowThread()

# Open the webcam.
cap = cv2.VideoCapture(0)

frame_size = (640, 480)

select_region = True;
region_a = None
region_b = None
current_mouse_position = None

def select_region(event,x,y,flags,param):
    global region_a, region_b, select_region, current_mouse_position
    if event == cv2.EVENT_MOUSEMOVE:
        current_mouse_position = (x,y)
    if event == cv2.EVENT_LBUTTONDOWN:
        if region_a == None:
            region_a = (x,y)
        elif region_b == None:
            region_b = (x,y)
            select_region = False

def order_points(a,b):
    x1,x2 = sorted([a[0],b[0]])
    y1,y2 = sorted([a[1],b[1]])
    return (x1,y1), (x2,y2)

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', select_region)

# Allow the user to select a region that will be monitored.
while(select_region):
    ret, frame = cap.read()
    frame = cv2.resize(frame, frame_size)
    
    if current_mouse_position != None:
        if region_a != None:
            cv2.rectangle(frame,region_a,current_mouse_position,(200,200,200),2)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
assert(region_a != None)
assert(region_b != None)
region_a, region_b = order_points(region_a, region_b)

# Find a template image by cropping out the region the user selected.
_, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
template_image = frame[region_a[1]:region_b[1], region_a[0]:region_b[0]]
template_keypoints, template_descriptors = sift.detectAndCompute(template_image, None)

last_saw_template = datetime.now()
last_alerted_template = None

fps = 15.

# Stores past frames for a short period of time.
# If a human is detected, the saved video will include
# any frames still stored in the buffer.
frame_buffer = []

# Timestamps corresponding to each frame in `frame_buffer`.
time_buffer = []

# The length of the frame buffer in frames.
frame_buffer_length = 10

# Whether or not old frames should be removed from the buffer
# when it exceeds `frame_buffer_length`.
do_flush_buffer = True

# Video writer used to save files.
out = None
    
class State:
    # The normal state of the application.
    SCAN = "scanning"
    # We may have detected a person. Write the frames to file.
    ALERT = "alert"
    # We lost sight of a person. We dont' know if they're really gone
    # or if they just stopped being detected.
    # We should keep writing to the video file for a while in case
    # we re-detect them.
    LOST = "lost"


# If we are in State.LOST, this keeps track of when
# we entered the current state.
entered_lost = None

current_state = State.SCAN
    
while(True):
    now = datetime.now()
    
    # Capture the next frame from the webcam.
    ret, frame = cap.read()

    # Resize and convert to grayscale for faster detection.
    frame = cv2.resize(frame, frame_size)

    # Runs HOG to detect any potential humans.
    # This returns a bounding box for each potential human.
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8), scale=1.01)
    boxes = [[x, y, x + w, y + h] for (x, y, w, h) in boxes]
    boxes, weights, suppressed = non_max_suppression(boxes, list(weights))
    
    # Prune any boxes floating above the ground
    # (this depends heavily on the camera angle and environment,
    # and should be tuneable).
    # Ideally we could mathematically estimate the distance from the ground
    # in a more sophisticated way.
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # As y increases, we move further down the image.
        if y2 < frame_size[1] / 2:
            boxes.pop(i)
            weights.pop(i)
            suppressed.append([x1, y1, x2, y2])

    # Draw the bounding boxes in the image.
    for (x1, y1, x2, y2), c in zip(boxes, weights):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{c:.3f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 1, cv2.LINE_AA)
    # Draw the bounding boxes that have been suppressed.
    # This is mainly just to ensure that non-max suppression works.
    for (x1, y1, x2, y2) in suppressed:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 50, 0), 2)
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    matched_points = get_sift_matches(sift, gray_frame, template_image, template_keypoints, template_descriptors)
    if len(matched_points) > 0:
        last_saw_template = now
        do_flush_buffer = True
    else:
        do_flush_buffer = False
        needs_alerting = last_alerted_template == None or last_saw_template > last_alerted_template
        if needs_alerting and (now - last_saw_template).seconds >= 1:
            current_state = State.ALERT
            last_alerted_template = now
            
            # Start a recording. Save the current frame buffer,
            # to give the recording more context when viewed.
            out = start_recording(now, frame_size, fps)
            for frame in frame_buffer:
                out.write(frame.astype('uint8'))
            frame_buffer.clear()
            time_buffer.clear()
    
    frame = cv2.drawKeypoints(frame, matched_points, 0, (0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    
    def start_recording(timestamp, frame_size, fps):
        return cv2.VideoWriter(
            now.strftime("Recording-%d-%m-%y--%H-%M-%S.avi"),
            cv2.VideoWriter_fourcc(*'MJPG'),
            fps,
            frame_size)
        
    
    match current_state:
        case State.SCAN:
        
            # Add the current frame to the buffer.
            frame_buffer.append(frame)
            time_buffer.append(datetime.now())
        
            # If we detected a person, transition to the alert state.
            if len(boxes) != 0:
                current_state = State.ALERT
                
                # Start a recording. Save the current frame buffer,
                # to give the recording more context when viewed.
                out = start_recording(now, frame_size, fps)
                for frame in frame_buffer:
                    out.write(frame.astype('uint8'))
                frame_buffer.clear()
                time_buffer.clear()
                
            # Remove old frames from the buffer.
            while do_flush_buffer and len(time_buffer) > 0 and (now - time_buffer[0]).seconds > 10:
                print("Removing an old frame...")
                frame_buffer.pop(0)
                time_buffer.pop(0)
            
        # We saw a person last frame. See if they're still there.
        case State.ALERT:
            out.write(frame.astype('uint8'))
            
            # Draw a filled recording icon.
            cv2.circle(frame, (15,15), 10, (0,0,255), -1)
        
            if len(boxes) != 0:
                print("still maybe a person")
            else:
                current_state = State.LOST
                entered_lost = datetime.now()
        
        # We just lost sight of a person. See if we can find them again.
        case State.LOST:
            out.write(frame.astype('uint8'))
            
            # Draw a hollow recording icon.
            cv2.circle(frame, (15,15), 10, (0,0,255), 3)
            
            if len(boxes) != 0:
                current_state = State.ALERT
            elif (datetime.now() - entered_lost).seconds > 5:
                current_state = State.SCAN
                out.release()
                out = None
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
if out != None:
    out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
