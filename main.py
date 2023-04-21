import numpy as np
import cv2
from datetime import datetime

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# Open the webcam.
cap = cv2.VideoCapture(0)


fps = 15.

# Stores past frames for a short period of time.
# If a human is detected, the saved video will include
# any frames still stored in the buffer.
frame_buffer = []

# Timestamps corresponding to each frame in `frame_buffer`.
time_buffer = []

# The length of the frame buffer in frames.
frame_buffer_length = 10

frame_size = (640, 480)

# Video writer used to save files.
out = None
    
# The normal state of the application.
STATE_SCAN = "scanning"
# We may have detected a person. Write the frames to file.
STATE_ALERT = "alert"
# We lost sight of a person. We dont' know if they're really gone
# or if they just stopped being detected.
# We should keep writing to the video file for a while in case
# we re-detect them.
STATE_LOST = "lost"

# If we are in STATE_LOST, this keeps track of when
# we entered the current state.
entered_lost = None

current_state = STATE_SCAN
    
while(True):
    # Capture the next frame from the webcam.
    ret, frame = cap.read()

    # Resize and convert to grayscale for faster detection.
    frame = cv2.resize(frame, frame_size)

    # Runs HOG to detect any potential humans.
    # This returns a bounding box for each potential human.
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8))
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    # Draw the bounding boxes in the image.
    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    if current_state == STATE_SCAN:
        now = datetime.now()
    
        # Add the current frame to the buffer.
        frame_buffer.append(frame)
        time_buffer.append(datetime.now())
    
        if len(boxes) != 0:
            current_state = STATE_ALERT
            print("Maybe a person?")
            
            file_name = now.strftime("Recording-%d-%m-%y--%H-%M-%S.avi")
            print(file_name)
            out = cv2.VideoWriter(
                file_name,
                cv2.VideoWriter_fourcc(*'MJPG'),
                fps,
                frame_size)
            for frame in frame_buffer:
                out.write(frame.astype('uint8'))
            frame_buffer.clear()
            time_buffer.clear()
            
        # Remove old frames from the buffer.
        while len(time_buffer) > 0 and (now - time_buffer[0]).seconds > 10:
            print("Removing an old frame...")
            frame_buffer.pop(0)
            time_buffer.pop(0)
    # We saw a person last frame. See if they're still there.
    elif current_state == STATE_ALERT:
        out.write(frame.astype('uint8'))
        
        # Draw a recording icon.
        cv2.circle(frame, (15,15), 10, (0,0,255), -1)
    
        if len(boxes) != 0:
            print("still maybe a person")
        else:
            current_state = STATE_LOST
            entered_lost = datetime.now()
            print("We lost sight of the person.")
    # We just lost sight of a person. See if we can find them again.
    elif current_state == STATE_LOST:
        out.write(frame.astype('uint8'))
        
        # Draw a recording icon.
        cv2.circle(frame, (15,15), 10, (0,0,255), -1)
        
        if len(boxes) != 0:
            current_state = STATE_ALERT
            print("We found the person again!")
        elif (datetime.now() - entered_lost).seconds > 5:
            current_state = STATE_SCAN
            out.release()
            out = None
            print("We're assuming the person is gone now.")
        else:
            print("Looking for the person again...")
    
    
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
