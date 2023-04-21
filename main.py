import numpy as np
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# Open the webcam.
cap = cv2.VideoCapture(0)


# Stores past frames for a short period of time.
# If a human is detected, the saved video will include
# any frames still stored in the buffer.
frame_buffer = []

# The length of the frame buffer in seconds.
frame_buffer_length = 10

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))
    
# The normal state of the application.
STATE_SCAN = "scanning"
# We may have detected a person. Write the frames to file.
STATE_ALERT = "alert"
# We lost sight of a person. We dont' know if they're really gone
# or if they just stopped being detected.
# We should keep writing to the video file for a while in case
# we re-detect them.
STATE_LOST = "lost"

current_state = STATE_SCAN
    
while(True):
    # Capture the next frame from the webcam.
    ret, frame = cap.read()

    # Resize and convert to grayscale for faster detection.
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Runs HOG to detect any potential humans.
    # This returns a bounding box for each potential human.
    boxes, weights = hog.detectMultiScale(gray, winStride=(8,8) )
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    # Draw the bounding boxes in the image.
    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
    
    # Write the output video 
    #out.write(frame.astype('uint8'))
    
    if current_state == STATE_SCAN:
        if len(boxes) != 0:
            current_state = STATE_ALERT
            print("Maybe a person?")
    # We saw a person last frame. See if they're still there.
    elif current_state == STATE_ALERT:
        if len(boxes) != 0:
            print("still maybe a person")
        else:
            current_state = STATE_LOST
            print("We lost sight of the person.")
    # We just lost sight of a person. See if we can find them again.
    elif current_state == STATE_LOST:
        if len(boxes) != 0:
            current_state = STATE_ALERT
            print("We found the person again!")
        else:
            print("Looking for the person again...")
    
    frame_buffer.append(frame)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
