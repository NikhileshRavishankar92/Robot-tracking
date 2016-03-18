# Import the necessary libraries

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

frame_count = 0
Frame = range(68)
# Insert the file location of the video to create an object cap
cap = cv2.VideoCapture('C:\\Users\\nikhilesh\\Desktop\\Ground Test\\60\\60_2\\60_2.MOV')
centre_buffer = []
velocity_buffer = np.loadtxt('velocity_correct_at_60PWM_2.txt')
i = 0
# Create Window to see the video 
cv2.namedWindow('Window',cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Window', cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

# Create an out object to save the video file
out = cv2.VideoWriter('output_3.avi',-1, 20.0, (600,337))


while frame_count < 323:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width = 600)
    frame_count+=1
    
    # Remove noise from the frames
    frame1 = cv2.medianBlur(frame, 5)
    frame2 = cv2.GaussianBlur(frame1, (5,5),0)

    hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    
    # Threshold the image such that only the pencil is visible
    upper_range = np.array([179,255,255])
    lower_range = np.array([7, 77, 84])
    frame3 = cv2.inRange(hsv, lower_range, upper_range)
    erode = cv2.erode(frame3,None, iterations = 2)
    dilate = cv2.dilate(erode,None, iterations = 2)
    result = cv2.bitwise_and(dilate,dilate, mask = frame3)
    
    # Apply the Canny edge detector
    result2 = cv2.Canny(result, 35, 125)
    
    # Find contours in the masked image and select the maximum one
    (contours,_)= cv2.findContours(result2.copy(),cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0 and frame_count > 255:
        c_max = max(contours, key = cv2.contourArea)
        M = cv2.moments(c_max)
        cx = np.int0(M['m10']/M['m00'])
        cy = np.int0(M['m01']/M['m00'])    
        peri = cv2.arcLength(c_max, True)
        approx = cv2.approxPolyDP(c_max, 0.05 * peri, True)
        cv2.circle(frame,(cx,cy),2,[0,255,0],-1)    
        cv2.putText(frame, "%.4f m/s" % velocity_buffer[i],(60,60) ,cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 255, 0), 2)
        out.write(frame)
        centre_buffer.append((cx,cy,frame_count))
        i+=1
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break     
                        
cap.release()
out.release()
cv2.destroyAllWindows()  

# Plot the velocity profile of the vehicle
plt.plot(Frame[1:],velocity_buffer[1:])
plt.xlabel('Frame')
plt.ylabel('Velocity m/s')
plt.show()
     
           
    
    
    
    