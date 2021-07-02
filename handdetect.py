import cv2 as cv
import mediapipe as mp


# FaceMesh model object
# performance decreases dramatically when creating an instance inside the function call
mp_hands = mp.solutions.hands
mh = mp_hands.Hands(static_image_mode=False, max_num_hands = 2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
 
def detectHands(frame):

    # for drawing detections over the origanl frame 
    mp_draw = mp.solutions.drawing_utils

    # detecting face mesh
    frame_detections = mh.process(frame)
    
    # modify the original frame object if detections available
    if frame_detections.multi_hand_landmarks:
        for hand_landmarks in frame_detections.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
   
    # returns the original frame if no detection available
    return frame

if __name__ == '__main__':
    # capture object 
    cap = cv.VideoCapture(0)
    
    # start capturing from cam
    while True:
        ret, frame = cap.read()
        
        #flip frame horizontally
        frame = cv.flip(frame, 1)
       
        # show original frame
        cv.imshow('frame', frame)

        frame.flags.writeable = False
        
        # detect face via detectHands method
        hand_detect = detectHands(frame)

        # show hands detected frame
        cv.imshow('hand_detection frame', hand_detect)
       
        # press 'q' to quit capturing frames
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
