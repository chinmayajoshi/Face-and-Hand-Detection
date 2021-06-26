import cv2 as cv
import mediapipe as mp

def detectFace(frame):
    
    # FaceDetection model object
    mfd = mp.solutions.face_detection
    fd = mfd.FaceDetection(model_selection = 1, min_detection_confidence=0.5)
    
    # for drawing detections over the origanl frame 
    mp_draw = mp.solutions.drawing_utils
    
    # detecting face 
    frame_detections = fd.process(frame)
    
    # modify the original frame object if detections available
    if frame_detections.detections:
        for detection in frame_detections.detections:
            mp_draw.draw_detection(frame, detection)
    
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
        
        # detect face via detectFace method
        face_frame = detectFace(frame)

        # show face detected frame
        cv.imshow('face_detection frame', face_frame)
       
        # press 'q' to quit capturing frames
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
