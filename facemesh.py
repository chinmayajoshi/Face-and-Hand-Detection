import cv2 as cv
import mediapipe as mp

def detectFaceMesh(frame):

    # FaceMesh model object
    mfm = mp.solutions.face_mesh
    fm = mfm.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # for drawing detections over the origanl frame 
    mp_draw = mp.solutions.drawing_utils
    drawing_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)

    # detecting face mesh
    frame_detections = fm.process(frame)
    
    # modify the original frame object if detections available
    if frame_detections.multi_face_landmarks:
        for face_landmarks in frame_detections.multi_face_landmarks:
            mp_draw.draw_landmarks(image=frame, landmark_list=face_landmarks, connections=mfm.FACE_CONNECTIONS, landmark_drawing_spec=drawing_spec, connection_drawing_spec=drawing_spec)
    
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
        
        # detect face via detectFaceMesh method
        face_frame = detectFaceMesh(frame)

        # show facemesh detected frame
        cv.imshow('face_mesh frame', face_frame)
       
        # press 'q' to quit capturing frames
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
