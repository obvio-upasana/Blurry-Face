import os
import cv2 as cv
import mediapipe as mp

output= './output'
if not os.path.exists(output):
    # Create the output directory if it does not exist
    os.makedirs(output)

# Read Image
img_path='pranav2.jpg'
img=cv.imread(img_path)

H, W, _ = img.shape 


#detect faces
face_detect=mp.solutions.face_detection

with face_detect.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    img_rgb= cv.cvtColor(img, cv.COLOR_BGR2RGB)
    out=face_detection.process(img_rgb)

    if out.detections is not None:   #If a human face is detected
         for detection in out.detections:
            location_data= detection.location_data
            bbox = location_data.relative_bounding_box #Having a box around detection

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height #bounding box parameters

            x1 = int(x1*W)
            y1 = int(y1*H)            
            w = int(w*W)
            h = int(h*H)

            #img = cv.rectangle(img, (x1, y1), (x1 +w, y1 +h), (255,0,255), 10) #The actual shape of the bounding box and its parameters

            # Bluring the detected face
            img[y1: y1+h, x1:x1+w, :]  = cv.blur(img[y1: y1+h, x1:x1+w, :], (60,40)) #Blurring the exact area of the detected face faound in the bounding box
    cv.imshow('FAce', img)
    cv.waitKey(0)
# To save the img
cv.imwrite(os.path.join (output, 'output.png'), img) #Saving the image with blurred face


