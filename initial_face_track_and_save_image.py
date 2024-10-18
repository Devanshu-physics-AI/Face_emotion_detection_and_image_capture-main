import numpy as np
import cv2
import tensorflow as tf
import time
model=tf.keras.models.load_model("/users/devanshusharma/desktop/my_model3.keras")
#function to make image compactible with our code
def resize(x):
    img=cv2.resize(x,(224,224))
    img=np.expand_dims(img,axis=0)
    # img=img[0]
    img=img/255  #normalise my image
    return img
    

cap =cv2.VideoCapture(0)
# frame=cv2.imread("/users/devanshusharma/desktop/a.jpeg")

 #capture our phtotos
# while True:
a,frame=cap.read()
 #convert my image to 1 channel to use cascade function
channel1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
face_track=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
 # #since we have got the frames now track faces on that image
face=face_track.detectMultiScale(channel1,minSize=(25,25),minNeighbors=3,scaleFactor=1.05)
 #since Cascade is pre-defined function it give (x,y,w,h)=initial pixel,end pixel, width and the heigt
if len(face)>0:
    for x,y,w,h in face:
        # (x,y,w,h)=face[i]
        model_image=frame[y:y+h,x:x+w]
        #till now we have detected the region of interest/face area
        #now showing that area on the real image
        model_image=cv2.rectangle(model_image,(x,y),(x+w,y+h),color=(0,255,255),thickness=3)
        final_image=resize(model_image)
        print(final_image.shape)
        pred=model.predict(final_image)
        print(pred)
        pred=np.array(pred)
        max_index=np.argmax(pred)
        list=["angry","disgust","fear","happy","neutral","sad","surprise"]
        expression=list[max_index]
        frame=cv2.putText(frame,text=expression,org=(x,y-10),fontFace=cv2.FONT_HERSHEY_PLAIN,thickness=4,color=(235,206,135),fontScale=5)
        cv2.imwrite(f"/users/devanshusharma/desktop/img.jpeg",frame)
        print(expression)
        time.sleep(1)
        

        
else:
    print("NO image is detected")
    cv2.imwrite(f"/users/devanshusharma/desktop/img.jpeg",frame)