import numpy as np
import cv2
import streamlit as st
from keras.preprocessing import image
from keras.models import load_model
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
from PIL import Image
import os
import matplotlib.pyplot as plt
#from my_model.model import FacialExpressionModel
import time
#from bokeh.models.widgets import Div


# load model
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# load weights into new model
classifier =load_model(r'C:\Users\SOHAM\Desktop\PythonProject\EmotionDetectionCNN\model.h5')

#load face
try:
    face_cascade = cv2.CascadeClassifier(r'C:\Users\SOHAM\Desktop\PythonProject\EmotionDetectionCNN\haarcascade_frontalface_default.xml')
    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        # frame = np.array(frame.convert('RGB'))
        frame = frame.to_ndarray(format="bgr24")
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = img_gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = classifier.predict(roi)[0]
                print("Pred")
                print(prediction)
                print(prediction.argmax())
                label=emotion_labels[prediction.argmax()]
                print("label")
                print(label)
                label_position = (x,y)

                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('Emotion Detector',frame)
        

        return frame



def main():
    # Face Analysis Application #
    st.title("Live Emotion Detection Application")
    activities = ["Home", "Live Facial Emotion Detection", "Image Emotion Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    
 
    st.sidebar.markdown("""Developed by:""")
    st.sidebar.markdown(""" 1. Soham D. Deshpande  deshpandesoham@kbtcoe.org""")
    st.sidebar.markdown(""" 2. Achal kirti Shah shahachal123@kbtcoe.org""")
    st.sidebar.markdown(""" 3. Prachi P. Thete theteprachi123@kbtcoe.org""")
    st.sidebar.markdown(""" 4. Abhinandan A. Nahar naharabhinandan@kbtcoe.org""")
    st.sidebar.markdown(""" Guide: Dr. V.R.Sonawane""")
    
    
    def detect_faces(our_image):
        # font = cv2.FONT_HERSHEY_SIMPLEX
        our_image = np.array(our_image.convert('RGB'))
        image = cv2.cvtColor(our_image,1)
        # new_img = np.array(our_image.convert('RGB'))
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(new_img,1)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Detect faces
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)
	# Draw rectangle around the faces
        for (x, y, w, h) in faces:

            fc = img_gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(fc, (48, 48), interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)
            pred = classifier.predict(roi)[0]
            #pred = classifier.predict(roi)[0]
            print(pred)
            print("***************************")
            label=emotion_labels[pred.argmax()]
            print(label)
            print("///////////////////////////////")
            cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            #return img_gray,faces,pred 
            return image,faces,pred
    

    def load_image(image_file):
        img = Image.open(image_file)
        return img
    

    if choice == "Home":
        #from PIL import Image
        img = Image.open("img1.png")
        st.image(img, width=700)
        st.markdown(
                """
                <style>
                .reportview-container {
        img = Image.open("img1.png")
                   
                }
            .sidebar .sidebar-content {
        img = Image.open("img1.png")
                    
                }
                </style>
                """,
                unsafe_allow_html=True
            )
        html_temp_home1 = """<div style="background-color:#1152F7;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has two functionalities.
                 1. Real time face detection using web cam feed.
                 2. Real time face emotion recognization.
                 """)

    elif choice == "Live Facial Emotion Detection":
        
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)
        

    elif choice == "Image Emotion Detection":
        
        st.subheader("Image")


        image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
    
	#if image if uploaded,display the progress bar +the image
        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i+1)
            st.image(our_image)
        if image_file is None:
            st.error("No image uploaded yet")

        # Face Detection
        task = ["Faces"]
        feature_choice = st.sidebar.selectbox("Find Features",task)
        if st.button("Process"):
            if feature_choice == 'Faces':

				#process bar
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.05)
                    progress.progress(i+1)
				#end of process bar
				
                #result_img,result_faces,prediction = detect_faces(our_image)
                result_img,result_faces,prediction = detect_faces(our_image)
                if st.image(result_img) :
                    st.success("Found {} faces".format(len(result_faces)))
					

    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-image: url('img_girl.jpg');"">
                                    <h4 style="color:white;text-align:center;">
                                    Real time face emotion detection application using OpenCV, CNN model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                             		<div style="background-image: url('img1.png');">
                                 
                             		<h4 style="color:white;text-align:center;">This Application is developed by GROUP 1 using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose. </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()