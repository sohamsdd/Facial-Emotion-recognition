import numpy as np
import cv2
import streamlit as st
from keras.preprocessing import image
from keras.models import load_model
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode


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
    activities = ["Home", "Facial Emotion Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    
    # st.sidebar.markdown(
    #     """
    #         Developed by: 
    #         Soham D. Deshpande    
    #         Email : deshpandesoham@kbtcoe.org
    #         Achal Shah
    #         Email : shahachal@kbtcoe.org
    #         Prachi Thete
    #         Email : theteprachi@kbtcoe.org
    #         Abhinandan Nahar
    #         Email:  naharabhinandan@kbtcoe.org
    #         NDMVPS's KBTCOE, NASHIK
    #         Guide:
    #         Dr. V.R.Sonawane""")
    st.sidebar.markdown("""Developed by:""")
    st.sidebar.markdown(""" 1. Soham D. Deshpande  deshpandesoham@kbtcoe.org""")
    st.sidebar.markdown(""" 2. Achal kirti Shah shahachal123@kbtcoe.org""")
    st.sidebar.markdown(""" 3. Prachi P. Thete theteprachi123@kbtcoe.org""")
    st.sidebar.markdown(""" 4. Abhinandan A. Nahar naharabhinandan@kbtcoe.org""")
    st.sidebar.markdown(""" Guide: Dr. V.R.Sonawane""")

    
    if choice == "Home":
        from PIL import Image
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
    elif choice == "Facial Emotion Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)
        


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
                                 
                             		<h4 style="color:white;text-align:center;">This Application is developed by Soham Deshpande using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose. </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()