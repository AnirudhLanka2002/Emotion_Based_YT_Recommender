import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
import os

def reset_emotion_file():
    if os.path.exists("emotion1.npy"):
        os.remove("emotion1.npy")

def refresh_app():
    reset_emotion_file()
    st.session_state["running"] = "on"
    st.experimental_rerun()

holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

if "running" not in st.session_state:
    st.session_state["running"] = "on"

class EProcessor:
    def __init__(self):
        self.model, self.labels = load_model("model.h5"), np.load("labels.npy")
    
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []
        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)
            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)
            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)
            lst = np.array(lst).reshape(1, -1)
            pred = self.labels[np.argmax(self.model.predict(lst))]
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
            np.save("emotion1.npy", np.array([pred]))
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
        return av.VideoFrame.from_ndarray(frm, format="bgr24")

lang = st.text_input("Enter Desired Language:")
sg = st.text_input("Enter Singer Name:")

if st.session_state["running"] != "off":
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EProcessor)

btn1 = st.button("Recommend a song bro")

if btn1:
    emo = np.load("emotion1.npy")[0]
    webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emo}+songs+by+{sg}")
    reset_emotion_file()
    st.session_state["running"] = "off"
    st.experimental_rerun()

refresh_btn = st.button("Refresh")

if refresh_btn:
    lang = ""
    sg = ""
    refresh_app()
