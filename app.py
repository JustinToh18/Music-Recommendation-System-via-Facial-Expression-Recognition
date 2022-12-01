import streamlit as st
import pandas as pd
import pymongo
import cv2
import tensorflow as tf
import numpy as np
import os
import time
import streamlit.components.v1 as components
import av
import datetime
import matplotlib.pyplot as plt

from PIL import Image
from timeit import default_timer as timer
from datetime import datetime
from dateutil import tz
from pymongo import MongoClient
from sklearn.neighbors import NearestNeighbors
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

cluster = MongoClient("mongodb+srv://Admin:011819@cluster0.g8qznq0.mongodb.net/?retryWrites=true&w=majority")
db = cluster["FER"]
collection = db["User Authentication"]
collection2 = db["Usage Duration"]
collection3 = db["Emotion"]

def add_userdata(username,password):
    collection.insert_one({"name": username, "password": password})

def login_user(username,password):
    data = collection.find_one({"name": username, "password": password})
    return data

@st.cache
def login_start(username, start):
    collection2.insert_one({"name": username, "start time": start})
    collection2.update_one({"name": username}, {"$set":{"start time": start}})

@st.cache
def logout_start(username, end, experience, usage):
    collection2.insert_one({"name": username, "end time": end, "experience": experience, "usage": usage})
    collection2.update_one({"name": username}, {"$set":{"end time": end}})
    collection2.update_one({"name": username}, {"$set":{"experience": experience}})
    collection2.update_one({"name": username}, {"$set":{"usage": usage}})

def store_emotion(emotion):
    collection3.insert_one({"emotion": emotion})
    
@st.cache(allow_output_mutation=True, max_entries=3)
def load_model():
    model = tf.keras.models.load_model("Final_model_testing2")
    return model

@st.cache(allow_output_mutation=True, max_entries=3)
def load_haar():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return cascade

font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

def play_video():
    class VideoProcessor:
        def recv(self,frame):

            frame = frame.to_ndarray(format="bgr24")
            new_model = load_model()
            faceCascade = load_haar()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = faceCascade.detectMultiScale(gray,1.1,4)
            for x,y,w,h in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                facess = faceCascade.detectMultiScale(roi_gray)

                for (ex, ey, ew, eh) in facess:
                    global final_image
                    face_roi = roi_color[ey:ey + eh, ex:ex + ew]
            
                    final_image = cv2.resize(face_roi, (224, 224))
                    final_image = np.expand_dims(final_image, axis = 0)
                    final_image = final_image/255.0
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    Predictions = new_model.predict(final_image)
                    font_scale = 1.5
                    font = cv2.FONT_HERSHEY_PLAIN
                
                    if (np.argmax(Predictions) == 3):
                        status = "Happy"
                        x1, y1, w1, h1 = 0, 0, 175, 75
                        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
                        store_emotion(status)
                        break
                    elif (np.argmax(Predictions) == 4):
                        status="Neutral"
                        x1, y1, w1, h1 = 0, 0, 175, 75
                        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
                        store_emotion(status)
                        break
                    elif (np.argmax(Predictions) == 5):
                        status = "Sad"
                        x1, y1, w1, h1 = 0, 0, 175, 75
                        cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
                        store_emotion(status)
                        break
                        
            return av.VideoFrame.from_ndarray(frame, format='bgr24')
            
    webrtc_streamer(
        key="detect-emotion",
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
    )
    
@st.cache(ttl=7*60)
def start_timer():
    start = timer()
    return start 
    
def main():
    
    st.markdown("<h1 style='text-align: center; color: white; background-color:#8B0000; font-size:40px'>Music Recommendation System via Facial Expression Recognition</h1>", unsafe_allow_html=True)

    menu = ["Home", "Login", "Register"]
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == "Home":
        st.markdown("<h2 style='text-align: center; color: white; font-size:40px'>Home</h2>", unsafe_allow_html=True)
        image = Image.open('images\homepage2.jpg')
        st.image(image, caption='Enjoy quality music recommendation with our app')
        st.write("")
        st.write("")
        st.markdown("***")
        st.markdown("<h1 style='text-align: center; color: white; background-color:#8B0000; font-size:40px'>About</h1>", unsafe_allow_html=True)
        st.write("")
        image = Image.open('images\homepage.jpg')
        st.image(image)
        
        st.markdown("<h1 style='text-align: justify; color: white;font-weight: normal; font-size:18px; line-height:1.8'>There is a problem in Malaysia where many people are facing mental depression problems especially during the COVID-19 pandemic. Despite having access to many different entertainment methods and treatment, there is still a large growth in people suffering from mental illness. Hence, causing heavy and negative impact to many especially the adolescents. One possible cause of the problem was people were not able to easily identify the symptoms of mental illness. Hence, technology will be utilized to tackle this problem. Once their mental illness has been identified, music will play a significant role in improving their mood as music deeply connects to our life. Hence, music has the potential healing factor to the problem. This has facilitated the development of Music Recommendation System via Facial Expression Recognition which can allow users to identify their emotional state and enhance their mood whenever they like.  </h1>", unsafe_allow_html=True)
        
        st.markdown("<h1 style='text-align: justify; color: white;font-weight: normal; font-size:18px; line-height:1.8'> The system will allow users to: <br><br> ✔️ To live stream their facial expression and identify their real time emotional state <br><br> ✔️ To get a list of recommended songs which can help improve their mood</h1>", unsafe_allow_html=True)
        
        st.write("")
        st.write("")
        st.markdown("***")
        st.markdown("<h1 style='text-align: center; color: white; background-color:#8B0000; font-size:40px'>Depression</h1>", unsafe_allow_html=True)
        
        st.markdown("<h1 style='text-align: justify; color: white;font-weight: normal; font-size:18px; line-height:1.8'>What is depression?<br><br>➡ Depression is a serious matter that affects people of all ages by influencing their thoughts and action in a negative way. <br><br>What is the consequence if depression is left untreated? <br><br> ➡ Depression causes physical and emotional problem to a person, which leads to inability to carry out work in everyday life.</h1>", unsafe_allow_html=True)
        
        st.write("")
        st.write("")
        st.markdown("***")
        st.markdown("<h1 style='text-align: center; color: white; font-size:30px'>Click on the videos below to know more about depression!<br></h1>", unsafe_allow_html=True)
        st.write("")
        st.write("")
        col1, col2 = st.columns(2)
        with col1:
            st.video("https://youtu.be/d7NPnvKFs2Y")
            st.video("https://youtu.be/_eXdP7ojSkI")
        with col2:
            st.video("https://youtu.be/z-IR48Mb3W0")
            st.video("https://youtu.be/-MNp9bmNI60")

    elif choice == "Login":
        
        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type='password')
        
        if "login_state" not in st.session_state :
            st.session_state.login_state = False
            
        if st.sidebar.button("Login") or st.session_state.login_state:
            st.session_state.login_state = True
            result = login_user(username,password)
                
            if result:
                st.write("")
                st.success("Logged in as {}".format(username))
                start = start_timer()
        
                st.markdown("<h2 style='text-align: center; color: white; font-size:40px'>Tasks</h2>", unsafe_allow_html=True)
                task = st.selectbox("",["Read Policy", "Detect Emotion", "Play Recommended Music", "Rate Your Experience"])
                if task == "Read Policy":
                    st.markdown("***")
                    st.markdown("<h2 style='text-align: center; color: white; font-size:40px'>Terms of Use</h2>", unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: justify; color: white; font-size:20px; line-height: 1.8'>Please read the terms of use below before proceeding to using other features of our system.</h3>", unsafe_allow_html=True)
                    st.write("")
                    st.markdown("<h3 style='text-align: justify; color: white; font-weight:normal; font-size:18px; line-height:1.8'>• Internet connection and access to your device's webcam or front camera is required in order for the system to function as intended.<br><br> • You acknowledge and agree that any questions, comments, feedback and other information regarding the site provided by you to us are non-confidential and shall become our sole property. However, it shall be only used for educational purposes and data for future improvement of the system.<br><br> • The duration of your time spent using our system will be tracked and recorded. However, it shall be only used for educational purposes and data for future improvement of the system. </h3>", unsafe_allow_html=True)
                    st.write("")
                    
                    if "agree_state" not in st.session_state :
                        st.session_state.agree_state = False
                    
                    if st.button("I understand and agree" or st.session_state.agree_state):
                        st.session_state.agree_state = True
                        dt = datetime.now()
                        t1 = tz.gettz("Asia/Singapore")
                        current = dt.astimezone(t1)
                        login_start(username, current)
                        st.success("Please move on to the task.")
                    
                elif task == "Detect Emotion":
                    st.markdown("***")
                    st.markdown("<h2 style='text-align: center; color: white; font-size:40px'>Detect Your Emotion</h2>", unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: justify; color: white; font-size:20px; line-height: 1.8'>Feature:</h3>", unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: justify; color: white; font-weight:normal; font-size:18px; line-height:1.8'>• Launch your webcam and find out your real time emotional state.</h3>", unsafe_allow_html=True)
                    st.markdown("***")
                    st.markdown("<h3 style='text-align: justify; color: white; font-size:20px; line-height: 1.8'>Instruction:</h3>", unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: justify; color: white; font-weight:normal; font-size:18px; line-height:1.8'>1. Click the start button below to start your webcam. <br><br> 2. Click the allow button to allow access to your device's webcam or front camera. <br><br> 3. Wait for a few seconds for the system to load the livestream from your webcam. Please ensure a strong Internet connection on your device. <br><br>4. Once the livestream appeared, please face your frontal face towards your webcam. <br><br> 5. After 7 seconds, click the stop button to end the livestream. <br><br> 6. Click the Check Emotion button below to check your result.</h3>", unsafe_allow_html=True)
                    st.markdown("***")
                    st.markdown("<h3 style='text-align: justify; color: white; font-size:20px; line-height: 1.8'>Tips to troubleshoot:</h3>", unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: justify; color: white; font-weight:normal; font-size:18px; line-height:1.8'>• Ensure a strong Internet connection on your device to run the process smoothly. <br><br>• Face your frontal face towards the webcam on a distance of approximately 25cm away for a more accurate result.</h3>", unsafe_allow_html=True)
                    st.write("")
                    st.write("")
                    st.markdown("<h3 style='text-align: justify; color: white; font-size:20px; line-height: 1.8'>Click the start button below to begin:</h3>", unsafe_allow_html=True)
                    play_video()
                    st.write("")
    
                    my_bar = st.progress(0)
                    if st.button("Check Emotion"):
                        for percent_complete in range(100):
                            time.sleep(0.05)
                            my_bar.progress(percent_complete+1)
                        cursor = collection3.find({}, {'emotion': 1, '_id': 0})
                        emotions = []
                        for item in cursor:
                            if 'emotion' in item:
                                emotions.append(item['emotion'])
                        
                        emotions.reverse()
                        st.write("")
                        if (str(emotions[0])) == "Happy":
                            st.write ("Your current emotion is " + str(emotions[0]) + ", stay in high spirits! :laughing:")
                            st.success("Please move on to the task.")
                        elif (str(emotions[0])) == "Neutral" or (str(emotions[0])) == "Sad":
                            st.write ("Your current emotion is " + str(emotions[0]) + ", lets cheer up! :smile:")
                            st.success("Please move on to the task.")

                elif task == "Play Recommended Music":
                    st.markdown("***")
                    st.markdown("<h2 style='text-align: center; color: white; font-size:40px'>Play Recommended Music</h2>", unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: justify; color: white; font-size:20px; line-height: 1.8'>Feature:</h3>", unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: justify; color: white; font-weight:normal; font-size:18px; line-height:1.8'>• After detecting your real time emotional state, get a list of recommended songs to improve your mood.</h3>", unsafe_allow_html=True)
                    st.markdown("***")
                    st.markdown("<h3 style='text-align: justify; color: white; font-size:20px; line-height: 1.8'>Instruction:</h3>", unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: justify; color: white; font-weight:normal; font-size:18px; line-height:1.8'>1. Click on the button below to pick a genre of recommended songs you wish to listen. <br><br> 2. A list of 6 recommended songs will be displayed below. <br><br> 3. Click on the play button on one of the songs to listen to the preview of the song. <br><br> 4. Click on the song name to listen to the full track of the song. <br><br> 5. Click on Recommend More Songs button to get a new list of recommended song. </h3>", unsafe_allow_html=True)
                    st.markdown("***")
                    st.write("")
                    cursor = collection3.find({}, {'emotion': 1, '_id': 0})
                    emotions = []
                    for item in cursor:
                        if 'emotion' in item:
                            emotions.append(item['emotion'])

                    emotions.reverse()
                    st.markdown("<h3 style='text-align: justify; color: white; font-size:20px;'>Your emotional state: </h3>", unsafe_allow_html=True)
                    st.write("")
                    current_emotion = str(emotions[0])
                    
                    if current_emotion == "Happy":
                        st.write (current_emotion + ":smile:")
                        def load_data():
    
                            df = pd.read_csv(r"filtered_track_df.csv")
                            df['genres'] = df.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
                            exploded_track_df = df.explode("genres")
                            return exploded_track_df
                        
                        genre_names = ['pop', 'hip hop', 'alternative rock', 'electronic', 'r&b', 'rock', 'k-pop']
                        audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]

                        exploded_track_df = load_data()

                        def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):

                            genre = genre.lower()
                            genre_data = exploded_track_df[(exploded_track_df["genres"]==genre) & 
                                                           (exploded_track_df["release_year"]>=start_year) & 
                                                           (exploded_track_df["release_year"]<=end_year)]
                            genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]
                            neigh = NearestNeighbors()
                            neigh.fit(genre_data[audio_feats].to_numpy())
                            n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(genre_data), return_distance=False)[0]
                            uris = genre_data.iloc[n_neighbors]["uri"].tolist()
                            audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
                            return uris, audios
                        
                        st.write("")
                        st.markdown("<h3 style='text-align: justify; color: white; font-size:20px; line-height: 1.8'>Please choose the genre of songs you wish to listen to:</h3>", unsafe_allow_html=True)
                        with st.container():
                            col1, col2, col3, col4 = st.columns((2,0.5,0.5,0.5))
                            genre = st.radio(
                                "",
                                genre_names, index=genre_names.index("pop"))
                            with col1:
                                start_year, end_year = (2005,2022)
                                acousticness = 0.25
                                danceability = 0.8
                                energy = 0.7
                                instrumentalness = 0.5 
                                valence = 0.5 
                                tempo = 125.00
                            st.markdown("<h3 style='text-align: left; color: white; font-weight:normal; font-size:18px; line-height:1.8'> Enjoy your songs!</h3>", unsafe_allow_html=True)
                                
                        tracks_per_page = 6
                        test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
                        uris, audios = n_neighbors_uri_audio(genre, start_year, end_year, test_feat)
                        tracks = []

                        for uri in uris:
                            track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(uri)
                            tracks.append(track)
    
                        if 'previous_inputs' not in st.session_state:
                            st.session_state['previous_inputs'] = [genre, start_year, end_year] + test_feat
                                 
                        current_inputs = [genre, start_year, end_year] + test_feat

                        if current_inputs != st.session_state['previous_inputs']:
                            if 'start_track_i' in st.session_state:
                                st.session_state['start_track_i'] = 0
                            st.session_state['previous_inputs'] = current_inputs
    
                        if 'start_track_i' not in st.session_state:
                            st.session_state['start_track_i'] = 0
                                 
                        with st.container():
                            col1, col2, col3 = st.columns([2,1,2])
                            if st.button("Recommend More Songs"):
                                if st.session_state['start_track_i'] < len(tracks):
                                    st.session_state['start_track_i'] += tracks_per_page
                            current_tracks = tracks[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
                            current_audios = audios[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
                            if st.session_state['start_track_i'] < len(tracks):
                                for i, (track, audio) in enumerate(zip(current_tracks, current_audios)):
                                    if i%2==0:
                                        with col1:
                                            components.html(
                                                track,
                                                height=400,
                                            )
                                            with st.expander("See more details"):
                                                df = pd.DataFrame(dict(
                                                r=audio[:5],
                                                theta=audio_feats[:5]))
                                    else:
                                        with col3:
                                            components.html(
                                                track,
                                                height=400,
                                            )
                                            with st.expander("See more details"):
                                                df = pd.DataFrame(dict(
                                                    r=audio[:5],
                                                    theta=audio_feats[:5]))
                            else:
                                st.write("No songs left to recommend")

                    elif current_emotion == "Neutral" or current_emotion == "Sad":
                        st.write (current_emotion + ":cry:")
                        def load_data():
    
                            df = pd.read_csv(r"filtered_track2_df.csv")
                            df['genres'] = df.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
                            exploded_track_df = df.explode("genres")
                            return exploded_track_df
                        
                        genre_names = ['pop', 'r&b', 'hip hop', 'alternative rock', 'jazz', 'electronic', 'classical', 'k-pop']
                        audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]

                        exploded_track_df = load_data()
                        
                        def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):

                            genre = genre.lower()
                            genre_data = exploded_track_df[(exploded_track_df["genres"]==genre) & 
                                                           (exploded_track_df["release_year"]>=start_year) & 
                                                           (exploded_track_df["release_year"]<=end_year)]
                            genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]
                            neigh = NearestNeighbors()
                            neigh.fit(genre_data[audio_feats].to_numpy())
                            n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(genre_data), return_distance=False)[0]
                            uris = genre_data.iloc[n_neighbors]["uri"].tolist()
                            audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()
                            return uris, audios
                        
                        st.write("")
                        st.markdown("<h3 style='text-align: justify; color: white; font-size:20px; line-height: 1.8'>Please choose the genre of songs you wish to listen to:</h3>", unsafe_allow_html=True)
                        with st.container():
                            col1, col2, col3, col4 = st.columns((2,0.5,0.5,0.5))
                            genre = st.radio(
                                "",
                                genre_names, index=genre_names.index("pop"))
                            with col1:
                                start_year, end_year = (2000,2022)
                                acousticness = 0.6
                                danceability = 1.0
                                energy = 0.45
                                instrumentalness = 0.3
                                valence = 0.8  
                                tempo = 150.00
                            st.markdown("<h3 style='text-align: left; color: white; font-weight:normal; font-size:18px; line-height:1.8'> Enjoy your songs!</h3>", unsafe_allow_html=True)
                                
                        tracks_per_page = 6
                        test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
                        uris, audios = n_neighbors_uri_audio(genre, start_year, end_year, test_feat)
                        tracks = []
                        
                        for uri in uris:
                            track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(uri)
                            tracks.append(track)
    
                        if 'previous_inputs' not in st.session_state:
                            st.session_state['previous_inputs'] = [genre, start_year, end_year] + test_feat
                                 
                        current_inputs = [genre, start_year, end_year] + test_feat

                        if current_inputs != st.session_state['previous_inputs']:
                            if 'start_track_i' in st.session_state:
                                st.session_state['start_track_i'] = 0
                            st.session_state['previous_inputs'] = current_inputs
    
                        if 'start_track_i' not in st.session_state:
                            st.session_state['start_track_i'] = 0
                
                        with st.container():
                            col1, col2, col3 = st.columns([2,1,2])
                            if st.button("Recommend More Songs"):
                                if st.session_state['start_track_i'] < len(tracks):
                                    st.session_state['start_track_i'] += tracks_per_page
                            current_tracks = tracks[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
                            current_audios = audios[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
                            if st.session_state['start_track_i'] < len(tracks):
                                for i, (track, audio) in enumerate(zip(current_tracks, current_audios)):
                                    if i%2==0:
                                        with col1:
                                            components.html(
                                                track,
                                                height=400,
                                            )
                                            with st.expander("See more details"):
                                                df = pd.DataFrame(dict(
                                                r=audio[:5],
                                                theta=audio_feats[:5]))
                                    else:
                                        with col3:
                                            components.html(
                                                track,
                                                height=400,
                                            )
                                            with st.expander("See more details"):
                                                df = pd.DataFrame(dict(
                                                    r=audio[:5],
                                                    theta=audio_feats[:5]))
                            else:
                                st.write("No songs left to recommend")
                
                elif task == "Rate Your Experience":
                    st.markdown("***")
                    st.markdown("<h2 style='text-align: center; color: white; font-size:40px'>Rate Your Experience</h2>", unsafe_allow_html=True)
                    st.write("")
                    st.write("")
                    st.markdown("<h3 style='text-align: left; color: white; font-size:18px'>Please rate your experience using our system.</h3>", unsafe_allow_html=True)
                    number = st.slider("",0, 100, 50)
                    st.markdown("<h3 style='text-align: left; color: white; font-weight:normal; font-size:18px; line-height:1.8'> Unhappy = 0-49, Happy = 50-100</h3>", unsafe_allow_html=True)
                    st.text("")
                    submit = st.button("Submit")
                    if "submit_state" not in st.session_state :
                        st.session_state.submit_state = False
                        
                    if submit or st.session_state.submit_state: 
                        st.session_state.submit_state = True
                        if number >0 and number <=49:
                            st.markdown("<h3 style='text-align: left; color: white; font-weight:normal; font-size:18px; line-height:1.8'> Feedback submitted. Thank you for your feedback!</h3>", unsafe_allow_html=True)
                            st.markdown("***")
                            st.markdown("<h3 style='text-align: left; color: white; font-size:40px'>Usage Record</h3>", unsafe_allow_html=True)
                            st.write("")
                            st.write("This section allows you to check your usage record on our system. A graph is displayed to help analyze your usage.")
                            st.write("")
                            
                            dt2 = datetime.now()
                            t2 = tz.gettz("Asia/Singapore")
                            current2 = dt2.astimezone(t2)
                            end = timer()
                            usage = end-start
                            logout_start(username, current2, number, usage)

                            
                            ##display login time##
                            cursor = collection2.find({}, {'start time': 1, '_id': 0})
                            start_time = []
                            for item in cursor:
                                if 'start time' in item:
                                    start_time.append(item['start time'])

                            start_time.reverse()                            
                            e = str(start_time[0])
                            st.write("Your login time is " + datetime.strptime(e, '%Y-%m-%d %H:%M:%S.%f').strftime('%I:%M:%S %p') +" on " 
                                     + datetime.strptime(e, '%Y-%m-%d %H:%M:%S.%f').strftime('%d/%m/%Y'))
                            
                            ##display logout time##
                            cursor2 = collection2.find({}, {'end time': 1, '_id': 0})
                            end_time = []
                            for item in cursor2:
                                if 'end time' in item:
                                    end_time.append(item['end time'])

                            end_time.reverse()
                            y = str(end_time[0])
                            st.write("Your logout time is " + datetime.strptime(y, '%Y-%m-%d %H:%M:%S.%f').strftime('%I:%M:%S %p') +" on " 
                                     + datetime.strptime(y, '%Y-%m-%d %H:%M:%S.%f').strftime('%d/%m/%Y'))
                            
                            ##display usage duration##
                            hours, seconds = divmod(usage * 60, 3600)
                            minutes, seconds = divmod(seconds, 60)
                            result = "{:02.0f} minutes {:02.0f} seconds.".format(hours, minutes)
                            st.write("You have used our system for " + (str(result)))
    
                            ##display login frequency##
                            cursor3 = collection2.find({"name": username}, {'name': 1, '_id': 0})
                            login_frequency= []
                            for item in cursor3:
                                if 'name' in item:
                                    login_frequency.append(item['name'])

                            freq = int(len(login_frequency))
                            st.write("You have used our system for " + (str((freq)/2)) + " times for the past 2 weeks.")
                            
                            ##display experience##
                            st.write("You rated our system as " + str(number) + "/100, you are unhappy. :cry:")
                            st.write("")
                            
                            ##display graph##
                            cursor4 = collection2.find({"name": username}, {'experience': 1, '_id': 0})
                            experience = []
                            for item in cursor4:
                                if 'experience' in item:
                                    experience.append(item['experience'])
                            
                            experience.reverse()
                            experience.pop(0)
                            
                            cursor5 = collection2.find({"name": username}, {'usage': 1, '_id': 0})
                            usage = []
                            for item in cursor5:
                                if 'usage' in item:
                                    usage.append(item['usage'])
                                    
                            usage.reverse()
                            usage.pop(0)
                            usage_seconds = [int(usage) for usage in usage]
                            usage_minutes = [round(x,2) for x in [x/60 for x in usage_seconds]]
                                                       
                            st.write("Graph information: ")
                            st.write("X-axis = Experience ( Unhappy = 0-49, Happy = 50-100 )")
                            st.write("Y-axis = Duration used in minutes")
                            chart_data = pd.DataFrame(usage_minutes, experience, columns =["usage duration in minutes"])
                            st.bar_chart(chart_data)
                            
                            st.write("We would investigate your feedback and improve our system further. Thank you for using our system! :heart_eyes:")
                        
                        elif number >=50:
                            st.markdown("<h3 style='text-align: left; color: white; font-weight:normal; font-size:18px; line-height:1.8'> Feedback submitted. Thank you for your feedback!</h3>", unsafe_allow_html=True)
                            st.markdown("***")
                            st.markdown("<h3 style='text-align: left; color: white; font-size:40px'>Usage Record</h3>", unsafe_allow_html=True)
                            st.write("")
                            st.write("This section allows you to check your usage record on our system. A graph is displayed to help analyze your usage.")
                            st.write("")
                            
                            dt2 = datetime.now()
                            t2 = tz.gettz("Asia/Singapore")
                            current2 = dt2.astimezone(t2)
                            end = timer()
                            usage = end-start
                            logout_start(username, current2, number, usage)
                            
                            ##display login time##
                            cursor = collection2.find({}, {'start time': 1, '_id': 0})
                            start_time = []
                            for item in cursor:
                                if 'start time' in item:
                                    start_time.append(item['start time'])

                            start_time.reverse()
                            e = str(start_time[0])
                            st.write("Your login time is " + datetime.strptime(e, '%Y-%m-%d %H:%M:%S.%f').strftime('%I:%M:%S %p') +" on " 
                                     + datetime.strptime(e, '%Y-%m-%d %H:%M:%S.%f').strftime('%d/%m/%Y'))
                            
                            ##display logout time##
                            cursor2 = collection2.find({}, {'end time': 1, '_id': 0})
                            end_time = []
                            for item in cursor2:
                                if 'end time' in item:
                                    end_time.append(item['end time'])

                            end_time.reverse()
                            y = str(end_time[0])
                            st.write("Your logout time is " + datetime.strptime(y, '%Y-%m-%d %H:%M:%S.%f').strftime('%I:%M:%S %p') +" on " 
                                     + datetime.strptime(y, '%Y-%m-%d %H:%M:%S.%f').strftime('%d/%m/%Y'))
                            
                            ##display usage duration##
                            hours, seconds = divmod(usage * 60, 3600)
                            minutes, seconds = divmod(seconds, 60)
                            result = "{:02.0f} minutes {:02.0f} seconds.".format(hours, minutes)
                            st.write("You have used our system for " + (str(result)))
                                
                            ##display login frequency##
                            cursor3 = collection2.find({"name": username}, {'name': 1, '_id': 0})
                            login_frequency= []
                            for item in cursor3:
                                if 'name' in item:
                                    login_frequency.append(item['name'])

                            freq = int(len(login_frequency))
                            st.write("You have used our system for " + (str((freq)/2)) + " times for the past 2 weeks.")
    
                            ##display experience##
                            st.write("You rated our system as " + str(number) + "/100, you are happy! :smile:")
                            st.write("")
                
                            ##display graph##
                            cursor4 = collection2.find({"name": username}, {'experience': 1, '_id': 0})
                            experience = []
                            for item in cursor4:
                                if 'experience' in item:
                                    experience.append(item['experience'])
                            
                            experience.reverse()
                            experience.pop(0)
                            
                            cursor5 = collection2.find({"name": username}, {'usage': 1, '_id': 0})
                            usage = []
                            for item in cursor5:
                                if 'usage' in item:
                                    usage.append(item['usage'])
                                    
                            usage.reverse()
                            usage.pop(0)
                            usage_seconds = [int(usage) for usage in usage]
                            usage_minutes = [round(x,2) for x in [x/60 for x in usage_seconds]]
                            
                            st.write("Graph information: ")
                            st.write("X-axis = Experience ( Unhappy = 0-49, Happy = 50-100 )")
                            st.write("Y-axis = Duration used in minutes")
                            chart_data = pd.DataFrame(usage_minutes, experience, columns =["usage duration in minutes"])
                            st.bar_chart(chart_data)
                            
                            st.write("We appreciate your feedback and hope to provide pleasant experience to you always. Thank you for using our system! :heart_eyes:")

            else:
                st.write("")
                st.warning("Incorrect Username or Password, please try again.")
        
    elif choice == "Register":
        st.write("")
        st.markdown("<h2 style='text-align: left; color: white; font-size:40px'>Create New Account</h2>", unsafe_allow_html=True)
        st.write("")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password", type='password')
        
        if st.button("Register"):
            if len(new_user)>0 and len(new_password)>0:
                add_userdata(new_user,new_password)
                st.success("You have successfully created an account.")
                st.info("Use the navigation sidebar to explore the system.")
            else:
                st.error('Please fill up all the required fields')
                
if __name__ == '__main__':
    main()
    
        
        
        
        
        
        
        
        
        
        