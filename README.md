# Music-Recommendation-System-via-Facial-Expression-Recognition
The Music Recommendation System via Facial Expression Recognition is a system that allows users to identify their real time emotional state by using a webcam installed on the user’s device. After the emotional state is known to the user, the user is able to select their choice of recommended songs from a generated list of recommended music that is intended to cheer up the mood of the user based on the user’s emotional state. After choosing a song, the user is able to play and enjoy the song track in order to improve their mood. After finishing experiencing the system, the user is also able to rate their experience while using the system. This allows users to be able to view an analytic report that is analyzed by the system based on their past usage records on every consequent visit to the system. The analytic report aims to help users to better understand whether if the system is actually improving the user’s mood over time. Besides, a cloud database is used as it can store the user’s usage record in real time. <br /> 
Website link: https://justintoh18-music-recommendation-system-via-facial-e-app-hr2i2s.streamlit.app/

# Emotion Detection
The emotion detection model will predict one of the emotion among 3 emotions listed below:
- Happy
- Neutral
- Sad

# Music Recommendation Model
Every songs in the main dataset in filtered_track_df.csv and filtered_track2_df.csv are predicted into the one of the song category among 2 category listed below:
- Highly energetic
- Happy and calm

# Main Project Files
Our main project execution file is app.py file. The main project file for emotion detection model is EmotionDetectionModel.ipynb and the main project file for music dataset preprocessing is Preprocess Happy Music Dataset.ipynb and Preprocess Neutral&Sad Music Dataset.ipynb. <br /> 
Facial Expression Recognition Dataset: https://www.kaggle.com/datasets/msambare/fer2013 (FER-2013) <br />
Spotify Dataset Link: https://www.kaggle.com/datasets/saurabhshahane/spotgen-music-dataset (Spotify and Genius Track Dataset)
