import streamlit as st
import io
import os

st.title("VR-Conferencing")


video_or_webcam = st.text_input("Please enter 1 for video and 2 for webcam:", 2)

video_resolution_scale = st.text_input("Please enter the scale with which you want to increase the resolution of the video:", 2)

uploaded_video = None

if(video_or_webcam=='1'):
    uploaded_video = st.file_uploader("Choose a video:", type=["mp4"])

uploaded_background = st.file_uploader("Choose a background:", type=["jpg","jpeg","png"])

temporary_video = False
temporary_background = False



if uploaded_background is not None:
    if(video_or_webcam=='1'):
        if uploaded_video is not None:

            onscreen = st.empty()
            onscreen.text('Processing video...')

            g = io.BytesIO(uploaded_video.read())  ## BytesIO Object

            temporary_video = "temp.mp4"
            
            with open(temporary_video, 'wb') as out:  ## Open temporary file as bytes
                out.write(g.read())  ## Read bytes into file

            # close file
            out.close()
        
            g = io.BytesIO(uploaded_background.read())  ## BytesIO Object

            temporary_background = "temp_background.jpg"

            with open(temporary_background, 'wb') as out:  ## Open temporary file as bytes
                out.write(g.read())  ## Read bytes into file

            # close file
            out.close()

            if(video_resolution_scale=='1'):
                command =  "python . temp.mp4 -o output.mp4 -b temp_background.jpg -z model"
                process = os.system(command)
            else:
                command =  "python . temp.mp4 -o output.mp4 -b temp_background.jpg -z model -s " +video_resolution_scale+ ""
                process = os.system(command)

            command = "python adding_audio.py temp.mp4 output.mp4 final.mp4"
            process = os.system(command)

            video_file = open('final.mp4','rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

            os.remove("temp.mp4")
            os.remove("output.mp4")
            os.remove("temp_background.jpg")

            onscreen.empty()
            onscreen.text('Done!')

    else:

            g = io.BytesIO(uploaded_background.read())  ## BytesIO Object

            temporary_background = "temp_background.jpg"

            with open(temporary_background, 'wb') as out:  ## Open temporary file as bytes
                out.write(g.read())  ## Read bytes into file

            # close file
            out.close()

            if(video_resolution_scale=='1'):
                command =  "python . 0 -b temp_background.jpg -z model"
                process = os.system(command)
            else:
                command =  "python . 0 -b temp_background.jpg -z model -s " +video_resolution_scale+""
                process = os.system(command)
            os.remove("temp_background.jpg")

