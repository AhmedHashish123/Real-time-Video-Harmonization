from moviepy.editor import *
import subprocess as sp
import os
import sys

if __name__ == '__main__':
    input_video_source=sys.argv[1]
    input_video_dest=sys.argv[2]
    audio_file="audio.mp3"
    output_video=sys.argv[3]

    ###############################Extracting the audio and saving it as a .mp3 file###############################
    source_video = VideoFileClip(r""+input_video_source+"")
    source_video.audio.write_audiofile(r""+audio_file+"")
    source_video.close()

    ##############################Creating the command to combine the frames with the audio and form a new video###############################
    command = ['ffmpeg',
            '-y', #approve output file overwite
            '-i', input_video_dest,
            '-i', audio_file,
            '-c:v', 'libx264',
            '-c:a', 'aac', #to convert mp3 to aac 
            output_video]

    process = sp.Popen(command)
    process.wait()

    ##############################Deleting the audio file###############################
    os.remove(""+audio_file+"")


    