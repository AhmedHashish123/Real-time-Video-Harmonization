# Real-time-Video-Harmonization
This project segments the user from the foreground and allows him to add a different background and harmonize him with set background. All of this is done in real-time.

## Pipeline:
1. Human matting is made to remove the human subject (foreground) from the background.
2. The human foregrounnd subject is then added to the selected background.
3. The foreground and background are then harmonized which means that the human will not feel out of place from the background.
4. The output is post-processed and super resolved.

## Required Models:

Please download the following models in order for the code to work:

* [RVM](https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth) 
* [RainNet](https://drive.google.com/drive/folders/1NMvHbnD1kW-j1KKMxEb9R9IR5drMK3GQ?usp=sharing). Download "net_G_last.pth".

## How To Run:

* Through a web app (Recommended but requires Streamlit):

  streamlit run web_app.py
  
  You can select whether you want to use a web camera or a video. You will also be required to upload a background a choose the resolution scale.

* Through cli:
  * Web Camera:
  
    python . 0 -b Backgrounds/background6.jpg -z model -s 2 -p 300 0.5  (Harmonization, Super resolution & Post-processing)
    
    python . 0 -x 720 1280 -b Backgrounds/background6.jpg -s 2 (Super resolution)
    
  * Video:
  
    python . Videos/video.mp4 -o Videos/output.mp4 -x 720 1280 -b Backgrounds/background2.jpg -z model -s 2
    
* Parameters:
    
  * -x Input resolution. If specifid, the input stream will be clipped to the specified resolution. The input is then resized to 512 X 512.
  * -o Output file; if unspecified, the output will be displayed in a window. (Output video will not have audio. The webapp version's output video will have audio).
  * -b The background image.
  * -z Harmonization method:
    * "color" which will harmonize the human with the background using image processing which is faster but produces worse results than the model. (This option is only available when web camera is selected).
    * "model" which will harmoinze the human with the background using RainNet which is slower but produces better results than the image processing method.
  * -s Super resolution scale: 2,3,4,8.
  * -p Postprocessing window size and intensity; helps with flickering. If not sure, put it at 300 0.5
