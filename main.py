from time import time
import cv2
import torch
from torch.nn.functional import conv2d, interpolate
import torch.backends.cudnn as cudnn
from torchvision.transforms.functional import normalize
from kornia.color import hsv_to_rgb, rgb_to_hsv, ycbcr_to_rgb, rgb_to_ycbcr
from models import get_matting_model, get_harmonization_model, get_super_resolution_model
from harmonization import Palette
from utils import read_image, postprocessing_weights, to_torch, to_numpy, compose_tensors
import numpy as np

def process_stream(input_stream, input_resolution, output_stream,
                   background_path, harmonization,
                   super_resolution_factor, postprocessing):
    """The main stream processing function"""
    # Use GPU if available; if not, use CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Turning auto gradient off to save memory and time
    torch.set_grad_enabled(False)

    # Loading the input data/streams
    video = cv2.VideoCapture(input_stream)
    
    # Input resolution
    if input_resolution:
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, input_resolution[0])
        video.set(cv2.CAP_PROP_FRAME_WIDTH, input_resolution[1])
    else:
        input_resolution = (int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH)))
    
    # Preparing the output stream
    output_resolution = tuple((super_resolution_factor or 1) * i for i in reversed((512, 512) if background_path else input_resolution))
    if output_stream:
        output = cv2.VideoWriter(output_stream, cv2.VideoWriter_fourcc(*'mp4v'), video.get(cv2.CAP_PROP_FPS), output_resolution)

    # Preparing the matting model and backgrounds
    if background_path:
        input_resolution = (512, 910)  # Discard the old input resolution, as this must be used in frame resizing for the matthing model to work
        matting_model = get_matting_model(device)
        initial_background = to_torch(read_image(background_path), (512, 512), device)
        final_background = to_torch(read_image(background_path), output_resolution, device)
        # Define RVM model parameters
        recurrent_state = [None] * 4
        downsample_ratio = 0.5
    
        # Preparing the harmonization model
        if harmonization == 'model':
            harmonization_model = get_harmonization_model(device)
        elif harmonization == 'color':
            harmonization_palette = None

        # Preparing the post-processing window and weights
        if postprocessing:
            weights = postprocessing_weights(postprocessing[0], postprocessing[1], device, 3, 3)
            postprocessing_window = torch.empty(3, postprocessing[0], *((512, 512)), device=device)

    # Preparing the super resolution model
    if super_resolution_factor:
        super_resolution_model = get_super_resolution_model(super_resolution_factor, device)

    cudnn.benchmark = True  # NOTE Do NOT remove this! The super-resolution model gets mad at you if you do!

    while video.isOpened():
        start_time = time()
        
        # Preparing the frame
        ret, frame = video.read()
        if not ret: break
        frame = to_torch(frame, input_resolution, device)

        # Matting
        if background_path:
            frame = frame[..., 120:792]
            foreground, alpha, *recurrent_state = matting_model(frame, *recurrent_state, downsample_ratio)
            
            # Compose final frame (512, 512)
            frame = frame[..., 80:592]
            alpha = alpha[..., 80:592]
            foreground = foreground[..., 80:592]
            frame = foreground * alpha + initial_background * (1 - alpha)

            # Harmonization
            if harmonization == 'model':
                frame = normalize(frame, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                frame = (harmonization_model.processImage(frame, alpha, None) + 1.0) / 2.0
            elif harmonization == 'color':
                if harmonization_palette is None:  # First time here? Prepare a color palette from the first matted frame and the new background
                    harmonization_palette = Palette.extract(rgb_to_hsv(frame.squeeze()))
                hsv = rgb_to_hsv(frame.squeeze())
                harmonization_palette.apply(hsv)
                frame = hsv_to_rgb(hsv).unsqueeze(0)

            # Post-processing
            if postprocessing:
                postprocessing_window = postprocessing_window.roll(1, 1)
                postprocessing_window[:, 0, ...] = frame
                frame = conv2d(postprocessing_window, weights, padding='same').squeeze().unsqueeze(0)

        # Super resolution
        if super_resolution_factor:
            ycbcr = rgb_to_ycbcr(frame).unsqueeze(0)
            y, cb, cr = (ycbcr[:,:, i, ...] for i in range(3))
            y = super_resolution_model(y).to(torch.float32).squeeze().clip(0, 1)
            frame = ycbcr_to_rgb(torch.stack((
                    y,
                    interpolate(cb, y.shape, mode='bicubic').squeeze(),
                    interpolate(cr, y.shape, mode='bicubic').squeeze()))).unsqueeze(0)
            if matting_model: alpha = interpolate(alpha, scale_factor=super_resolution_factor)

        # Final composition
        if background_path:
            frame = compose_tensors(alpha, frame, final_background)

        # Writing the output
        frame = to_numpy(frame)
        if output_stream:
            output.write(frame)
        else:
            frame = cv2.medianBlur(frame,5)
            # frame = frame/np.max(frame)
            # cv2.imshow('VR Conferencing', cv2.resize(cv2.flip(frame, 1),(400,400)))
            cv2.imshow('VR Conferencing', cv2.flip(frame, 1))
            if cv2.pollKey() == ord('q'): break  # Quit with the Q key
        print(f'FPS: {round(1 / (time() - start_time)):02d}', end='\r')

    # Releasing the resources and flushing the output
    video.release()
    if output_stream: output.release()
    else: cv2.destroyAllWindows()
