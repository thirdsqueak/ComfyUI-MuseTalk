import os
import cv2
import numpy as np
import torch

import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)
diffusers_path = folder_paths.get_folder_paths("diffusers")[0]

MuseVCheckPointDir = os.path.join(
    diffusers_path, "TMElyralab/MuseTalk"
)

#ffmpeg_path = os.getenv('FFMPEG_PATH')
#if ffmpeg_path is None:
#    print("please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
#elif ffmpeg_path not in os.getenv('PATH'):
#    print("add ffmpeg to path")
#    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"

    
from musetalk.whisper.audio2feature import Audio2Feature
from musetalk.models.vae import VAE
from musetalk.models.unet import UNet,PositionalEncoding

# def load_all_model():
#     audio_processor = Audio2Feature(model_path=f"{MuseVCheckPointDir}/whisper/tiny.pt")
#     vae = VAE(model_path = f"{MuseVCheckPointDir}/sd-vae-ft-mse/")
#     unet = UNet(unet_config=f"{MuseVCheckPointDir}/musetalk/musetalk.json",
#                 model_path =f"{MuseVCheckPointDir}/musetalk/pytorch_model.bin")
#     pe = PositionalEncoding(d_model=384)
#     return audio_processor,vae,unet,pe

def load_all_model(device=None, dtype=torch.float32):
    if device is None:
        device = torch.device('cpu')  #'cuda' if torch.cuda.is_available() else 
    if device == 'cuda':
        try:
            dtype = torch.float8_e4m3fn
            print("Using float8_e4m3fn")
        except AttributeError:
            print("float8 not supported set bfloat16.")
        dtype = torch.bfloat16
    else:
        try:
            dtype = torch.float16
            print("Using float16")
        except AttributeError:
            print("float16 not supported set float32")
            dtype = torch.float32
        
    audio_processor = Audio2Feature(model_path=f"{MuseVCheckPointDir}/whisper/tiny.pt")

    vae = VAE(model_path=f"{MuseVCheckPointDir}/sd-vae-ft-mse/")
    unet = UNet(
        unet_config=f"{MuseVCheckPointDir}/musetalk/musetalk.json",
        model_path=f"{MuseVCheckPointDir}/musetalk/pytorch_model.bin"
    )
    pe = PositionalEncoding(d_model=384)  # 384  

    vae.vae = vae.vae.to(device=device, dtype=dtype)
    unet.model = unet.model.to(device=device, dtype=dtype)
    pe = pe.to(device=device, dtype=dtype)

    print(f"[vae] dtype: {next(vae.vae.parameters()).dtype}")
    print(f"[unet] dtype: {next(unet.model.parameters()).dtype}")
    # print(f"[pe] dtype: {next(pe.parameters()).dtype if hasattr(pe, 'parameters') else pe.dtype}")
    if hasattr(pe, "parameters") and any(True for _ in pe.parameters()):
        print(f"[pe] dtype: {next(pe.parameters()).dtype}")
    else:
        print(f"[pe] dtype: {getattr(pe, 'dtype', 'no dtype attr')}")
    
    print(device.type)    # cpu
    return audio_processor, vae, unet, pe


def get_file_type(video_path):
    _, ext = os.path.splitext(video_path)

    if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        return 'image'
    elif ext.lower() in ['.avi', '.mp4', '.mov', '.flv', '.mkv']:
        return 'video'
    else:
        return 'unsupported'

def get_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

def datagen(whisper_chunks,vae_encode_latents,batch_size=8,delay_frame = 0):
    whisper_batch, latent_batch = [], []
    for i, w in enumerate(whisper_chunks):
        idx = (i+delay_frame)%len(vae_encode_latents)
        latent = vae_encode_latents[idx]
        whisper_batch.append(w)
        latent_batch.append(latent)

        if len(latent_batch) >= batch_size:
            whisper_batch = np.asarray(whisper_batch)
            latent_batch = torch.cat(latent_batch, dim=0)
            yield whisper_batch, latent_batch
            whisper_batch, latent_batch = [], []

    # the last batch may smaller than batch size
    if len(latent_batch) > 0:
        whisper_batch = np.asarray(whisper_batch)
        latent_batch = torch.cat(latent_batch, dim=0)

        yield whisper_batch, latent_batch
        
# def unload_model(vae=None, unet=None, pe=None):
#     del vae
#     del unet
#     del pe
#     torch.cuda.empty_cache()
#     torch.cuda.ipc_collect()
