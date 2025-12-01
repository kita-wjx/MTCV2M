import torchaudio
import moviepy.editor as mp
from pydub import AudioSegment
import os
import torch
import torch.nn.functional as F
import argparse

from vm2m.models import vmmusicgen
from vm2m.data.audio import audio_write

def main():
    parser = argparse.ArgumentParser(description="Video to Music Generation")
    parser.add_argument('--model_path', type=str, default='./checkpoints/stage1_medium', help='Path to the pretrained model')
    parser.add_argument('--video_dir', type=str, default='./features/egs/video', help='Directory containing input videos')
    parser.add_argument('--video_fea', type=str, default="./features/egs/clipcls", help='Directory containing clip features')
    parser.add_argument('--image_fea', type=str, default="./features/egs/image_features", help='Directory containing image features')
    parser.add_argument('--mae_dir', type=str, default="./features/egs/videomae", help='Directory containing videomae features')
    parser.add_argument('--syn_path', type=str, default='./features/egs/music', help='Directory to save generated music and videos')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = args.model_path
    video_dir = args.video_dir
    video_fea = args.video_fea
    image_fea = args.image_fea
    mae_dir = args.mae_dir

    syn_path = args.syn_path

    model = vmmusicgen.MusicGen.get_pretrained(model_path, device=device)

    for file in sorted(os.listdir(video_dir)):
        if file.endswith('.mp4'):
            video_path = os.path.join(video_dir, file)
            video_name = file.split('.')[0]
            if os.path.exists(os.path.join(syn_path, video_name+'.wav')):
                continue
            cls_embed = os.path.join(video_fea, video_name+'.pt')
            cls = torch.load(cls_embed).to(device)
            image_embed = os.path.join(image_fea, video_name+'.pt')
            image = torch.load(image_embed).to(device)
            mae_embed = os.path.join(mae_dir, video_name+'.pt')
            mae = torch.load(mae_embed).to(device)
            model.set_generation_params(
                duration=cls.shape[0]
            )

            res = model.generate_with_video(cls = cls, image = image, mae = mae)

            for idx, one_wav in enumerate(res):
                # Will save under filename.wav, with loudness normalization at -14 db LUFS.
                wav_path = os.path.join(syn_path, video_name+'.wav')
                wav_name = os.path.join(syn_path, video_name)
                audio_write(f'{wav_name}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
                video_mp = mp.VideoFileClip(str(video_path))
                audio_clip = AudioSegment.from_wav(wav_path)
                audio_clip[0:int(video_mp.duration*1000)].export(wav_path)
                # Render generated music into input video
                audio_mp = mp.AudioFileClip(wav_path)

                audio_mp = audio_mp.subclip(0, video_mp.duration)
                final = video_mp.set_audio(audio_mp)
                try:
                    final.write_videofile(os.path.join(syn_path, video_name+'.mp4'),
                        codec='libx264', 
                        audio_codec='aac', 
                        temp_audiofile='temp-audio.m4a',
                        remove_temp=True
                    )
                except Exception as e:
                    print(f"Error: {e}")

if __name__ == "__main__":
    main()