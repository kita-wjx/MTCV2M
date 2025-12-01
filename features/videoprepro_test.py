from models.clip import ViTload
import os
import torch
from models.mae import vit_large_patch16_224
from utils import capture_video
from moviepy.editor import VideoFileClip
import argparse

def main():
    parser = argparse.ArgumentParser(description="Video Preprocessing")
    parser.add_argument('--video_dir', type=str, default="./egs/video", help='Directory containing input videos')
    parser.add_argument('--video_fea', type=str, default="./egs/clipcls", help='Directory to save clip features')
    parser.add_argument('--image_fea', type=str, default="./egs/image_features", help='Directory to save image features')
    parser.add_argument('--mae_output_dir', type=str, default="./egs/videomae", help='Directory to save videomae features')
    parser.add_argument('--mae_model_path', type=str, default="../checkpoints/VideoMAELv2/state_dict.bin", help='Path to the VideoMAE model checkpoint')

    args = parser.parse_args()

    # config
    video_dir = args.video_dir
    video_fea = args.video_fea
    image_fea = args.image_fea
    mae_output_dir = args.mae_output_dir

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dir_map = sorted(os.listdir(video_dir))

    # clip
    clip_name = "ViT-L/14@336px"
    clip_model, _ = ViTload(clip_name)
    # videomaev2
    mae_model_path = args.mae_model_path
    state_dict = torch.load(mae_model_path)['state_dict']
    mae_model = vit_large_patch16_224(num_classes=0)
    mae_model.load_state_dict(state_dict)
    mae_model = mae_model.to(device)

    for file in dir_map:
        if os.path.exists(os.path.join(video_fea, file[:-4] + '.pt')):
            continue
        video_path = os.path.join(video_dir, file)
        
        # duration
        video_clip = VideoFileClip(video_path)
        duration = int(video_clip.duration)
        
        video_clip = capture_video(video_path, 'clip', 336, 1, device, duration)
        video_mae = capture_video(video_path, 'videomae', 224, 2, device, duration)
        video_mae = video_mae.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
        
        with torch.no_grad():
            # clip
            clip_embeds, clip_embeds_patch = clip_model(video_clip.to(torch.float16), need_detail=True)
            # videomaev2
            segments = duration // 8
            if segments == 0:
                B, C, T, H, W = video_mae.shape
                padding_size = 16 - T
                padding_val = video_mae[:,:,-1:].expand(-1,-1,padding_size,-1,-1)
                video_mae = torch.cat([video_mae, padding_val], dim=2)
                mae_embed = mae_model.forward_features(video_mae, need_pool=False)
                mae_embed = mae_embed.reshape(8, 196, -1)
                mae_embed = mae_embed[0:duration, :, :]
                mae_embed = mae_embed.mean(dim=1)
            else:
                B, C, T, H, W = video_mae.shape
                segments = (T - 2) // 14
                mae_embed = mae_model.forward_features(video_mae[:,:,0:16], need_pool=False)
                for i in range(1, segments):
                    tmp_embed = mae_model.forward_features(video_mae[:,:,i*14:(i+1)*14+2], need_pool=False)
                    mae_embed = torch.cat([mae_embed, tmp_embed[:,196:]], dim=1)
                if (T - 2) % 14 != 0:
                    video_mae = video_mae[:,:,T-16:]
                    tmp_embed = mae_model.forward_features(video_mae, need_pool=False)
                    S = (((T - 2) % 14) // 2) * 196
                    mae_embed = torch.cat([mae_embed, tmp_embed[:,-S:]], dim=1)
                mae_embed = mae_embed.reshape(duration, 196, -1)
                mae_embed = mae_embed.mean(dim=1)
            
            torch.save(clip_embeds, os.path.join(video_fea, file[:-4] + '.pt'))
            torch.save(clip_embeds_patch, os.path.join(image_fea, file[:-4] + '.pt'))
            torch.save(mae_embed, os.path.join(mae_output_dir, file[:-4] + '.pt'))

            print("{} saved".format(file[:-4]))

if __name__ == "__main__":
    main()
