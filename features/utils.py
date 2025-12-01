import cv2
from PIL import Image
import torch

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
    
try:
    from torchvision.transforms import InterpolationMode
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BILINEAR = Image.BILINEAR

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
def _transform2(n_px):
    return Compose([
        Resize(n_px, interpolation=BILINEAR),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def _transform_vidmuse(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
    ])


def capture_video(video_path, type = 'clip', n_px = 336, frame_per_second=5, device='cpu', duration = 30):
    '''return torch.Tensor'''
    if video_path.endswith('.'):
        video_path = video_path[:-1]
    videocapture = cv2.VideoCapture(video_path)
    fps = int(videocapture.get(cv2.CAP_PROP_FPS))
    total_slice = int(videocapture.get(cv2.CAP_PROP_FRAME_COUNT))
    cnt = 0

    if frame_per_second > fps:
        # frame_per_second = fps
        raise(ValueError("frame_per_second > fps:{}".format(fps)))
        
    videos = []	
    
    if videocapture.isOpened():
        for i in range(int(total_slice)):
            success, img = videocapture.read()
            # cv2.imwrite('img.png', img)
            if i % int(fps / frame_per_second) == 0:
                if cnt == duration * frame_per_second:
                    break
                img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                # img.save('img.png')
                if type == 'clip':
                    img = _transform(n_px)(img)
                elif type == 'videomae':
                    img = _transform2(n_px)(img)
                else:
                    raise(ValueError("type error"))
                videos.append(img.tolist())
                cnt += 1
    
    return torch.Tensor(videos).to(device)

def capture_video_vidmuse(video_path, type = 'clip', n_px = 336, frame_per_second=5, device='cpu', duration = 10):
    '''return torch.Tensor'''
    if video_path.endswith('.'):
        video_path = video_path[:-1]
    videocapture = cv2.VideoCapture(video_path)
    fps = int(videocapture.get(cv2.CAP_PROP_FPS))
    total_slice = int(videocapture.get(cv2.CAP_PROP_FRAME_COUNT))
    cnt = 0

    if frame_per_second > fps:
        # frame_per_second = fps
        raise(ValueError("frame_per_second > fps:{}".format(fps)))
        
    videos = []	
    
    if videocapture.isOpened():
        for i in range(int(total_slice)):
            success, img = videocapture.read()
            # cv2.imwrite('img.png', img)
            if i % int(fps / frame_per_second) == 0:
                if cnt == duration * frame_per_second:
                    break
                img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                img = _transform_vidmuse(n_px)(img)
                videos.append(img.tolist())
                cnt += 1
    
    return torch.Tensor(videos).to(device)
