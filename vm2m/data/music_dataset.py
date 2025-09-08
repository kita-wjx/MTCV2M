# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Modified by https://github.com/YatingMusic/MusiConGen/blob/main/audiocraft/audiocraft/data/music_dataset.py
"""Dataset of music tracks with rich metadata.
"""
from dataclasses import dataclass, fields
import json
import logging
from pathlib import Path
import typing as tp

import torch
import torch.nn.functional as F
import numpy as np

from .info_audio_dataset import (
    InfoAudioDataset,
    AudioInfo
)
from ..models.conditioners import (
    ConditioningAttributes,
    ClsCondition,
    FineGrainCondition,
    WavCondition,
    BeatCondition,
    IntensityCondition,
    MelodyCondition,
    VACondition
)

logger = logging.getLogger(__name__)


@dataclass
class MusicInfo(AudioInfo):
    """Segment info augmented with music metadata.
    """
    # music-specific metadata
    visual_content: tp.Optional[ClsCondition] = None
    finegrained_content: tp.Optional[FineGrainCondition] = None
    name: tp.Optional[str] = None
    intensity: tp.Optional[IntensityCondition] = None
    beat: tp.Optional[BeatCondition] = None
    melody: tp.Optional[MelodyCondition] = None
    va: tp.Optional[VACondition] = None
    # original wav accompanying the metadata
    self_wav: tp.Optional[WavCondition] = None
    
    @property
    def has_music_meta(self) -> bool:
        return self.name is not None

    def to_condition_attributes(self) -> ConditioningAttributes:
        out = ConditioningAttributes()
        for _field in fields(self):
            key, value = _field.name, getattr(self, _field.name)
            if key == 'self_wav':
                out.wav[key] = value
            elif key == 'beat':
                out.beat[key] = value
            elif key == 'intensity':
                out.intensity[key] = value
            elif key == 'melody':
                out.melody[key] = value
            elif key == 'va':
                out.va[key] = value
            elif key == 'visual_content':
                out.video[key] = value
            elif key == 'finegrained_content':
                out.finegrained_content[key] = value
        return out

    @classmethod
    def from_dict(cls, dictionary: dict, fields_required: bool = False):
        _dictionary: tp.Dict[str, tp.Any] = {}

        # allow a subset of attributes to not be loaded from the dictionary
        # these attributes may be populated later
        post_init_attributes = ['self_wav', 'beat', 'intensity', 'visual_content', 'finegrained_content', 'melody', 'va']
        optional_fields = ['keywords']

        for _field in fields(cls):
            if _field.name in post_init_attributes:
                continue
            elif _field.name not in dictionary:
                if fields_required and _field.name not in optional_fields:
                    raise KeyError(f"Unexpected missing key: {_field.name}")
            else:
                value = dictionary[_field.name]
                _dictionary[_field.name] = value
        return cls(**_dictionary)

class MusicDataset(InfoAudioDataset):
    """Music dataset is an AudioDataset with music-related metadata.

    Args:
        info_fields_required (bool): Whether to enforce having required fields.

    """
    def __init__(self, *args, info_fields_required: bool = True,
                 **kwargs):
        kwargs['return_info'] = True  # We require the info for each song of the dataset.
        super().__init__(*args, **kwargs)
        self.info_fields_required = info_fields_required
        self.downsample_rate = 640
        self.sr = 32000

    def __getitem__(self, index):
        wav, info = super().__getitem__(index)
        info_data = info.to_dict()
        
        root_path = Path(info.meta.path).parent
        music_info_path = str(info.meta.path).replace('music.wav', 'tags.json') # not used
        intensity_path = str(info.meta.path).replace('music.wav', 'intensity.npy')
        beats_path = str(info.meta.path).replace('music.wav', 'beats.npy')
        clipcls_path = str(info.meta.path).replace('music.wav', 'clipcls.pt')
        image_path = str(info.meta.path).replace('music.wav', 'image.pt')
        mae_path = str(info.meta.path).replace('music.wav', 'mae.pt')
        melody_path = str(info.meta.path).replace('music.wav', 'melody.pt')
        va_path = str(info.meta.path).replace('music.wav', 'va.pt')
        
        if Path(music_info_path).exists():
            with open(music_info_path, 'r') as json_file:
                music_data = json.load(json_file)
                music_data.update(info_data)
                music_info = MusicInfo.from_dict(music_data, fields_required=self.info_fields_required)
        else:
            music_info = MusicInfo.from_dict(info_data, fields_required=False)

        # clipcls
        if Path(clipcls_path).exists():
            clipcls = torch.load(clipcls_path, weights_only=True) # sec, 768
            duration = clipcls.shape[0]
            clipcls = clipcls[int(info.seek_time):, :]
            music_info.visual_content = ClsCondition(
                clipcls=clipcls[None], length=torch.tensor([clipcls.shape[0]]),
                path=[root_path])
        # finegrained_content
        if Path(clipcls_path).exists() and Path(image_path).exists() and Path(mae_path).exists():
            image = torch.load(image_path, weights_only=True) # sec, 577, 1024
            mae = torch.load(mae_path, weights_only=True) # sec, 1024
            image = image[int(info.seek_time):, :, :]
            image = image.transpose(0, 2)
            image = F.pad(image, (0, duration - image.shape[-1]))
            image = image.transpose(0, 2)
            mae = mae[int(info.seek_time):, :]
            mae = mae.transpose(0, 1)
            mae = F.pad(mae, (0, duration - mae.shape[-1]))
            mae = mae.transpose(0, 1)
            clipcls = clipcls.transpose(0, 1)
            clipcls = F.pad(clipcls, (0, duration - clipcls.shape[-1]))
            clipcls = clipcls.transpose(0, 1)
            music_info.finegrained_content = FineGrainCondition(
                clipcls=clipcls[None], image=image[:,1:,:][None], mae=mae[None], length=torch.tensor([clipcls.shape[0]]),
                path=[root_path])
        
        feat_hz = self.sr / self.downsample_rate
        n_frames_feat = int(info.n_frames // self.downsample_rate) + 1
        # beat
        if Path(beats_path).exists():
            ## beat&bar: 2 x T
            feat_beats = np.zeros((2, n_frames_feat))
            
            beats_np = np.load(beats_path)
            beat_time = beats_np[:, 0]
            bar_time = beats_np[np.where(beats_np[:, 1] == 1)[0], 0]
            beat_frame = [
                int((t-info.seek_time)*feat_hz) for t in beat_time
                    if (t >= info.seek_time and t < info.seek_time + self.segment_duration)]
            bar_frame =[
                int((t-info.seek_time)*feat_hz) for t in bar_time
                    if (t >= info.seek_time and t < info.seek_time + self.segment_duration)]
            feat_beats[0, beat_frame] = 1
            feat_beats[1, bar_frame] = 1
            kernel = np.array([0.05, 0.1, 0.3, 0.9, 0.3, 0.1, 0.05])
            feat_beats[0] = np.convolve(feat_beats[0] , kernel, 'same') # apply soft kernel
            beat_events = feat_beats[0] + feat_beats[1]
            beat_events = torch.tensor(beat_events).unsqueeze(0) # [T] -> [1, T]   1, 1500(sec*framerate)
            music_info.beat = BeatCondition(beat=beat_events[None], length=torch.tensor([n_frames_feat]),
                                            path=[root_path])
        
        # intensity
        if Path(intensity_path).exists():
            intensity = torch.tensor(np.load(intensity_path)).unsqueeze(0) # 1, 1501(sec*framerate)
            intensity = intensity[:, int(info.seek_time*feat_hz):]
            music_info.intensity = IntensityCondition(
                intensity=intensity[None], length=torch.tensor([n_frames_feat]),
                path=[root_path])

        # melody
        if Path(melody_path).exists():
            melody = torch.load(melody_path, weights_only=True) # 12, 1501(sec*framerate)
            melody = melody[:, int(info.seek_time*feat_hz):]
            music_info.melody = MelodyCondition(
                melody=melody[None], length=torch.tensor([n_frames_feat]),
                path=[root_path])
            
        if Path(va_path).exists():
            va = torch.load(va_path, weights_only=True) # 2, 60(sec*2)
            assert va.dim() == 2 and va.shape[0] == 2 and va.shape[1] == 2 * self.segment_duration # a, v
            replication_factor = int(feat_hz) // 2
            va = va.unsqueeze(2).expand(2, va.shape[1], replication_factor).reshape(2, int(feat_hz * self.segment_duration))
            va = va[:, int(info.seek_time*feat_hz):]
            music_info.va = VACondition(
                va=va[None], length=torch.tensor([n_frames_feat]),
                path=[root_path])

        music_info.self_wav = WavCondition(
            wav=wav[None], length=torch.tensor([info.n_frames]),
            sample_rate=[info.sample_rate], path=[info.meta.path], seek_time=[info.seek_time])

        return wav, music_info
