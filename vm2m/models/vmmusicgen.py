# Modified from Audiocraft (https://github.com/facebookresearch/audiocraft)

"""
Main model for using MusicGen. This will combine all the required components
and provide easy access to the generation API.
"""

import typing as tp
import warnings

import torch
import numpy as np

from .encodec import CompressionModel
from .genmodel import BaseGenModel
from .lm import LMModel
from .loaders import load_compression_model, load_lm_model
from ..data.audio_utils import convert_audio
from .conditioners import ConditioningAttributes, WavCondition, StyleConditioner,\
                        ClsCondition, FineGrainCondition, MelodyCondition, IntensityCondition, BeatCondition, VACondition

MelodyList = tp.List[tp.Optional[torch.Tensor]]
MelodyType = tp.Union[torch.Tensor, MelodyList]

class MusicGen(BaseGenModel):
    """MusicGen main model with convenient generation API.

    Args:
        name (str): name of the model.
        compression_model (CompressionModel): Compression model
            used to map audio to invertible discrete representations.
        lm (LMModel): Language model over discrete representations.
        max_duration (float, optional): maximum duration the model can produce,
            otherwise, inferred from the training params.
    """
    def __init__(self, name: str, compression_model: CompressionModel, lm: LMModel,
                 max_duration: tp.Optional[float] = None):
        super().__init__(name, compression_model, lm, max_duration)
        self.set_generation_params(duration=15)  # default duration

    @staticmethod
    def get_pretrained(name: str = 'facebook/musicgen-melody', device=None):
        """Return pretrained model, we provide four models:
        - facebook/musicgen-small (300M), text to music,
          # see: https://huggingface.co/facebook/musicgen-small
        - facebook/musicgen-medium (1.5B), text to music,
          # see: https://huggingface.co/facebook/musicgen-medium
        - facebook/musicgen-melody (1.5B) text to music and text+melody to music,
          # see: https://huggingface.co/facebook/musicgen-melody
        - facebook/musicgen-large (3.3B), text to music,
          # see: https://huggingface.co/facebook/musicgen-large
        - facebook/musicgen-style (1.5 B), text and style to music,
          # see: https://huggingface.co/facebook/musicgen-style
        """
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        lm = load_lm_model(name, device=device)
        compression_model = load_compression_model(name, device=device)
        if 'self_wav' in lm.condition_provider.conditioners:
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True
            lm.condition_provider.conditioners['self_wav']._use_masking = False

        return MusicGen(name, compression_model, lm)

    def set_generation_params(self, use_sampling: bool = True, top_k: int = 250,
                              top_p: float = 0.0, temperature: float = 1.0,
                              duration: float = 30.0, cfg_coef: float = 3.0,
                              cfg_coef_beta: tp.Optional[float] = None,
                              two_step_cfg: bool = False, extend_stride: float = 18,):
        """Set the generation parameters for MusicGen.

        Args:
            use_sampling (bool, optional): Use sampling if True, else do argmax decoding. Defaults to True.
            top_k (int, optional): top_k used for sampling. Defaults to 250.
            top_p (float, optional): top_p used for sampling, when set to 0 top_k is used. Defaults to 0.0.
            temperature (float, optional): Softmax temperature parameter. Defaults to 1.0.
            duration (float, optional): Duration of the generated waveform. Defaults to 30.0.
            cfg_coef (float, optional): Coefficient used for classifier free guidance. Defaults to 3.0.
            cfg_coef_beta (float, optional): beta coefficient in double classifier free guidance.
                Should be only used for MusicGen melody if we want to push the text condition more than
                the audio conditioning. See paragraph 4.3 in https://arxiv.org/pdf/2407.12563 to understand
                double CFG.
            two_step_cfg (bool, optional): If True, performs 2 forward for Classifier Free Guidance,
                instead of batching together the two. This has some impact on how things
                are padded but seems to have little impact in practice.
            extend_stride: when doing extended generation (i.e. more than 30 seconds), by how much
                should we extend the audio each time. Larger values will mean less context is
                preserved, and shorter value will require extra computations.
        """
        assert extend_stride < self.max_duration, "Cannot stride by more than max generation duration."
        self.extend_stride = extend_stride
        self.duration = duration
        self.generation_params = {
            'use_sampling': use_sampling,
            'temp': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'cfg_coef': cfg_coef,
            'two_step_cfg': two_step_cfg,
            'cfg_coef_beta': cfg_coef_beta,
        }

    def set_style_conditioner_params(self, eval_q: int = 3, excerpt_length: float = 3.0,
                                     ds_factor: tp.Optional[int] = None,
                                     encodec_n_q: tp.Optional[int] = None) -> None:
        """Set the parameters of the style conditioner
        Args:
            eval_q (int): the number of residual quantization streams used to quantize the style condition
                the smaller it is, the narrower is the information bottleneck
            excerpt_length (float): the excerpt length in seconds that is extracted from the audio
                conditioning
            ds_factor: (int): the downsampling factor used to downsample the style tokens before
                using them as a prefix
            encodec_n_q: (int, optional): if encodec is used as a feature extractor, sets the number
                of streams that is used to extract features
        """
        assert isinstance(self.lm.condition_provider.conditioners.self_wav, StyleConditioner), \
            "Only use this function if you model is MusicGen-Style"
        self.lm.condition_provider.conditioners.self_wav.set_params(eval_q=eval_q,
                                                                    excerpt_length=excerpt_length,
                                                                    ds_factor=ds_factor,
                                                                    encodec_n_q=encodec_n_q)

    def generate_with_chroma(self, videos: tp.List[str], melody_wavs: MelodyType,
                             melody_sample_rate: int, progress: bool = False,
                             return_tokens: bool = False) -> tp.Union[torch.Tensor,
                                                                      tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples conditioned on text and melody.

        Args:
            videos (list of str): A list of file path strings used as video conditioning.
            melody_wavs: (torch.Tensor or list of Tensor): A batch of waveforms used as
                melody conditioning. Should have shape [B, C, T] with B matching the video length,
                C=1 or 2. It can be [C, T] if there is a single video. It can also be
                a list of [C, T] tensors.
            melody_sample_rate: (int): Sample rate of the melody waveforms.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        if isinstance(melody_wavs, torch.Tensor):
            if melody_wavs.dim() == 2:
                melody_wavs = melody_wavs[None]
            if melody_wavs.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            melody_wavs = list(melody_wavs)
        else:
            for melody in melody_wavs:
                if melody is not None:
                    assert melody.dim() == 2, "One melody in the list has the wrong number of dims."

        melody_wavs = [
            convert_audio(wav, melody_sample_rate, self.sample_rate, self.audio_channels)
            if wav is not None else None
            for wav in melody_wavs]
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(videos=videos, prompt=None,
                                                                        melody_wavs=melody_wavs)
        assert prompt_tokens is None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    def generate_with_video(self, cls: MelodyType, image: MelodyType, mae: MelodyType, 
                            progress: bool = False, return_tokens: bool = False) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples conditioned on video.
        Args:
            
        """
        if isinstance(cls, torch.Tensor):
            if cls.dim() == 2:
                cls = cls[None]
            if cls.dim() != 3:
                raise ValueError("cls should have a shape [B, T, C].")
            cls = list(cls)
        else:
            for scls in cls:
                if scls is not None:
                    assert scls.dim() == 2, "One cls in the list has the wrong number of dims."
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image[None]
            if image.dim()!= 4:
                raise ValueError("image should have a shape [B, T, P, C].")
            image = list(image)
        else:
            for simage in image:
                if simage is not None:
                    assert simage.dim() == 3, "One image in the list has the wrong number of dims."
        if isinstance(mae, torch.Tensor):
            if mae.dim() == 2:
                mae = mae[None]
            if mae.dim()!= 3:
                raise ValueError("mae should have a shape [B, T, C].")
            mae = list(mae)
        else:
            for smae in mae:
                if smae is not None:
                    assert smae.dim() == 2, "One mae in the list has the wrong number of dims."
        
        attributes, prompt_tokens = self._prepare_tokens_and_attributes_with_video(cls=cls, image=image, mae=mae)
        assert prompt_tokens is None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)
    
    def generate_with_muticond(self, cls: MelodyType, image: MelodyType, mae: MelodyType, 
                               melody: MelodyType, intensity: MelodyType = [None], beats = [None], va = [None], 
                            progress: bool = False, return_tokens: bool = False) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples conditioned on muti-condition.
        Args:
            
        """
        if isinstance(cls, torch.Tensor):
            if cls.dim() == 2:
                cls = cls[None]
            if cls.dim() != 3:
                raise ValueError("cls should have a shape [B, T, C].")
            cls = list(cls)
        else:
            for scls in cls:
                if scls is not None:
                    assert scls.dim() == 2, "One cls in the list has the wrong number of dims."
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image[None]
            if image.dim()!= 4:
                raise ValueError("image should have a shape [B, T, P, C].")
            image = list(image)
        else:
            for simage in image:
                if simage is not None:
                    assert simage.dim() == 3, "One image in the list has the wrong number of dims."
        if isinstance(mae, torch.Tensor):
            if mae.dim() == 2:
                mae = mae[None]
            if mae.dim()!= 3:
                raise ValueError("mae should have a shape [B, T, C].")
            mae = list(mae)
        else:
            for smae in mae:
                if smae is not None:
                    assert smae.dim() == 2, "One mae in the list has the wrong number of dims."
        
        if isinstance(melody, torch.Tensor):
            if melody.dim() == 2:
                melody = melody[None]
            if melody.dim()!= 3:
                raise ValueError("melody should have a shape [B, T, C].")
            melody = list(melody)
        else:
            for smelody in melody:
                if smelody is not None:
                    assert smelody.dim() == 2, "One melody in the list has the wrong number of dims."
        
        if isinstance(intensity, torch.Tensor):
            if intensity.dim() == 2:
                intensity = intensity[None]
            if intensity.dim()!= 3:
                raise ValueError("intensity should have a shape [B, T, C].")
            intensity = list(intensity)
        else:
            for sintensity in intensity:
                if sintensity is not None:
                    assert sintensity.dim() == 2, "One intensity in the list has the wrong number of dims."
        
        if isinstance(beats, np.ndarray):
            assert beats.ndim == 2, "One beats in the list has the wrong number of dims."
            feat_beats = np.zeros((2, self.duration * self.frame_rate+1))
            beat_time = beats[:, 0]
            bar_time = beats[np.where(beats[:, 1] == 1)[0], 0]
            beat_frame = [
                int(t*self.frame_rate) for t in beat_time
                    if (t >= 0 and t < self.duration)]
            bar_frame =[
                int(t*self.frame_rate) for t in bar_time
                    if (t >= 0 and t < self.duration)]
            feat_beats[0, beat_frame] = 1
            feat_beats[1, bar_frame] = 1
            kernel = np.array([0.05, 0.1, 0.3, 0.9, 0.3, 0.1, 0.05])
            feat_beats[0] = np.convolve(feat_beats[0] , kernel, 'same') # apply soft kernel
            beat_events = feat_beats[0] + feat_beats[1]
            beat_events = torch.tensor(beat_events).unsqueeze(0) # [T] -> [1, T]   1, 1500(sec*framerate)
            beat_events = beat_events[None]
            if beat_events.dim()!= 3:
                raise ValueError("beat should have a shape [B, T, C].")
            beats = list(beat_events)
        elif isinstance(beats, torch.Tensor):
            if beats.dim() == 2:
                beats = beats[None]
            if beats.dim()!= 3:
                raise ValueError("beats should have a shape [B, T, C].")
            beats = list(beats)
        else:
            for beat in beats:
                assert beat is None
                
        if isinstance(va, torch.Tensor):
            if va.dim() == 2:
                va = va[None]
            if va.dim()!= 3:
                raise ValueError("va should have a shape [B, T, C].")
            va = list(va)
        else:
            for sva in va:
                if sva is not None:
                    assert sva.dim() == 2, "One va value in the list has the wrong number of dims."

        attributes, prompt_tokens = self._prepare_tokens_and_attributes_with_muticond(cls=cls, image=image, mae=mae, melody=melody, intensity=intensity, beat=beats, va=va)
        assert prompt_tokens is None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    @torch.no_grad()
    def _prepare_tokens_and_attributes_with_video(
            self,
            cls: tp.Optional[MelodyList],
            image: tp.Optional[MelodyList],
            mae: tp.Optional[MelodyList],
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[torch.Tensor]]:
        """Prepare model inputs.

        Args:
        """
        attributes = [
            ConditioningAttributes(
                video={'visual_content': ClsCondition(video[None].to(device=self.device), 
                        torch.tensor([video.shape[1]], device=self.device), 
                        path=[None])})
            for video in cls]
        
        assert image is not None and mae is not None
        if 'finegrained_content' not in self.lm.condition_provider.conditioners:
            raise RuntimeError("This model doesn't support finegrained_content conditioning. "
                                "Use the `finegrained_content` model.")
        assert len(image) == len(cls), \
            f"number of image must match number of videos! " \
            f"got image len={len(image)}, and videos len={len(cls)}"
        assert len(mae) == len(cls), \
            f"number of mae must match number of videos! " \
            f"got mae len={len(mae)}, and videos len={len(cls)}"
        for attr, simage, smae in zip(attributes, image, mae):
            if simage is None and smae is None:
                attr.finegrained_content['finegrained_content'] = FineGrainCondition(
                    torch.zeros((1, 1, 1, 1), device=self.device),
                    attr.video['visual_content'].clipcls,
                    torch.zeros((1, 1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    path=[None])
            elif simage is not None and smae is not None:
                attr.finegrained_content['finegrained_content'] = FineGrainCondition(
                    simage[None][:,:,1:,].to(device=self.device),
                    attr.video['visual_content'].clipcls,
                    smae[None].to(device=self.device),
                    attr.video['visual_content'].length,
                    path=[None],
                )
            else:
                raise ValueError("Both image and mae must be provided or both must be None.")

        prompt_tokens = None
        return attributes, prompt_tokens
    
    @torch.no_grad()
    def _prepare_tokens_and_attributes_with_muticond(
            self,
            cls: tp.Optional[MelodyList],
            image: tp.Optional[MelodyList],
            mae: tp.Optional[MelodyList],
            melody: tp.Optional[MelodyList],
            intensity: tp.Optional[MelodyList] = None,
            beat: tp.Optional[MelodyList] = None,
            va: tp.Optional[MelodyList] = None,
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[torch.Tensor]]:
        """Prepare model inputs.

        Args:
        """
        attributes = [
            ConditioningAttributes(
                video={'visual_content': ClsCondition(video[None].to(device=self.device), 
                        torch.tensor([video.shape[1]], device=self.device), 
                        path=[None])})
            for video in cls]
        
        assert image is not None and mae is not None
        if 'finegrained_content' not in self.lm.condition_provider.conditioners:
            raise RuntimeError("This model doesn't support finegrained_content conditioning. "
                                "Use the `finegrained_content` model.")
        assert len(image) == len(cls), \
            f"number of image must match number of videos! " \
            f"got image len={len(image)}, and videos len={len(cls)}"
        assert len(mae) == len(cls), \
            f"number of mae must match number of videos! " \
            f"got mae len={len(mae)}, and videos len={len(cls)}"
        for attr, simage, smae, smelody, sintensity, sbeat, sva in zip(attributes, image, mae, melody, intensity, beat, va):
            if simage is None and smae is None:
                attr.finegrained_content['finegrained_content'] = FineGrainCondition(
                    torch.zeros((1, 1, 1, 1), device=self.device),
                    attr.video['visual_content'].clipcls,
                    torch.zeros((1, 1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    path=[None])
            elif simage is not None and smae is not None:
                attr.finegrained_content['finegrained_content'] = FineGrainCondition(
                    simage[None][:,:,1:,].to(device=self.device),
                    attr.video['visual_content'].clipcls,
                    smae[None].to(device=self.device),
                    attr.video['visual_content'].length,
                    path=[None],
                )
            else:
                raise ValueError("Both image and mae must be provided or both must be None.")
            
            if sintensity is None:
                attr.intensity['intensity'] = IntensityCondition(
                    torch.zeros((1, 1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    path=[None])
            else:
                attr.intensity['intensity'] = IntensityCondition(
                    sintensity[None].to(device=self.device),
                    attr.video['visual_content'].length*self.frame_rate,
                    path=[None])
            
            if smelody is None:
                attr.melody['melody'] = MelodyCondition(
                    torch.zeros((1, 12, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    path=[None])
            else:
                attr.melody['melody'] = MelodyCondition(
                        smelody[None].to(device=self.device),
                        attr.video['visual_content'].length*self.frame_rate,
                        path=[None])

            if sbeat is None:
                attr.beat['beat'] = BeatCondition(
                    torch.zeros((1, 1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    path=[None])
            else:
                attr.beat['beat'] = BeatCondition(
                    sbeat[None].to(device=self.device),
                    attr.video['visual_content'].length*self.frame_rate,
                    path=[None])
                
            if sva is None:
                attr.va['va'] = VACondition(
                    torch.zeros((1, 2, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    path=[None])
            else:
                attr.va['va'] = VACondition(
                    sva[None].to(device=self.device),
                    attr.video['visual_content'].length*self.frame_rate,
                    path=[None])

        prompt_tokens = None
        return attributes, prompt_tokens

    @torch.no_grad()
    def _prepare_tokens_and_attributes(
            self,
            videos: tp.Sequence[tp.Optional[str]],
            prompt: tp.Optional[torch.Tensor],
            melody_wavs: tp.Optional[MelodyList] = None,
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[torch.Tensor]]:
        """Prepare model inputs.

        Args:
            videos (list of str): A list of file path strings used as video conditioning.
            prompt (torch.Tensor): A batch of waveforms used for continuation.
            melody_wavs (torch.Tensor, optional): A batch of waveforms
                used as melody conditioning. Defaults to None.
        """
        attributes = [
            ConditioningAttributes(video={'visual_content': video})
            for video in videos]

        if melody_wavs is None:
            for attr in attributes:
                attr.wav['self_wav'] = WavCondition(
                    torch.zeros((1, 1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    sample_rate=[self.sample_rate],
                    path=[None])
        else:
            if 'self_wav' not in self.lm.condition_provider.conditioners:
                raise RuntimeError("This model doesn't support melody conditioning. "
                                   "Use the `melody` model.")
            assert len(melody_wavs) == len(videos), \
                f"number of melody wavs must match number of videos! " \
                f"got melody len={len(melody_wavs)}, and videos len={len(videos)}"
            for attr, melody in zip(attributes, melody_wavs):
                if melody is None:
                    attr.wav['self_wav'] = WavCondition(
                        torch.zeros((1, 1, 1), device=self.device),
                        torch.tensor([0], device=self.device),
                        sample_rate=[self.sample_rate],
                        path=[None])
                else:
                    attr.wav['self_wav'] = WavCondition(
                        melody[None].to(device=self.device),
                        torch.tensor([melody.shape[-1]], device=self.device),
                        sample_rate=[self.sample_rate],
                        path=[None],
                    )

        if prompt is not None:
            if videos is not None:
                assert len(videos) == len(prompt), "Prompt and nb. videos doesn't match"
            prompt = prompt.to(self.device)
            prompt_tokens, scale = self.compression_model.encode(prompt)
            assert scale is None
        else:
            prompt_tokens = None
        return attributes, prompt_tokens

    def _generate_tokens(self, attributes: tp.List[ConditioningAttributes],
                         prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False) -> torch.Tensor:
        """Generate discrete audio tokens given audio prompt and/or conditions.

        Args:
            attributes (list of ConditioningAttributes): Conditions used for generation (text/melody).
            prompt_tokens (torch.Tensor, optional): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        """
        total_gen_len = int(self.duration * self.frame_rate)
        max_prompt_len = int(min(self.duration, self.max_duration) * self.frame_rate)
        current_gen_offset: int = 0

        def _progress_callback(generated_tokens: int, tokens_to_generate: int):
            generated_tokens += current_gen_offset
            if self._progress_callback is not None:
                # Note that total_gen_len might be quite wrong depending on the
                # codebook pattern used, but with delay it is almost accurate.
                self._progress_callback(generated_tokens, tokens_to_generate)
            else:
                print(f'{generated_tokens: 6d} / {tokens_to_generate: 6d}', end='\r')

        if prompt_tokens is not None:
            assert max_prompt_len >= prompt_tokens.shape[-1], \
                "Prompt is longer than audio to generate"

        callback = None
        if progress:
            callback = _progress_callback

        if self.duration <= self.max_duration:
            # generate by sampling from LM, simple case.
            with self.autocast:
                gen_tokens = self.lm.generate(
                    prompt_tokens, attributes,
                    callback=callback, max_gen_len=total_gen_len, **self.generation_params)

        else:
            # now this gets a bit messier, we need to handle prompts,
            # melody conditioning etc.
            ref_wavs = [attr.wav['self_wav'] for attr in attributes]
            all_tokens = []
            if prompt_tokens is None:
                prompt_length = 0
            else:
                all_tokens.append(prompt_tokens)
                prompt_length = prompt_tokens.shape[-1]

            assert self.extend_stride is not None, "Stride should be defined to generate beyond max_duration"
            assert self.extend_stride < self.max_duration, "Cannot stride by more than max generation duration."
            stride_tokens = int(self.frame_rate * self.extend_stride)

            while current_gen_offset + prompt_length < total_gen_len:
                time_offset = current_gen_offset / self.frame_rate
                chunk_duration = min(self.duration - time_offset, self.max_duration)
                max_gen_len = int(chunk_duration * self.frame_rate)
                for attr, ref_wav in zip(attributes, ref_wavs):
                    wav_length = ref_wav.length.item()
                    if wav_length == 0:
                        continue
                    # We will extend the wav periodically if it not long enough.
                    # we have to do it here rather than in conditioners.py as otherwise
                    # we wouldn't have the full wav.
                    initial_position = int(time_offset * self.sample_rate)
                    wav_target_length = int(self.max_duration * self.sample_rate)
                    positions = torch.arange(initial_position,
                                             initial_position + wav_target_length, device=self.device)
                    attr.wav['self_wav'] = WavCondition(
                        ref_wav[0][..., positions % wav_length],
                        torch.full_like(ref_wav[1], wav_target_length),
                        [self.sample_rate] * ref_wav[0].size(0),
                        [None], [0.])
                with self.autocast:
                    gen_tokens = self.lm.generate(
                        prompt_tokens, attributes,
                        callback=callback, max_gen_len=max_gen_len, **self.generation_params)
                if prompt_tokens is None:
                    all_tokens.append(gen_tokens)
                else:
                    all_tokens.append(gen_tokens[:, :, prompt_tokens.shape[-1]:])
                prompt_tokens = gen_tokens[:, :, stride_tokens:]
                prompt_length = prompt_tokens.shape[-1]
                current_gen_offset += stride_tokens

            gen_tokens = torch.cat(all_tokens, dim=-1)
        return gen_tokens
