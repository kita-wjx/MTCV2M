# modified by https://github.com/YatingMusic/MusiConGen/blob/main/audiocraft/audiocraft/modules/conditioners.py
import torch
from torch import nn
import math
import typing as tp
from pathlib import Path
import logging
import random
import flashy
import torch.nn.functional as F
from dataclasses import dataclass, field
from itertools import chain
from collections import defaultdict
from copy import deepcopy

import einops
import omegaconf

from ..utils.cache import EmbeddingCache
from ..utils.autocast import TorchAutocast
from ..utils.utils import length_to_mask, collate, range_to_mask
from ..data.audio import audio_read
from ..data.audio_utils import convert_audio
from ..quantization import ResidualVectorQuantizer
from .chroma import ChromaExtractor
from .transformer import create_sin_embedding, StreamingTransformer
from .fusionnet import FusionNet
from .temp_aggregator import TemporalAggregatorWithConv
from .streaming import StreamingModule
from ..data.audio_dataset import SegmentInfo
from .DynamicWeights import DWNet

logger = logging.getLogger(__name__)
ConditionType = tp.Tuple[torch.Tensor, torch.Tensor]  # condition, mask

class ClsCondition(tp.NamedTuple):
    clipcls: torch.Tensor
    length: torch.Tensor
    path: tp.List[tp.Optional[str]] = []

class FineGrainCondition(tp.NamedTuple):
    image: torch.Tensor
    clipcls: torch.Tensor
    mae: torch.Tensor
    length: torch.Tensor
    path: tp.List[tp.Optional[str]] = []

class WavCondition(tp.NamedTuple):
    wav: torch.Tensor
    length: torch.Tensor
    sample_rate: tp.List[int]
    path: tp.List[tp.Optional[str]] = []
    seek_time: tp.List[tp.Optional[float]] = []

class IntensityCondition(tp.NamedTuple):
    intensity: torch.Tensor
    length: torch.Tensor
    path: tp.List[tp.Optional[str]] = []

class BeatCondition(tp.NamedTuple):
    beat: torch.Tensor
    length: torch.Tensor
    path: tp.List[tp.Optional[str]] = []

class MelodyCondition(tp.NamedTuple):
    melody: torch.Tensor
    length: torch.Tensor
    path: tp.List[tp.Optional[str]] = []

class VACondition(tp.NamedTuple):
    va: torch.Tensor
    length: torch.Tensor
    path: tp.List[tp.Optional[str]] = []

@dataclass
class ConditioningAttributes:
    video: tp.Dict[str, ClsCondition] = field(default_factory=dict)
    finegrained_content: tp.Dict[str, FineGrainCondition] = field(default_factory=dict)
    wav: tp.Dict[str, WavCondition] = field(default_factory=dict)
    intensity: tp.Dict[str, IntensityCondition] = field(default_factory=dict)
    beat: tp.Dict[str, BeatCondition] = field(default_factory=dict)
    melody: tp.Dict[str, MelodyCondition] = field(default_factory=dict)
    va: tp.Dict[str, VACondition] = field(default_factory=dict)

    def __getitem__(self, item):
        return getattr(self, item)
    
    @property
    def video_attributes(self):
        return self.video.keys()
    
    @property
    def finegrained_content_attributes(self):
        return self.finegrained_content.keys()
    
    @property
    def wav_attributes(self):
        return self.wav.keys()
    
    @property
    def intensity_attributes(self):
        return self.intensity.keys()
    
    @property
    def beat_attributes(self):
        return self.beat.keys()
    
    @property
    def melody_attributes(self):
        return self.melody.keys()
    
    @property
    def va_attributes(self):
        return self.va.keys()

    @property
    def attributes(self):
        return {
            "video": self.video_attributes,
            "finegrained_content": self.finegrained_content_attributes,
            "wav": self.wav_attributes,
            "intensity": self.intensity_attributes,
            "beat": self.beat_attributes,
            "melody": self.melody_attributes,
            "va": self.va_attributes,
        }

    def to_flat_dict(self):
        return {
            **{f"video.{k}": v for k, v in self.video.items()},
            **{f"finegrained_content.{k}": v for k, v in self.finegrained_content.items()},
            **{f"wav.{k}": v for k, v in self.wav.items()},
            **{f"intensity.{k}": v for k, v in self.intensity.items()},
            **{f"beat.{k}": v for k, v in self.beat.items()},
            **{f"melody.{k}": v for k, v in self.melody.items()},
            **{f"va.{k}": v for k, v in self.va.items()},
        }

    @classmethod
    def from_flat_dict(cls, x):
        out = cls()
        for k, v in x.items():
            kind, att = k.split(".")
            out[kind][att] = v
        return out

class SegmentWithAttributes(SegmentInfo):
    """Base class for all dataclasses that are used for conditioning.
    All child classes should implement `to_condition_attributes` that converts
    the existing attributes to a dataclass of type ConditioningAttributes.
    """
    def to_condition_attributes(self) -> ConditioningAttributes:
        raise NotImplementedError()

class BaseConditioner(nn.Module):
    """Base model for all conditioner modules.
    
    Args:
        dim (int): Hidden dim of the model.
        output_dim (int): Output dim of the conditioner.
    """
    def __init__(self, dim: int, output_dim: int):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.output_proj = nn.Linear(dim, output_dim)

    def tokenize(self, *args, **kwargs) -> tp.Any:
        """Should be any part of the processing that will lead to a synchronization
        point, e.g. BPE tokenization with transfer to the GPU.

        The returned value will be saved and return later when calling forward().
        """
        raise NotImplementedError()

    def forward(self, inputs: tp.Any) -> ConditionType:
        """Gets input that should be used as conditioning (e.g, video, description or a waveform).
        Outputs a ConditionType, after the input data was embedded as a dense vector.

        Returns:
            ConditionType:
                - A tensor of size [B, T, D] where B is the batch size, T is the length of the
                  output embedding and D is the dimension of the embedding.
                - And a mask indicating where the padding tokens.
        """
        raise NotImplementedError()

class VideoConditioner(BaseConditioner):
    ...

class CLIPConditioner(VideoConditioner):
    """CLIP-based VideoConditioner.

    Args:
        name (str): Name of the CLIP model.
        output_dim (int): Output dim of the conditioner.
        device (str): Device for CLIP Conditioner.
        finetune (bool): Whether to fine-tune the last layer of CLIP at train time.
    """
    MODELS = ["ViT-L/14@336px"]
    MODELS_DIMS = {
        "ViT-L/14@336px": 768,
    }
    def __init__(self, name: str, output_dim: int, device: str, kernel_size: int = 3):
        assert name in self.MODELS, f"Unrecognized CLIP model name (should in {self.MODELS})"
        super().__init__(self.MODELS_DIMS[name], output_dim)
        self.device = device
        self.name = name
        self.aggregator = TemporalAggregatorWithConv(self.MODELS_DIMS[name], self.MODELS_DIMS[name], kernel_size=kernel_size)

    def tokenize(self, x: ClsCondition) -> ClsCondition:
        cls, length, path = x
        assert length is not None
        return ClsCondition(cls.to(self.device), length.to(self.device), path)

    def forward(self, inputs: ClsCondition) -> ConditionType:
        clipcls, lengths, *_ = inputs
        embeds = self.aggregator(clipcls) # B, 1 or sec, C -> B, C
        embeds = embeds.unsqueeze(1) # B, 1, C
        embeds = embeds.to(self.output_proj.weight)
        embeds = self.output_proj(embeds) # B, 1, C
        if lengths is not None:
            mask = length_to_mask(lengths, max_len=embeds.shape[1]).int()  # B, 1
        else:
            mask = torch.ones_like(embeds[..., 0])
        embeds = (embeds * mask.unsqueeze(-1).to(self.device))
        return embeds, mask

class FineGrainFusionConditioner(BaseConditioner):
    """Fine grain fusion conditioner.
    """
    def __init__(self, output_dim: int, device: str, video_dim: int = 1024,
                 fusion_dim: int = 768, layers: int = 4, heads: int = 8, video_len: int = 30, **kwargs):
        super().__init__(dim=fusion_dim, output_dim=output_dim)
        self.fusionnet = FusionNet(video_dim=video_dim, dim=fusion_dim, layers=layers, heads=heads)
        self.device = device
        self.video_len = video_len
        
    def tokenize(self, x: FineGrainCondition) -> FineGrainCondition:
        image, cls, mae, length, path = x
        assert length is not None
        return FineGrainCondition(image.to(self.device), cls.to(self.device), mae.to(self.device), length.to(self.device), path)
    
    def forward(self, inputs: FineGrainCondition) -> ConditionType:
        image, clipcls, mae, lengths, *_ = inputs
        embeds = self.fusionnet(image, mae, clipcls) # B, sec, 768
        embeds = embeds.to(self.output_proj.weight)
        embeds = self.output_proj(embeds)
        if lengths is not None:
            mask = length_to_mask(lengths, max_len=embeds.shape[1]).int()  # type: ignore
        else:
            mask = torch.ones_like(embeds[..., 0])
        embeds = (embeds * mask.unsqueeze(-1).to(self.device))
        
        return embeds, mask

class WaveformConditioner(BaseConditioner):
    """Base class for all conditioners that take a waveform as input.
    Classes that inherit must implement `_get_wav_embedding` that outputs
    a continuous tensor, and `_downsampling_factor` that returns the down-sampling
    factor of the embedding model.

    Args:
        dim (int): The internal representation dimension.
        output_dim (int): Output dimension.
        device (tp.Union[torch.device, str]): Device.
    """
    def __init__(self, dim: int, output_dim: int, device: tp.Union[torch.device, str]):
        super().__init__(dim, output_dim)
        self.device = device
        # if False no masking is done, used in ChromaStemConditioner when completing by periodicity a sample.
        self._use_masking = True

    def tokenize(self, x: WavCondition) -> WavCondition:
        wav, length, sample_rate, path, seek_time = x
        assert length is not None
        return WavCondition(wav.to(self.device), length.to(self.device), sample_rate, path, seek_time)

    def _get_wav_embedding(self, x: WavCondition) -> torch.Tensor:
        """Gets as input a WavCondition and returns a dense embedding."""
        raise NotImplementedError()

    def _downsampling_factor(self):
        """Returns the downsampling factor of the embedding model."""
        raise NotImplementedError()

    def forward(self, x: WavCondition) -> ConditionType:
        """Extract condition embedding and mask from a waveform and its metadata.
        Args:
            x (WavCondition): Waveform condition containing raw waveform and metadata.
        Returns:
            ConditionType: a dense vector representing the conditioning along with its mask
        """
        wav, lengths, *_ = x
        with torch.no_grad():
            embeds = self._get_wav_embedding(x)
        embeds = embeds.to(self.output_proj.weight)
        embeds = self.output_proj(embeds)

        if lengths is not None and self._use_masking:
            lengths = lengths / self._downsampling_factor()
            mask = length_to_mask(lengths, max_len=embeds.shape[1]).int()  # type: ignore
        else:
            mask = torch.ones_like(embeds[..., 0])
        embeds = (embeds * mask.unsqueeze(-1))
        return embeds, mask

class ChromaStemConditioner(WaveformConditioner):
    """Chroma conditioner based on stems.
    The ChromaStemConditioner uses DEMUCS to first filter out drums and bass, as
    the drums and bass often dominate the chroma leading to the chroma features
    not containing information about the melody.

    Args:
        output_dim (int): Output dimension for the conditioner.
        sample_rate (int): Sample rate for the chroma extractor.
        n_chroma (int): Number of chroma bins for the chroma extractor.
        radix2_exp (int): Size of stft window for the chroma extractor (power of 2, e.g. 12 -> 2^12).
        duration (int): duration used during training. This is later used for correct padding
            in case we are using chroma as prefix.
        match_len_on_eval (bool, optional): if True then all chromas are padded to the training
            duration. Defaults to False.
        eval_wavs (str, optional): path to a dataset manifest with waveform, this waveforms are used as
            conditions during eval (for cases where we don't want to leak test conditions like MusicCaps).
            Defaults to None.
        n_eval_wavs (int, optional): limits the number of waveforms used for conditioning. Defaults to 0.
        device (tp.Union[torch.device, str], optional): Device for the conditioner.
        **kwargs: Additional parameters for the chroma extractor.
    """
    def __init__(self, output_dim: int, sample_rate: int, n_chroma: int, radix2_exp: int,
                 duration: float, match_len_on_eval: bool = True, eval_wavs: tp.Optional[str] = None,
                 n_eval_wavs: int = 0, cache_path: tp.Optional[tp.Union[str, Path]] = None,
                 device: tp.Union[torch.device, str] = 'cpu', **kwargs):
        from demucs import pretrained
        super().__init__(dim=n_chroma, output_dim=output_dim, device=device)
        self.autocast = TorchAutocast(enabled=device != 'cpu', device_type=self.device, dtype=torch.float32)
        self.sample_rate = sample_rate
        self.match_len_on_eval = match_len_on_eval
        if match_len_on_eval:
            self._use_masking = False
        self.duration = duration
        self.__dict__['demucs'] = pretrained.get_model('htdemucs').to(device)
        stem_sources: list = self.demucs.sources  # type: ignore
        self.stem_indices = torch.LongTensor([stem_sources.index('vocals'), stem_sources.index('other')]).to(device)
        self.chroma = ChromaExtractor(sample_rate=sample_rate, n_chroma=n_chroma,
                                      radix2_exp=radix2_exp, **kwargs).to(device)
        self.chroma_len = self._get_chroma_len()
        self.eval_wavs: tp.Optional[torch.Tensor] = self._load_eval_wavs(eval_wavs, n_eval_wavs)
        self.cache = None
        if cache_path is not None:
            self.cache = EmbeddingCache(Path(cache_path) / 'wav', self.device,
                                        compute_embed_fn=self._get_full_chroma_for_cache,
                                        extract_embed_fn=self._extract_chroma_chunk)

    def _downsampling_factor(self) -> int:
        return self.chroma.winhop

    def _load_eval_wavs(self, path: tp.Optional[str], num_samples: int) -> tp.Optional[torch.Tensor]:
        """Load pre-defined waveforms from a json.
        These waveforms will be used for chroma extraction during evaluation.
        This is done to make the evaluation on MusicCaps fair (we shouldn't see the chromas of MusicCaps).
        """
        if path is None:
            return None

        logger.info(f"Loading evaluation wavs from {path}")
        from ..data.audio_dataset import AudioDataset
        dataset: AudioDataset = AudioDataset.from_meta(
            path, segment_duration=self.duration, min_audio_duration=self.duration,
            sample_rate=self.sample_rate, channels=1)

        if len(dataset) > 0:
            eval_wavs = dataset.collater([dataset[i] for i in range(num_samples)]).to(self.device)
            logger.info(f"Using {len(eval_wavs)} evaluation wavs for chroma-stem conditioner")
            return eval_wavs
        else:
            raise ValueError("Could not find evaluation wavs, check lengths of wavs")

    def reset_eval_wavs(self, eval_wavs: tp.Optional[torch.Tensor]) -> None:
        self.eval_wavs = eval_wavs

    def has_eval_wavs(self) -> bool:
        return self.eval_wavs is not None

    def _sample_eval_wavs(self, num_samples: int) -> torch.Tensor:
        """Sample wavs from a predefined list."""
        assert self.eval_wavs is not None, "Cannot sample eval wavs as no eval wavs provided."
        total_eval_wavs = len(self.eval_wavs)
        out = self.eval_wavs
        if num_samples > total_eval_wavs:
            out = self.eval_wavs.repeat(num_samples // total_eval_wavs + 1, 1, 1)
        return out[torch.randperm(len(out))][:num_samples]

    def _get_chroma_len(self) -> int:
        """Get length of chroma during training."""
        dummy_wav = torch.zeros((1, int(self.sample_rate * self.duration)), device=self.device)
        dummy_chr = self.chroma(dummy_wav)
        return dummy_chr.shape[1]

    @torch.no_grad()
    def _get_stemmed_wav(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Get parts of the wav that holds the melody, extracting the main stems from the wav."""
        from demucs.apply import apply_model
        from demucs.audio import convert_audio
        with self.autocast:
            wav = convert_audio(
                wav, sample_rate, self.demucs.samplerate, self.demucs.audio_channels)  # type: ignore
            stems = apply_model(self.demucs, wav, device=self.device)  # type: ignore
            stems = stems[:, self.stem_indices]  # extract relevant stems for melody conditioning
            mix_wav = stems.sum(1)  # merge extracted stems to single waveform
            mix_wav = convert_audio(mix_wav, self.demucs.samplerate, self.sample_rate, 1)  # type: ignore
            return mix_wav

    @torch.no_grad()
    def _extract_chroma(self, wav: torch.Tensor) -> torch.Tensor:
        """Extract chroma features from the waveform."""
        with self.autocast:
            return self.chroma(wav)

    @torch.no_grad()
    def _compute_wav_embedding(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Compute wav embedding, applying stem and chroma extraction."""
        # avoid 0-size tensors when we are working with null conds
        if wav.shape[-1] == 1:
            return self._extract_chroma(wav)
        stems = self._get_stemmed_wav(wav, sample_rate)
        chroma = self._extract_chroma(stems)
        return chroma

    @torch.no_grad()
    def _get_full_chroma_for_cache(self, path: tp.Union[str, Path], x: WavCondition, idx: int) -> torch.Tensor:
        """Extract chroma from the whole audio waveform at the given path."""
        wav, sr = audio_read(path)
        wav = wav[None].to(self.device)
        wav = convert_audio(wav, sr, self.sample_rate, to_channels=1)
        chroma = self._compute_wav_embedding(wav, self.sample_rate)[0]
        return chroma

    def _extract_chroma_chunk(self, full_chroma: torch.Tensor, x: WavCondition, idx: int) -> torch.Tensor:
        """Extract a chunk of chroma from the full chroma derived from the full waveform."""
        wav_length = x.wav.shape[-1]
        seek_time = x.seek_time[idx]
        assert seek_time is not None, (
            "WavCondition seek_time is required "
            "when extracting chroma chunks from pre-computed chroma.")
        full_chroma = full_chroma.float()
        frame_rate = self.sample_rate / self._downsampling_factor()
        target_length = int(frame_rate * wav_length / self.sample_rate)
        index = int(frame_rate * seek_time)
        out = full_chroma[index: index + target_length]
        out = F.pad(out[None], (0, 0, 0, target_length - out.shape[0]))[0]
        return out.to(self.device)

    @torch.no_grad()
    def _get_wav_embedding(self, x: WavCondition) -> torch.Tensor:
        """Get the wav embedding from the WavCondition.
        The conditioner will either extract the embedding on-the-fly computing it from the condition wav directly
        or will rely on the embedding cache to load the pre-computed embedding if relevant.
        """
        sampled_wav: tp.Optional[torch.Tensor] = None
        if not self.training and self.eval_wavs is not None:
            # warn_once(logger, "Using precomputed evaluation wavs!")
            sampled_wav = self._sample_eval_wavs(len(x.wav))

        no_undefined_paths = all(p is not None for p in x.path)
        no_nullified_cond = x.wav.shape[-1] > 1
        if sampled_wav is not None:
            chroma = self._compute_wav_embedding(sampled_wav, self.sample_rate)
        elif self.cache is not None and no_undefined_paths and no_nullified_cond:
            paths = [Path(p) for p in x.path if p is not None]
            chroma = self.cache.get_embed_from_cache(paths, x)
        else:
            assert all(sr == x.sample_rate[0] for sr in x.sample_rate), "All sample rates in batch should be equal."
            chroma = self._compute_wav_embedding(x.wav, x.sample_rate[0])

        if self.match_len_on_eval:
            B, T, C = chroma.shape
            if T > self.chroma_len:
                chroma = chroma[:, :self.chroma_len]
                logger.debug(f"Chroma was truncated to match length! ({T} -> {chroma.shape[1]})")
            elif T < self.chroma_len:
                n_repeat = int(math.ceil(self.chroma_len / T))
                chroma = chroma.repeat(1, n_repeat, 1)
                chroma = chroma[:, :self.chroma_len]
                logger.debug(f"Chroma was repeated to match length! ({T} -> {chroma.shape[1]})")

        return chroma

    def tokenize(self, x: WavCondition) -> WavCondition:
        """Apply WavConditioner tokenization and populate cache if needed."""
        x = super().tokenize(x)
        no_undefined_paths = all(p is not None for p in x.path)
        if self.cache is not None and no_undefined_paths:
            paths = [Path(p) for p in x.path if p is not None]
            self.cache.populate_embed_cache(paths, x)
        return x

class FeatureExtractor(WaveformConditioner):
    """
    Feature Extractor used for the style conditioner of the paper AUDIO CONDITIONING
        FOR MUSIC GENERATION VIA DISCRETE BOTTLENECK FEATURES.

    Given a waveform, we extract an excerpt of defined length randomly subsampled.
        Then, we feed this excerpt to a feature extractor.

    Args:
        model_name (str): 'encodec' or 'mert'.
        sample_rate (str): sample rate of the input audio. (32000)
        encodec_checkpoint (str): if encodec is used as a feature extractor, checkpoint
            of the model. ('//pretrained/facebook/encodec_32khz' is the default)
        encodec_n_q (int): if encodec is used as a feature extractor it sets the number of
            quantization streams used in it.
        length (float): length in seconds of the random subsampled excerpt that is used
            for conditioning.
        dim (int): The internal representation dimension.
        output_dim (int): Output dimension for the conditioner.
        device (tp.Union[torch.device, str], optional): Device for the conditioner.
        compute_mask (bool): whether to mask the tokens corresponding to the subsampled
            excerpt in the computation of the music language model cross-entropy loss.
        use_middle_of_segment (bool): if True, always take the middle of the input
            instead of a random subsampled excerpt.
        ds_rate_compression (int): downsampling parameter of the compression model used
            for the music language model. (640 for encodec_32khz)
        num_codebooks_lm (int): the number of codebooks used by the music language model.
    """
    def __init__(
        self, model_name: str,
        sample_rate: int, encodec_checkpoint: str, encodec_n_q: int, length: float,
        dim: int, output_dim: int, device: tp.Union[torch.device, str],
        compute_mask: bool = True,
        use_middle_of_segment: bool = False, ds_rate_compression: int = 640,
        num_codebooks_lm: int = 4
    ):
        assert model_name in ['encodec', 'mert']
        if model_name == 'encodec':
            # from ..solvers.compression import CompressionSolver
            # feat_extractor = CompressionSolver.model_from_checkpoint(encodec_checkpoint, device)
            from .builders import get_compression_model
            encodec_state = torch.load(encodec_checkpoint, 'cpu')
            cfg = omegaconf.OmegaConf.create(encodec_state['xp.cfg'])
            cfg.device = device
            feat_extractor = get_compression_model(cfg).to(device)
            assert feat_extractor.sample_rate == cfg.sample_rate, "Compression model sample rate should match"
            assert 'best_state' in encodec_state and encodec_state['best_state'] != {}
            feat_extractor.load_state_dict(encodec_state['best_state'])
            feat_extractor.eval()
        elif model_name == 'mert':
            from transformers import AutoModel
            feat_extractor = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        super().__init__(
            dim=dim,
            output_dim=output_dim,
            device=device
        )
        self.sample_rate = sample_rate
        self.compute_mask = compute_mask
        self.feat_extractor: nn.Module
        self.embed: tp.Union[nn.ModuleList, nn.Linear]
        if model_name == 'encodec':
            self.__dict__["feat_extractor"] = feat_extractor.to(device)
            self.encodec_n_q = encodec_n_q
            self.embed = nn.ModuleList([nn.Embedding(feat_extractor.cardinality, dim) for _ in range(encodec_n_q)])
        if model_name == 'mert':
            self.__dict__["feat_extractor"] = feat_extractor.eval().to(device)
            self.embed = nn.Linear(768, dim)  # hardcoded
        self.length_subwav = int(length * sample_rate)
        self.ds_rate_compression = ds_rate_compression
        self.model_name = model_name
        self.use_middle_of_segment = use_middle_of_segment
        self.num_codebooks_lm = num_codebooks_lm

    def _get_wav_embedding(self, x: WavCondition) -> torch.Tensor:
        if x.wav.shape[-1] == 1:
            self.temp_mask = None
            return torch.zeros(x.wav.shape[0], 1, self.dim, device=self.device)
        else:
            with torch.no_grad():
                if self.use_middle_of_segment:
                    start = int((x.wav.shape[-1] - self.length_subwav) / 2)
                    self.start = start
                    wav = x.wav[:, :, start:start+self.length_subwav]
                else:
                    start = random.randint(0, x.wav.shape[-1] - self.length_subwav)
                    self.start = start
                    wav = x.wav[:, :, start:start+self.length_subwav]
                if self.compute_mask:
                    self.temp_mask = self._get_mask_wav(x, start)
                if self.model_name == 'encodec':
                    tokens = self.feat_extractor.encode(wav)[0]  # type: ignore [B,4,T]
                elif self.model_name == 'mert':
                    wav = convert_audio(wav, from_rate=x.sample_rate[0], to_rate=24000, to_channels=1)
                    embeds = self.feat_extractor(wav.squeeze(-2)).last_hidden_state
            if self.model_name == 'encodec':
                tokens = tokens[:, :self.encodec_n_q]
                embeds = sum([self.embed[k](tokens[:, k]) for k in range(self.encodec_n_q)])  # type: ignore
            else:
                embeds = self.embed(embeds)

            return embeds  # [B, T, dim]

    def _downsampling_factor(self):
        if self.model_name == 'encodec':
            return self.sample_rate / self.feat_extractor.frame_rate
        elif self.model_name == 'mert':
            return self.sample_rate / 75

    def _get_mask_wav(self, x: WavCondition, start: int) -> tp.Union[torch.Tensor, None]:
        if x.wav.shape[-1] == 1:
            return None
        total_length = int(x.wav.shape[-1] / self.ds_rate_compression)
        mask_length = int(self.length_subwav / self.ds_rate_compression)
        start = int(start / self.ds_rate_compression)
        mask = torch.ones(x.wav.shape[0], self.num_codebooks_lm,
                          total_length, device=self.device, dtype=torch.bool)
        mask[:, :, start:start+mask_length] = 0
        return mask

class StyleConditioner(FeatureExtractor):
    """Conditioner from the paper AUDIO CONDITIONING FOR MUSIC GENERATION VIA
    DISCRETE BOTTLENECK FEATURES.
    Given an audio input, it is passed through a Feature Extractor and a
    transformer encoder. Then it is quantized through RVQ.

    Args:
        transformer_scale (str): size of the transformer. See in the __init__ to have more infos.
        ds_factor (int): the downsampling factor applied to the representation after quantization.
        encodec_n_q (int): if encodec is used as a feature extractor it sets the number of
            quantization streams used in it.
        n_q_out (int): the number of quantization streams used for the RVQ. If increased, there
            is more information passing as a conditioning.
        eval_q (int): the number of quantization streams used for the RVQ at evaluation time.
        q_dropout (bool): if True, at training time, a random number of stream is sampled
            at each step in the interval [1, n_q_out].
        bins (int): the codebook size used for each quantization stream.
        varying_lengths (List[float]): list of the min and max duration in seconds for the
            randomly subsampled excerpt at training time. For each step a length is sampled
            in this interval.
        batch_norm (bool): use of batch normalization after the transformer. Stabilizes the
            training.
        rvq_threshold_ema_dead_code (float): threshold for dropping dead codes in the
            RVQ.
    """
    def __init__(self, transformer_scale: str = 'default', ds_factor: int = 15, encodec_n_q: int = 4,
                 n_q_out: int = 6, eval_q: int = 3, q_dropout: bool = True, bins: int = 1024,
                 varying_lengths: tp.List[float] = [1.5, 4.5],
                 batch_norm: bool = True, rvq_threshold_ema_dead_code: float = 0.1,
                 n_chroma: int = 12, radix2_exp: int = 14, match_len_on_eval: bool = False,
                 **kwargs):
        tr_args: tp.Dict[str, tp.Any]
        if transformer_scale == 'xsmall':
            tr_args = {'d_model': 256, 'num_heads': 8, 'num_layers': 4}
        elif transformer_scale == 'large':
            tr_args = {'d_model': 1024, 'num_heads': 16, 'num_layers': 24}
        elif transformer_scale == 'default':
            tr_args = {'d_model': 512, 'num_heads': 8, 'num_layers': 8}
        elif transformer_scale == 'none':
            tr_args = {'d_model': 512}
        tr_args.update({
            'memory_efficient': True, 'activation': 'gelu',
            'norm_first': True, 'causal': False, 'layer_scale': None,
            'bias_ff': False, 'bias_attn': False,
        })
        dim = tr_args['d_model']
        super().__init__(dim=dim, encodec_n_q=encodec_n_q, **kwargs)

        self.ds_factor = ds_factor
        if transformer_scale == 'none':
            self.transformer = None
        else:
            self.transformer = StreamingTransformer(dim_feedforward=int(4 * dim), **tr_args)
        self.n_q_out = n_q_out
        self.eval_q = eval_q
        self.rvq = None
        if n_q_out > 0:
            self.rvq = ResidualVectorQuantizer(dim, n_q=n_q_out, q_dropout=q_dropout, bins=bins,
                                               threshold_ema_dead_code=rvq_threshold_ema_dead_code)
        self.autocast = TorchAutocast(enabled=self.device != 'cpu', device_type=self.device, dtype=torch.float32)
        self.varying_lengths = varying_lengths
        self.batch_norm = None
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(dim, affine=False)
        self.mask = None
        
        # chroma params
        # from demucs import pretrained
        # self.match_len_on_eval = match_len_on_eval
        # if match_len_on_eval:
        #     self._use_masking = False
        # self.__dict__['demucs'] = pretrained.get_model('htdemucs').to(self.device)
        # stem_sources: list = self.demucs.sources  # type: ignore
        # self.stem_indices = torch.LongTensor([stem_sources.index('vocals'), stem_sources.index('other')]).to(self.device)
        # self.chroma = ChromaExtractor(sample_rate=self.sample_rate, n_chroma=n_chroma,
        #                               radix2_exp=radix2_exp).to(self.device)
        # self.chroma_proj = nn.Linear(n_chroma, self.output_dim)

    @torch.no_grad()
    def _extract_chroma(self, wav: torch.Tensor) -> torch.Tensor:
        """Extract chroma features from the waveform."""
        with self.autocast:
            return self.chroma(wav)

    @torch.no_grad()
    def _get_stemmed_wav(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Get parts of the wav that holds the melody, extracting the main stems from the wav."""
        from demucs.apply import apply_model
        from demucs.audio import convert_audio
        with self.autocast:
            wav = convert_audio(
                wav, sample_rate, self.demucs.samplerate, self.demucs.audio_channels)  # type: ignore
            stems = apply_model(self.demucs, wav, device=self.device)  # type: ignore
            stems = stems[:, self.stem_indices]  # extract relevant stems for melody conditioning
            mix_wav = stems.sum(1)  # merge extracted stems to single waveform
            mix_wav = convert_audio(mix_wav, self.demucs.samplerate, self.sample_rate, 1)  # type: ignore
            return mix_wav
    @torch.no_grad()
    def _compute_chroma_wav_embedding(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Compute wav embedding, applying stem and chroma extraction."""
        # avoid 0-size tensors when we are working with null conds
        if wav.shape[-1] == 1:
            return self._extract_chroma(wav)
        stems = self._get_stemmed_wav(wav, sample_rate)
        chroma = self._extract_chroma(stems)
        return chroma

    def _get_wav_embedding(self, wav: WavCondition) -> torch.Tensor:
        with self.autocast:
            # Sample the length of the excerpts
            if self.varying_lengths and self.training:
                assert len(self.varying_lengths) == 2
                length = random.uniform(self.varying_lengths[0], self.varying_lengths[1])
                self.length_subwav = int(length * self.sample_rate)
            z1 = super()._get_wav_embedding(wav)
            if self.compute_mask:
                self.mask = self.temp_mask  # type: ignore
            self.temp_mask = None
            if self.transformer is not None:
                out1 = self.transformer(z1)
            else:
                out1 = z1
            if self.batch_norm:
                out1 = self.batch_norm(out1.transpose(1, 2)).transpose(1, 2)
            # Apply quantization
            if self.rvq:
                if self.training:
                    self.rvq.set_num_codebooks(self.n_q_out)
                else:
                    self.rvq.set_num_codebooks(self.eval_q)
                out1 = self.rvq(out1.transpose(1, 2), frame_rate=1.)
                if self.training:
                    flashy.distrib.average_tensors(self.rvq.buffers())
                out1 = out1.x.transpose(1, 2)
            # Apply fix downsample
            out1 = out1[:, ::self.ds_factor]

        return out1

    def set_params(self, eval_q: int = 3,
                   excerpt_length: float = 3.0,
                   ds_factor: tp.Optional[int] = None, encodec_n_q: tp.Optional[int] = None):
        """Modify the parameters of the SSL or introduce new parameters to add noise to
        the conditioning or to downsample it

        Args:
            eval_q (int): number of codebooks used when evaluating the model
            excerpt_length (float): the length of the excerpts used to condition the model
        """
        self.eval_q = eval_q
        self.length_subwav = int(excerpt_length * self.sample_rate)
        if ds_factor is not None:
            self.ds_factor = ds_factor
        if encodec_n_q is not None:
            self.encodec_n_q = encodec_n_q

    def _downsampling_factor(self):
        df = super()._downsampling_factor()
        return df * self.ds_factor

    def forward(self, x: WavCondition) -> ConditionType:
        wav, lengths, *_ = x

        embeds = self._get_wav_embedding(x)
        embeds = embeds.to(self.output_proj.weight)
        embeds = self.output_proj(embeds)

        lengths = lengths / self._downsampling_factor()
        mask = length_to_mask(lengths, max_len=embeds.shape[1]).int()  # type: ignore

        embeds = (embeds * mask.unsqueeze(2).to(self.device))

        # if x.wav.shape[-1] != 1:
        #     with torch.no_grad():
        #         assert all(sr == x.sample_rate[0] for sr in x.sample_rate), "All sample rates in batch should be equal."
        #         chroma_wav = x.wav[:, :,  self.start:self.start+self.length_subwav]
        #         chroma_embeds = self._compute_chroma_wav_embedding(chroma_wav, x.sample_rate[0])
        #         chroma_embeds = self.chroma_proj(chroma_embeds)
        #         embeds = torch.cat([embeds, chroma_embeds], dim=1)
        #         print("embeds: ", chroma_embeds.shape)

        return embeds, mask

def generate_random_pairs(B, T):
    # Generate two lists of random integers
    random_numbers = (torch.rand(B, 2) * T).floor().type(torch.int64)
    # Sort each pair to ensure the first number is less than the second
    sorted_numbers, _ = torch.sort(random_numbers, dim=1)
    return sorted_numbers

class BeatConditioner(BaseConditioner):
    """Beat conditioning supporting beat conditioning.

    Args:
        dim (int): Dimension.
        output_dim (int): Output dimension.
        device (str): Device.
    """

    def __init__(self, output_dim: int, device: str, beat_channel: int = 1,
                 time_drop_out: float = 0.5, seed: int = 12, frame_rate: int = 50):
        super().__init__(dim=beat_channel, output_dim=output_dim // 4)
        self.device = device
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        self.time_dropout = time_drop_out
        self.frame_rate = frame_rate
    
    def forward(self, x: BeatCondition) -> ConditionType:            
        beat, lengths, *_ = x
        embeds = beat.to(self.output_proj.weight) # [B, 0 or sec*framerate, C]
        embeds = self.output_proj(embeds)

        if lengths is not None:
            mask = length_to_mask(lengths, max_len=embeds.shape[1]).int()  # type: ignore
        else:
            mask = torch.ones_like(embeds[..., 0])
            
        if mask.shape[-1] > 1:
            random_num = torch.rand(1, generator=self.rng).item()
            B, fs = mask.shape
            T = fs // self.frame_rate
            ranges = generate_random_pairs(B, T).to(self.device)
            ranges *= self.frame_rate
            if random_num < self.time_dropout // 2:
                mask = range_to_mask(mask, ranges)
            elif random_num < self.time_dropout:
                mask = range_to_mask(mask, ranges, mode="mask1")

        embeds = (embeds * mask.unsqueeze(-1).to(self.device))

        return embeds, mask
    
    def tokenize(self, x: BeatCondition) -> BeatCondition:
        """Apply BeatConditioner tokenization and populate cache if needed."""
        beat, length, path= x
        beat = beat.permute(0, 2, 1) # [B, T, C]
        x = BeatCondition(beat.to(self.device), length.to(self.device), path)
        return x

class IntensityConditioner(BaseConditioner):
    """Intensity conditioning supporting intensity conditioning.

    Args:
        dim (int): Dimension.
        output_dim (int): Output dimension.
        device (str): Device.
    """

    def __init__(self, output_dim: int, device: str, intensity_channel: int = 1,
                 time_drop_out: float = 0.5, seed: int = 123, frame_rate: int = 50):
        super().__init__(dim=intensity_channel, output_dim=output_dim // 4)
        self.device = device
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        self.time_dropout = time_drop_out
        self.frame_rate = frame_rate
    
    def forward(self, x: IntensityCondition) -> ConditionType:            
        intensity, lengths, *_ = x
        embeds = intensity.to(self.output_proj.weight) # [B, 0 or sec*framerate, C]
        embeds = self.output_proj(embeds)

        if lengths is not None:
            mask = length_to_mask(lengths, max_len=embeds.shape[1]).int()  # type: ignore
        else:
            mask = torch.ones_like(embeds[..., 0])
        
        if mask.shape[-1] > 1:
            random_num = torch.rand(1, generator=self.rng).item()
            B, fs = mask.shape
            T = fs // self.frame_rate
            ranges = generate_random_pairs(B, T).to(self.device)
            ranges *= self.frame_rate
            if random_num < self.time_dropout // 2:
                mask = range_to_mask(mask, ranges)
            elif random_num < self.time_dropout:
                mask = range_to_mask(mask, ranges, mode="mask1")

        embeds = (embeds * mask.unsqueeze(-1).to(self.device))

        return embeds, mask
    
    def tokenize(self, x: IntensityCondition) -> IntensityCondition:
        """Apply IntensityConditioner tokenization and populate cache if needed."""
        intensity, length, path = x
        intensity = intensity.permute(0, 2, 1) # [B, T, 1]
        x = IntensityCondition(intensity.to(self.device), length.to(self.device), path)
        return x

class MelodyConditioner(BaseConditioner):
    """Melody conditioning supporting melody conditioning.

    Args:
        output_dim (int): Output dimension.
        device (str): Device.
    """

    def __init__(self, output_dim: int, device: str, melody_channel: int = 12,
                 time_drop_out: float = 0.5, seed: int = 123456, frame_rate: int = 50):
        super().__init__(dim=melody_channel, output_dim=output_dim // 4)
        self.device = device
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        self.time_dropout = time_drop_out
        self.frame_rate = frame_rate
    
    def forward(self, x: MelodyCondition) -> ConditionType:            
        melody, lengths, *_ = x
        embeds = melody.to(self.output_proj.weight) # [B, 0 or sec*framerate, C]
        embeds = self.output_proj(embeds)

        if lengths is not None:
            mask = length_to_mask(lengths, max_len=embeds.shape[1]).int()  # type: ignore
        else:
            mask = torch.ones_like(embeds[..., 0])
        
        if mask.shape[-1] > 1:
            random_num = torch.rand(1, generator=self.rng).item()
            B, fs = mask.shape
            T = fs // self.frame_rate
            ranges = generate_random_pairs(B, T).to(self.device)
            ranges *= self.frame_rate
            if random_num < self.time_dropout // 2:
                mask = range_to_mask(mask, ranges)
            elif random_num < self.time_dropout:
                mask = range_to_mask(mask, ranges, mode="mask1")

        embeds = (embeds * mask.unsqueeze(-1).to(self.device))

        return embeds, mask
    
    def tokenize(self, x: MelodyCondition) -> MelodyCondition:
        """Apply MelodyConditioner tokenization and populate cache if needed."""
        melody, length, path = x
        melody = melody.permute(0, 2, 1) # [B, T, 12]
        x = MelodyCondition(melody.to(self.device), length.to(self.device), path)
        return x

class VAConditioner(BaseConditioner):
    """VA conditioning supporting va conditioning.

    Args:
        output_dim (int): Output dimension.
        device (str): Device.
    """

    def __init__(self, output_dim: int, device: str, va_channel: int = 2,
                 time_drop_out: float = 0.5, seed: int = 1234567, frame_rate: int = 50):
        super().__init__(dim=va_channel, output_dim=output_dim // 4)
        self.device = device
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        self.time_dropout = time_drop_out
        self.frame_rate = frame_rate
    
    def forward(self, x: VACondition) -> ConditionType:            
        va, lengths, *_ = x
        embeds = va.to(self.output_proj.weight) # [B, 0 or sec*framerate, C]
        embeds = self.output_proj(embeds)

        if lengths is not None:
            mask = length_to_mask(lengths, max_len=embeds.shape[1]).int()  # type: ignore
        else:
            mask = torch.ones_like(embeds[..., 0])
        
        if mask.shape[-1] > 1:
            random_num = torch.rand(1, generator=self.rng).item()
            B, fs = mask.shape
            T = fs // self.frame_rate
            ranges = generate_random_pairs(B, T).to(self.device)
            ranges *= self.frame_rate
            if random_num < self.time_dropout // 2:
                mask = range_to_mask(mask, ranges)
            elif random_num < self.time_dropout:
                mask = range_to_mask(mask, ranges, mode="mask1")

        embeds = (embeds * mask.unsqueeze(-1).to(self.device))

        return embeds, mask
    
    def tokenize(self, x: VACondition) -> VACondition:
        """Apply VAConditioner tokenization and populate cache if needed."""
        va, length, path = x
        va = va.permute(0, 2, 1) # [B, T, 2]
        x = VACondition(va.to(self.device), length.to(self.device), path)
        return x

def _drop_video_condition(conditions: tp.List[ConditioningAttributes]) -> tp.List[ConditioningAttributes]:
    """Drop the video condition but keep the wav conditon on a list of ConditioningAttributes.
    This is useful to calculate l_style in the double classifier free guidance formula.
    Influenced by paragraph 4.3 in https://arxiv.org/pdf/2407.12563

    Args:
        conditions (tp.List[ConditioningAttributes]): List of conditions.
    """
    # We assert that video and self_wav are in the conditions
    for condition in conditions:
        assert 'visual_content' in condition.video.keys()
        assert 'self_wav' in condition.wav.keys()
    return AttributeDropout(p={'video': {'visual_content': 1.0},
                               'wav': {'self_wav': 0.0}})(conditions)

def nullify_condition(condition: ConditionType, dim: int = 1):
    """Transform an input condition to a null condition.
    The way it is done by converting it to a single zero vector.

    Args:
        condition (ConditionType): A tuple of condition and mask (tuple[torch.Tensor, torch.Tensor])
        dim (int): The dimension that will be truncated (should be the time dimension)
        WARNING!: dim should not be the batch dimension!
    Returns:
        ConditionType: A tuple of null condition and mask
    """
    assert dim != 0, "dim cannot be the batch dimension!"
    assert isinstance(condition, tuple) and \
        isinstance(condition[0], torch.Tensor) and \
        isinstance(condition[1], torch.Tensor), "'nullify_condition' got an unexpected input type!"
    cond, mask = condition
    B = cond.shape[0]
    last_dim = cond.dim() - 1
    out = cond.transpose(dim, last_dim)
    out = 0. * out[..., :1]
    out = out.transpose(dim, last_dim)
    mask = torch.zeros((B, 1), device=out.device).int()
    assert cond.dim() == out.dim()
    return out, mask

def nullify_clipcls(cond: ClsCondition) -> ClsCondition:
    """
    Args:
        cond (ClsCondition): Cls condition with clipcls, tensor of shape [B, T, C].
    Returns:
        ClsCondition: Nullified cls condition.
    """
    null_cls, _ = nullify_condition((cond.clipcls, torch.zeros_like(cond.clipcls)), dim=cond.clipcls.dim() - 2)
    return ClsCondition(
        clipcls=null_cls,
        length=torch.tensor([0] * cond.clipcls.shape[0], device=cond.clipcls.device),
        path=[None] * cond.clipcls.shape[0],
    )

def nullify_wav(cond: WavCondition) -> WavCondition:
    """Transform a WavCondition to a nullified WavCondition.
    It replaces the wav by a null tensor, forces its length to 0, and replaces metadata by dummy attributes.

    Args:
        cond (WavCondition): Wav condition with wav, tensor of shape [B, T].
    Returns:
        WavCondition: Nullified wav condition.
    """
    null_wav, _ = nullify_condition((cond.wav, torch.zeros_like(cond.wav)), dim=cond.wav.dim() - 1)
    return WavCondition(
        wav=null_wav,
        length=torch.tensor([0] * cond.wav.shape[0], device=cond.wav.device),
        sample_rate=cond.sample_rate,
        path=[None] * cond.wav.shape[0],
        seek_time=[None] * cond.wav.shape[0],
    )

def nullify_beat(cond: BeatCondition) -> BeatCondition:
    """
    Args:
        cond (BeatCondition): Beat condition with beat, tensor of shape [B, C, T].
    Returns:
        BeatCondition: Nullified beat condition.
    """
    null_beat, _ = nullify_condition((cond.beat, torch.zeros_like(cond.beat)), dim=cond.beat.dim() - 1)
    return BeatCondition(
        beat=null_beat,
        length=torch.tensor([0] * cond.beat.shape[0], device=cond.beat.device),
        path=[None] * cond.beat.shape[0],
    )
    
def nullify_intensity(cond: IntensityCondition) -> IntensityCondition:
    """
    Args:
        cond (IntensityCondition): Intensity condition with intensity, tensor of shape [B, C, T].
    Returns:
        IntensityCondition: Nullified intensity condition.
    """
    null_intensity, _ = nullify_condition((cond.intensity, torch.zeros_like(cond.intensity)), dim=cond.intensity.dim() - 1)
    return IntensityCondition(
        intensity=null_intensity,
        length=torch.tensor([0] * cond.intensity.shape[0], device=cond.intensity.device),
        path=[None] * cond.intensity.shape[0],
    )
    
def nullify_melody(cond: MelodyCondition) -> MelodyCondition:
    """
    Args:
        cond (MelodyCondition): Melody condition with melody, tensor of shape [B, C, T].
    Returns:
        MelodyCondition: Nullified melody condition.
    """
    null_melody, _ = nullify_condition((cond.melody, torch.zeros_like(cond.melody)), dim=cond.melody.dim() - 1)
    return MelodyCondition(
        melody=null_melody,
        length=torch.tensor([0] * cond.melody.shape[0], device=cond.melody.device),
        path=[None] * cond.melody.shape[0],
    )

def nullify_va(cond: VACondition) -> VACondition:
    """
    Args:
        cond (VACondition): VA condition with va, tensor of shape [B, C, T].
    Returns:
        VACondition: Nullified va condition.
    """
    null_va, _ = nullify_condition((cond.va, torch.zeros_like(cond.va)), dim=cond.va.dim() - 1)
    return VACondition(
        va=null_va,
        length=torch.tensor([0] * cond.va.shape[0], device=cond.va.device),
        path=[None] * cond.va.shape[0],
    )

def dropout_condition(sample: ConditioningAttributes, condition_type: str, condition: str) -> ConditioningAttributes:
    """Utility function for nullifying an attribute inside an ConditioningAttributes object.
    If the condition is of type "wav", then nullify it using `nullify_condition` function.
    If the condition is of any other type, set its value to None.
    Works in-place.
    """
    if condition_type not in ['video', 'wav', 'beat', 'intensity', 'melody', 'va']:
        raise ValueError(
            "dropout_condition got an unexpected condition type!"
            f" expected 'video', 'wav', 'beat', 'intensity', 'va' or 'melody' but got '{condition_type}'"
        )

    if condition not in getattr(sample, condition_type):
        raise ValueError(
            "dropout_condition received an unexpected condition!"
            f" expected wav={sample.wav.keys()}, video={sample.video.keys()}, beat={sample.beat.keys()}, intensity={sample.intensity.keys()}, melody={sample.melody.keys()}, va={sample.va.keys()}"
            f" but got '{condition}' of type '{condition_type}'!"
        )

    if condition_type == 'wav':
        wav_cond = sample.wav[condition]
        sample.wav[condition] = nullify_wav(wav_cond)
    elif condition_type == 'beat':
        beat_cond = sample.beat[condition]
        sample.beat[condition] = nullify_beat(beat_cond)
    elif condition_type == 'intensity':
        intensity_cond = sample.intensity[condition]
        sample.intensity[condition] = nullify_intensity(intensity_cond)
    elif condition_type == 'video':
        video_cond = sample.video[condition]
        sample.video[condition] = nullify_clipcls(video_cond)
    elif condition_type == 'melody':
        melody_cond = sample.melody[condition]
        sample.melody[condition] = nullify_melody(melody_cond)
    elif condition_type == 'va':
        va_cond = sample.va[condition]
        sample.va[condition] = nullify_va(va_cond)

    return sample

class DropoutModule(nn.Module):
    """Base module for all dropout modules."""
    def __init__(self, seed: int = 1234):
        super().__init__()
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

class AttributeDropout(DropoutModule):
    """Dropout with a given probability per attribute.
    This is different from the behavior of ClassifierFreeGuidanceDropout as this allows for attributes
    to be dropped out separately. For example, "artist" can be dropped while "genre" remains.
    This is in contrast to ClassifierFreeGuidanceDropout where if "artist" is dropped "genre"
    must also be dropped.

    Args:
        p (tp.Dict[str, float]): A dict mapping between attributes and dropout probability. For example:
            ...
            "intensity": 0.1,
            "beat": 0.5,
            "wav": 0.25,
            ...
        active_on_eval (bool, optional): Whether the dropout is active at eval. Default to False.
        seed (int, optional): Random seed.
    """
    def __init__(self, p: tp.Dict[str, tp.Dict[str, float]], active_on_eval: bool = False, seed: int = 1234):
        super().__init__(seed=seed)
        self.active_on_eval = active_on_eval
        # construct dict that return the values from p otherwise 0
        self.p = {}
        for condition_type, probs in p.items():
            self.p[condition_type] = defaultdict(lambda: 0, probs)

    def forward(self, samples: tp.List[ConditioningAttributes]) -> tp.List[ConditioningAttributes]:
        """
        Args:
            samples (list[ConditioningAttributes]): List of conditions.
        Returns:
            list[ConditioningAttributes]: List of conditions after certain attributes were set to None.
        """
        if not self.training and not self.active_on_eval:
            return samples

        samples = deepcopy(samples)
        for condition_type, ps in self.p.items():  # for condition types [video, wav]
            for condition, p in ps.items():  # for attributes of each type (e.g., [artist, genre])
                if torch.rand(1, generator=self.rng).item() < p:
                    for sample in samples:
                        dropout_condition(sample, condition_type, condition)
        return samples

    def __repr__(self):
        return f"AttributeDropout({dict(self.p)})"

class ClassifierFreeGuidanceDropout(DropoutModule):
    """Classifier Free Guidance dropout.
    All attributes are dropped with the same probability.

    Args:
        p (float): Probability to apply condition dropout during training.
        seed (int): Random seed.
    """
    def __init__(self, p: float, seed: int = 1234):
        super().__init__(seed=seed)
        self.p = p

    def forward(self, samples: tp.List[ConditioningAttributes]) -> tp.List[ConditioningAttributes]:
        """
        Args:
            samples (list[ConditioningAttributes]): List of conditions.
        Returns:
            list[ConditioningAttributes]: List of conditions after all attributes were set to None.
        """
        if not self.training:
            return samples

        # decide on which attributes to drop in a batched fashion
        drop = torch.rand(1, generator=self.rng).item() < self.p
        if not drop:
            return samples

        # nullify conditions of all attributes
        samples = deepcopy(samples)
        for condition_type in ["wav", "video"]:
            for sample in samples:
                for condition in sample.attributes[condition_type]:
                    dropout_condition(sample, condition_type, condition)
        return samples

    def __repr__(self):
        return f"ClassifierFreeGuidanceDropout(p={self.p})"

class ConditioningProvider(nn.Module):
    """Prepare and provide conditions given all the supported conditioners.

    Args:
        conditioners (dict): Dictionary of conditioners.
        device (torch.device or str, optional): Device for conditioners and output condition types.
    """
    def __init__(self, conditioners: tp.Dict[str, BaseConditioner], device: tp.Union[torch.device, str] = "cpu"):
        super().__init__()
        self.device = device
        self.conditioners = nn.ModuleDict(conditioners)

    @property
    def video_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, VideoConditioner)]
    
    @property
    def finegrained_content_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, FineGrainFusionConditioner)]

    @property
    def wav_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, WaveformConditioner)]
    
    @property
    def intensity_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, IntensityConditioner)]
    
    @property
    def beat_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, BeatConditioner)]
    
    @property
    def melody_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, MelodyConditioner)]
    
    @property
    def va_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, VAConditioner)]

    @property
    def has_wav_condition(self):
        return len(self.wav_conditions) > 0

    def tokenize(self, inputs: tp.List[ConditioningAttributes]) -> tp.Dict[str, tp.Any]:
        """Match attributes/wavs with existing conditioners in self, and compute tokenize them accordingly.
        This should be called before starting any real GPU work to avoid synchronization points.
        This will return a dict matching conditioner names to their arbitrary tokenized representations.

        Args:
            inputs (list[ConditioningAttributes]): List of ConditioningAttributes objects containing
                video, wav, beat, intensity, melody conditions.
        """
        assert all([isinstance(x, ConditioningAttributes) for x in inputs]), (
            "Got unexpected types input for conditioner! should be tp.List[ConditioningAttributes]",
            f" but types were {set([type(x) for x in inputs])}"
        )

        output = {}
        video = self._collate_video(inputs)
        wavs = self._collate_wavs(inputs)
        beats = self._collate_beats(inputs)
        intensity = self._collate_intensity(inputs)
        melody = self._collate_melody(inputs)
        va = self._collate_va(inputs)
        finegrained_content = self._collate_finegrained_content(inputs)
        
        assert set(video.keys() | wavs.keys() | beats.keys() | intensity.keys() | finegrained_content.keys() | melody.keys() | va.keys()).issubset(set(self.conditioners.keys())), (
            f"Got an unexpected attribute! Expected {self.conditioners.keys()}, ",
            f"got {video.keys(), wavs.keys(), beats.keys(), intensity.keys(), finegrained_content.keys(), melody.keys(), va.keys()}"
        )
        
        for attribute, batch in chain(video.items(), wavs.items(), beats.items(), intensity.items(), finegrained_content.items(), melody.items(), va.items()):
            output[attribute] = self.conditioners[attribute].tokenize(batch)
        return output

    def forward(self, tokenized: tp.Dict[str, tp.Any]) -> tp.Dict[str, ConditionType]:
        """Compute pairs of `(embedding, mask)` using the configured conditioners and the tokenized representations.
        The output is for example:
        {
            "genre": (torch.Tensor([B, 1, D_genre]), torch.Tensor([B, 1])),
            "video": (torch.Tensor([B, T, D_desc]), torch.Tensor([B, T])),
            ...
        }

        Args:
            tokenized (dict): Dict of tokenized representations as returned by `tokenize()`.
        """
        output = {}
        for attribute, inputs in tokenized.items():
            condition, mask = self.conditioners[attribute](inputs)
            output[attribute] = (condition, mask)
        return output
    
    def _collate_video(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, ClsCondition]:
        """Given a list of ConditioningAttributes objects, compile a dictionary where the keys
        are the attributes and the values are the aggregated input per attribute.
        Args:
            samples (list of ConditioningAttributes): List of ConditioningAttributes samples.
        Returns:
            dict[str, list[str, optional]]: A dictionary mapping an attribute name to video batch.
        """
        clipcls = defaultdict(list)
        lengths = defaultdict(list)
        paths = defaultdict(list)
        out: tp.Dict[str, ClsCondition] = {}

        for sample in samples:
            for attribute in self.video_conditions:
                visual_content, length, path = sample.video[attribute]
                assert visual_content.dim() == 3, f"Got visual_content with dim={visual_content.dim()}, but expected 3 [1, T, C]"
                assert visual_content.size(0) == 1, f"Got visual_content [B, T, C] with shape={visual_content.shape}, but expected B == 1"
                clipcls[attribute].append(visual_content.squeeze(0))  # [1, T, C] -> [N * [T, C]]
                lengths[attribute].append(length.to(self.device)) # [N, 1]
                paths[attribute].extend(path) # [N]

        # stack all cls to a single tensor
        for attribute in self.video_conditions:
            stacked_cls, _ = collate(clipcls[attribute], dim=0) # tensor padded here
            out[attribute] = ClsCondition(
                stacked_cls, torch.cat(lengths[attribute]), paths[attribute])
        return out
    
    def _collate_finegrained_content(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, FineGrainCondition]:
        """Given a list of ConditioningAttributes objects, compile a dictionary where the keys
        are the attributes and the values are the aggregated input per attribute.
        Args:
            samples (list of ConditioningAttributes): List of ConditioningAttributes samples.
        Returns:
            dict[str, list[str, optional]]: A dictionary mapping an attribute name to video batch.
        """
        cls = defaultdict(list)
        images = defaultdict(list)
        maes = defaultdict(list)
        lengths = defaultdict(list)
        paths = defaultdict(list)
        out: tp.Dict[str, FineGrainCondition] = {}

        for sample in samples:
            for attribute in self.finegrained_content_conditions:
                image, visual_content, mae, length, path = sample.finegrained_content[attribute]
                cls[attribute].append(visual_content.squeeze(0))
                images[attribute].append(image.squeeze(0))
                maes[attribute].append(mae.squeeze(0))
                lengths[attribute].append(length.to(self.device)) # [N, 1]
                paths[attribute].extend(path) # [N]

        # stack all cls to a single tensor
        for attribute in self.finegrained_content_conditions:
            stacked_cls, _ = collate(cls[attribute], dim=0) # tensor padded here
            stacked_image, _ = collate(images[attribute], dim=0)
            stacked_mae, _ = collate(maes[attribute], dim=0)
            out[attribute] = FineGrainCondition(
                stacked_image, stacked_cls, stacked_mae, torch.cat(lengths[attribute]), paths[attribute])
        return out

    def _collate_wavs(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, WavCondition]:
        """Generate a dict where the keys are attributes by which we fetch similar wavs,
        and the values are Tensors of wavs according to said attributes.

        *Note*: by the time the samples reach this function, each sample should have some waveform
        inside the "wav" attribute. It should be either:
        1. A real waveform
        2. A null waveform due to the sample having no similar waveforms (nullified by the dataset)
        3. A null waveform due to it being dropped in a dropout module (nullified by dropout)

        Args:
            samples (list of ConditioningAttributes): List of ConditioningAttributes samples.
        Returns:
            dict[str, WavCondition]: A dictionary mapping an attribute name to wavs.
        """
        wavs = defaultdict(list)
        lengths = defaultdict(list)
        sample_rates = defaultdict(list)
        paths = defaultdict(list)
        seek_times = defaultdict(list)
        out: tp.Dict[str, WavCondition] = {}

        for sample in samples:
            for attribute in self.wav_conditions:
                wav, length, sample_rate, path, seek_time = sample.wav[attribute]
                assert wav.dim() == 3, f"Got wav with dim={wav.dim()}, but expected 3 [1, C, T]"
                assert wav.size(0) == 1, f"Got wav [B, C, T] with shape={wav.shape}, but expected B == 1"
                # mono-channel conditioning
                wav = wav.mean(1, keepdim=True)  # [1, 1, T]
                wavs[attribute].append(wav.flatten())  # [T]
                lengths[attribute].append(length.to(self.device))
                sample_rates[attribute].extend(sample_rate)
                paths[attribute].extend(path)
                seek_times[attribute].extend(seek_time)

        # stack all wavs to a single tensor
        for attribute in self.wav_conditions:
            stacked_wav, _ = collate(wavs[attribute], dim=0)
            out[attribute] = WavCondition(
                stacked_wav.unsqueeze(1), torch.cat(lengths[attribute]), sample_rates[attribute],
                paths[attribute], seek_times[attribute])

        return out

    def _collate_intensity(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, IntensityCondition]:
        """Generate a dict where the keys are attributes by which we fetch similar wavs,
        and the values are Tensors of wavs according to said attributes.

        Args:
            samples (list of ConditioningAttributes): List of ConditioningAttributes samples.
        Returns:
            dict[str, WavCondition]: A dictionary mapping an attribute name to wavs.
        """
        intensities = defaultdict(list)
        lengths = defaultdict(list)
        paths = defaultdict(list)
        out: tp.Dict[str, IntensityCondition] = {}

        for sample in samples:
            for attribute in self.intensity_conditions:
                intensity, length, path = sample.intensity[attribute]
                assert intensity.dim() == 3, f"Got intensity with dim={intensity.dim()}, but expected 3 [1, C, T]"
                assert intensity.size(0) == 1, f"Got intensity [B, C, T] with shape={intensity.shape}, but expected B == 1"
                intensities[attribute].append(intensity.squeeze(0))  # [1, C, T] -> [N * [C, T]]
                lengths[attribute].append(length.to(self.device)) # [N, 1]
                paths[attribute].extend(path) # [N]

        # stack all intensities to a single tensor
        for attribute in self.intensity_conditions:
            stacked_intensity, _ = collate(intensities[attribute], dim=1) # tensor padded here
            out[attribute] = IntensityCondition(
                stacked_intensity, torch.cat(lengths[attribute]), paths[attribute])
        return out

    def _collate_beats(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, BeatCondition]:
        """Generate a dict where the keys are attributes by which we fetch similar wavs,
        and the values are Tensors of wavs according to said attributes.

        Args:
            samples (list of ConditioningAttributes): List of ConditioningAttributes samples.
        Returns:
            dict[str, WavCondition]: A dictionary mapping an attribute name to wavs.
        """
        beats = defaultdict(list)
        lengths = defaultdict(list)
        paths = defaultdict(list)
        out: tp.Dict[str, BeatCondition] = {}

        for sample in samples:
            for attribute in self.beat_conditions:
                beat, length, path = sample.beat[attribute]
                assert beat.dim() == 3, f"Got beat with dim={beat.dim()}, but expected 3 [1, C, T]"
                assert beat.size(0) == 1, f"Got beat [B, C, T] with shape={beat.shape}, but expected B == 1"
                beats[attribute].append(beat.squeeze(0))  # [1, C, T] -> [N * [C, T]]
                lengths[attribute].append(length.to(self.device)) # [N, 1]
                paths[attribute].extend(path) # [N]

        # stack all beats to a single tensor
        for attribute in self.beat_conditions:
            stacked_beat, _ = collate(beats[attribute], dim=1) # tensor padded here
            out[attribute] = BeatCondition(
                stacked_beat, torch.cat(lengths[attribute]), paths[attribute])
        return out
    
    def _collate_melody(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, MelodyCondition]:
        """Generate a dict where the keys are attributes by which we fetch similar wavs,
        and the values are Tensors of wavs according to said attributes.

        Args:
            samples (list of ConditioningAttributes): List of ConditioningAttributes samples.
        Returns:
            dict[str, WavCondition]: A dictionary mapping an attribute name to wavs.
        """
        melodies = defaultdict(list)
        lengths = defaultdict(list)
        paths = defaultdict(list)
        out: tp.Dict[str, MelodyCondition] = {}

        for sample in samples:
            for attribute in self.melody_conditions:
                melody, length, path = sample.melody[attribute]
                assert melody.dim() == 3, f"Got melody with dim={melody.dim()}, but expected 3 [1, C, T]"
                assert melody.size(0) == 1, f"Got melody [B, C, T] with shape={melody.shape}, but expected B == 1"
                melodies[attribute].append(melody.squeeze(0))  # [1, C, T] -> [N * [C, T]]
                lengths[attribute].append(length.to(self.device)) # [N, 1]
                paths[attribute].extend(path) # [N]

        # stack all melodies to a single tensor
        for attribute in self.melody_conditions:
            stacked_melody, _ = collate(melodies[attribute], dim=1) # tensor padded here
            out[attribute] = MelodyCondition(
                stacked_melody, torch.cat(lengths[attribute]), paths[attribute])
        return out

    def _collate_va(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, VACondition]:
        """Generate a dict where the keys are attributes by which we fetch similar wavs,
        and the values are Tensors of wavs according to said attributes.

        Args:
            samples (list of ConditioningAttributes): List of ConditioningAttributes samples.
        Returns:
            dict[str, WavCondition]: A dictionary mapping an attribute name to wavs.
        """
        vas = defaultdict(list)
        lengths = defaultdict(list)
        paths = defaultdict(list)
        out: tp.Dict[str, VACondition] = {}

        for sample in samples:
            for attribute in self.va_conditions:
                va, length, path = sample.va[attribute]
                assert va.dim() == 3, f"Got va with dim={va.dim()}, but expected 3 [1, C, T]"
                assert va.size(0) == 1, f"Got va [B, C, T] with shape={va.shape}, but expected B == 1"
                vas[attribute].append(va.squeeze(0))  # [1, C, T] -> [N * [C, T]]
                lengths[attribute].append(length.to(self.device)) # [N, 1]
                paths[attribute].extend(path) # [N]

        # stack all vas to a single tensor
        for attribute in self.va_conditions:
            stacked_va, _ = collate(vas[attribute], dim=1) # tensor padded here
            out[attribute] = VACondition(
                stacked_va, torch.cat(lengths[attribute]), paths[attribute])
        return out

class ConditionFuser(StreamingModule):
    """Condition fuser handles the logic to combine the different conditions
    to the actual model input.

    Args:
        fuse2cond (tp.Dict[str, str]): A dictionary that says how to fuse
            each condition. For example:
            {
                "prepend": ["visual_content"],
                "sum": ["beat", "intensity"],
                "cross": ["visual_content"],
            }
        cross_attention_pos_emb (bool, optional): Use positional embeddings in cross attention.
        cross_attention_pos_emb_scale (int): Scale for positional embeddings in cross attention if used.
    """
    FUSING_METHODS = ["sum", "prepend", "cross", "input_interpolate"]

    def __init__(self, fuse2cond: tp.Dict[str, tp.List[str]], cross_attention_pos_emb: bool = False,
                 cross_attention_pos_emb_scale: float = 1.0, in_atten: bool = False):
        super().__init__()
        assert all(
            [k in self.FUSING_METHODS for k in fuse2cond.keys()]
        ), f"Got invalid fuse method, allowed methods: {self.FUSING_METHODS}"
        self.cross_attention_pos_emb = cross_attention_pos_emb
        self.cross_attention_pos_emb_scale = cross_attention_pos_emb_scale
        self.fuse2cond: tp.Dict[str, tp.List[str]] = fuse2cond
        self.cond2fuse: tp.Dict[str, tp.List[str]] = {}
        self.in_atten = in_atten
        
        for fuse_method, conditions in fuse2cond.items():
            for condition in conditions:
                if not condition in self.cond2fuse.keys():
                    self.cond2fuse[condition] = [fuse_method]
                else:
                    self.cond2fuse[condition].append(fuse_method)

    def forward(
        self,
        input: torch.Tensor,
        conditions: tp.Dict[str, ConditionType]
    ) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """Fuse the conditions to the provided model input.

        Args:
            input (torch.Tensor): Transformer input.
            conditions (dict[str, ConditionType]): Dict of conditions.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: The first tensor is the transformer input
                after the conditions have been fused. The second output tensor is the tensor
                used for cross-attention or None if no cross attention inputs exist.
        """
        B, T, _ = input.shape
        if self.in_atten:
            in_atten_cond = torch.zeros_like(input)
            cond_sums = []
        else:
            in_atten_cond = None

        if 'offsets' in self._streaming_state:
            first_step = False
            offsets = self._streaming_state['offsets']
        else:
            first_step = True
            offsets = torch.zeros(input.shape[0], dtype=torch.long, device=input.device)

        assert set(conditions.keys()).issubset(set(self.cond2fuse.keys())), \
            f"given conditions contain unknown attributes for fuser, " \
            f"expected {self.cond2fuse.keys()}, got {conditions.keys()}"
        cross_attention_output = None
        
        for cond_type, (cond, cond_mask) in conditions.items():
            fuse_methods = self.cond2fuse[cond_type]
            for op in fuse_methods:
                if op == 'sum':
                    cond_sum = cond[:, offsets[0]:offsets[0]+T]
                    if cond_sum.shape[1] >= 0:
                        if cond_sum.shape[1] < T:
                            cond_sum = F.pad(cond_sum, (0, 0, 0, T-cond_sum.shape[1]), "constant", 0) # pad last special token dim
                        cond_sums.append(cond_sum)
                elif op == 'input_interpolate':
                    cond = einops.rearrange(cond, "b t d -> b d t")
                    cond = F.interpolate(cond, size=input.shape[1])
                    input += einops.rearrange(cond, "b d t -> b t d")
                elif op == 'prepend':
                    if first_step:
                        input = torch.cat([cond, input], dim=1)
                elif op == 'cross':
                    if cross_attention_output is not None:
                        cross_attention_output = torch.cat([cross_attention_output, cond], dim=1)
                    else:
                        cross_attention_output = cond
                else:
                    raise ValueError(f"unknown op ({op})")

        if self.in_atten:
            cond_sums = torch.cat(cond_sums, dim=2)
            in_atten_cond += cond_sums
        
        if self.cross_attention_pos_emb and cross_attention_output is not None:
            positions = torch.arange(
                cross_attention_output.shape[1],
                device=cross_attention_output.device
            ).view(1, -1, 1)
            pos_emb = create_sin_embedding(positions, cross_attention_output.shape[-1])
            cross_attention_output = cross_attention_output + self.cross_attention_pos_emb_scale * pos_emb

        if self._is_streaming:
            self._streaming_state['offsets'] = offsets + T

        return input, cross_attention_output, in_atten_cond

class MutiConditionFuser(nn.Module):
    FUSING_METHODS = ["sum"]
    MUTI_COND = ["intensity", "beat", "melody", "va"]
    def __init__(self, fuse2cond: tp.Dict[str, tp.List[str]], in_atten: bool = False, 
                 dim: int = 1024, num_blocks: int = 3, patch_length: int = 50):
        super().__init__()
        assert all(
            [k in self.FUSING_METHODS for k in fuse2cond.keys()]
        ), f"Got invalid fuse method, allowed methods: {self.FUSING_METHODS}"
        self.fuse2cond: tp.Dict[str, tp.List[str]] = fuse2cond
        self.cond2fuse: tp.Dict[str, tp.List[str]] = {}
        self.in_atten = in_atten
        self.patch_length = patch_length
        
        for fuse_method, conditions in fuse2cond.items():
            for condition in conditions:
                if not condition in self.cond2fuse.keys():
                    self.cond2fuse[condition] = [fuse_method]
                else:
                    self.cond2fuse[condition].append(fuse_method)

        if self.in_atten:
            self.dwnet = DWNet(dim, num_blocks, patch_length)
    
    def forward(self, conditions: tp.Dict[str, ConditionType]) -> tp.Dict[str, ConditionType]:
        max_T = max([cond_mask.shape[1] for (cond, cond_mask) in conditions.values()])
        if self.in_atten and max_T > 1:
            max_T = math.ceil(max_T / self.patch_length) * self.patch_length
            dims = [cond.shape[-1] for (cond_type, (cond, cond_mask)) in conditions.items() if cond_type in self.MUTI_COND]
            cond_sum = []
            conditions_tmp = []
            for cond_type, (cond, cond_mask) in conditions.items():
                if cond_type not in self.MUTI_COND:
                    continue
                fuse_methods = self.cond2fuse[cond_type]
                for op in fuse_methods:
                    if op == 'sum':
                        cond = F.pad(cond, (0, 0, 0, max_T-cond.shape[1]), "constant", 0)
                        cond_sum.append(cond)
                        conditions_tmp.append((cond_type, cond_mask))
            cond_sum = torch.cat(cond_sum, dim=2)
            cond_sum = self.dwnet(cond_sum)
            cond_sum = torch.split(cond_sum, dims, dim=2)
            for i, (cond_type, cond_mask) in enumerate(conditions_tmp):
                conditions[cond_type] = (cond_sum[i], cond_mask)
        return conditions
