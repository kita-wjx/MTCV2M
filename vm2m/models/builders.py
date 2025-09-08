# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
All the functions to build the relevant models and modules
from the Hydra config.
"""

import typing as tp

import omegaconf
import torch

from .. import quantization as qt

from .conditioners import (BaseConditioner, ChromaStemConditioner,
                            ConditioningProvider, StyleConditioner,
                            CLIPConditioner, ConditionFuser, MutiConditionFuser, FineGrainFusionConditioner,
                            BeatConditioner, IntensityConditioner, MelodyConditioner, VAConditioner)
from .codebooks_patterns import (CoarseFirstPattern,
                                CodebooksPatternProvider,
                                DelayedPatternProvider,
                                MusicLMPattern,
                                ParallelPatternProvider,
                                UnrolledPatternProvider)
from ..utils.utils import dict_from_config
from .encodec import (CompressionModel, EncodecModel)
from .seanet import SEANetEncoder, SEANetDecoder
from .lm import LMModel
from .lm_magnet import MagnetLMModel

def get_quantizer(
    quantizer: str, cfg: omegaconf.DictConfig, dimension: int
) -> qt.BaseQuantizer:
    klass = {"no_quant": qt.DummyQuantizer, "rvq": qt.ResidualVectorQuantizer}[
        quantizer
    ]
    kwargs = dict_from_config(getattr(cfg, quantizer))
    if quantizer != "no_quant":
        kwargs["dimension"] = dimension
    return klass(**kwargs)


def get_encodec_autoencoder(encoder_name: str, cfg: omegaconf.DictConfig):
    if encoder_name == "seanet":
        kwargs = dict_from_config(getattr(cfg, "seanet"))
        encoder_override_kwargs = kwargs.pop("encoder")
        decoder_override_kwargs = kwargs.pop("decoder")
        encoder_kwargs = {**kwargs, **encoder_override_kwargs}
        decoder_kwargs = {**kwargs, **decoder_override_kwargs}
        encoder = SEANetEncoder(**encoder_kwargs)
        decoder = SEANetDecoder(**decoder_kwargs)
        return encoder, decoder
    else:
        raise KeyError(f"Unexpected compression model {cfg.compression_model}")


def get_compression_model(cfg: omegaconf.DictConfig) -> CompressionModel:
    """Instantiate a compression model."""
    if cfg.compression_model == "encodec":
        kwargs = dict_from_config(getattr(cfg, "encodec"))
        encoder_name = kwargs.pop("autoencoder")
        quantizer_name = kwargs.pop("quantizer")
        encoder, decoder = get_encodec_autoencoder(encoder_name, cfg)
        quantizer = get_quantizer(quantizer_name, cfg, encoder.dimension)
        frame_rate = kwargs["sample_rate"] // encoder.hop_length
        renormalize = kwargs.pop("renormalize", False)
        # deprecated params
        kwargs.pop("renorm", None)
        return EncodecModel(
            encoder,
            decoder,
            quantizer,
            frame_rate=frame_rate,
            renormalize=renormalize,
            **kwargs,
        ).to(cfg.device)
    else:
        raise KeyError(f"Unexpected compression model {cfg.compression_model}")

def get_lm_model(cfg: omegaconf.DictConfig) -> LMModel:
    """Instantiate a transformer LM."""
    if cfg.lm_model in ["transformer_lm", "transformer_lm_magnet"]:
        kwargs = dict_from_config(getattr(cfg, "transformer_lm"))
        n_q = kwargs["n_q"]
        q_modeling = kwargs.pop("q_modeling", None)
        codebooks_pattern_cfg = getattr(cfg, "codebooks_pattern")
        attribute_dropout = dict_from_config(getattr(cfg, "attribute_dropout"))
        cls_free_guidance = dict_from_config(getattr(cfg, "classifier_free_guidance"))
        cfg_prob, cfg_coef = (
            cls_free_guidance["training_dropout"],
            cls_free_guidance["inference_coef"],
        )
        fuser = get_condition_fuser(cfg)
        muticondfuser = get_muticondfuser(cfg)
        condition_provider = get_conditioner_provider(kwargs["dim"], cfg).to(cfg.device)
        if len(fuser.fuse2cond["cross"]) > 0:  # enforce cross-att programmatically
            kwargs["cross_attention"] = True
        if codebooks_pattern_cfg.modeling is None:
            assert (
                q_modeling is not None
            ), "LM model should either have a codebook pattern defined or transformer_lm.q_modeling"
            codebooks_pattern_cfg = omegaconf.OmegaConf.create(
                {"modeling": q_modeling, "delay": {"delays": list(range(n_q))}}
            )
        
        # add
        if "finegrained_content" in fuser.fuse2cond["cross"]:
            kwargs["segment_cross_attention"] = True
            kwargs["cross_attention"] = False
        else:
            kwargs["segment_cross_attention"] = False
            kwargs["cross_attention"] = True

        pattern_provider = get_codebooks_pattern_provider(n_q, codebooks_pattern_cfg)
        lm_class = MagnetLMModel if cfg.lm_model == "transformer_lm_magnet" else LMModel
        return lm_class(
            pattern_provider=pattern_provider,
            condition_provider=condition_provider,
            fuser=fuser,
            cfg_dropout=cfg_prob,
            cfg_coef=cfg_coef,
            attribute_dropout=attribute_dropout,
            dtype=getattr(torch, cfg.dtype),
            device=cfg.device,
            muticondfuser=muticondfuser,
            **kwargs,
        ).to(cfg.device)
    else:
        raise KeyError(f"Unexpected LM model {cfg.lm_model}")

def get_conditioner_provider(
    output_dim: int, cfg: omegaconf.DictConfig
) -> ConditioningProvider:
    """Instantiate a conditioning model."""
    device = cfg.device
    duration = cfg.dataset.segment_duration
    cfg = getattr(cfg, "conditioners")
    dict_cfg = {} if cfg is None else dict_from_config(cfg)
    conditioners: tp.Dict[str, BaseConditioner] = {}
    condition_provider_args = dict_cfg.pop("args", {})
    condition_provider_args.pop("merge_text_conditions_p", None)
    condition_provider_args.pop("drop_desc_p", None)

    for cond, cond_cfg in dict_cfg.items():
        model_type = cond_cfg["model"]
        model_args = cond_cfg[model_type]
        if model_type == "clip":
            conditioners[str(cond)] = CLIPConditioner(
                output_dim=output_dim, device=device, **model_args
            )
        elif model_type == "chroma_stem":
            conditioners[str(cond)] = ChromaStemConditioner(
                output_dim=output_dim, duration=duration, device=device, **model_args
            )
        elif model_type == 'style':
            conditioners[str(cond)] = StyleConditioner(
                output_dim=output_dim,
                device=device,
                **model_args
            )
        elif model_type == 'fusionnet':
            conditioners[str(cond)] = FineGrainFusionConditioner(
                output_dim=output_dim,
                device=device,
                **model_args
            )
        elif model_type == 'beat':
            conditioners[str(cond)] = BeatConditioner(
                output_dim=output_dim,
                device=device,
                **model_args
            )
        elif model_type == 'intensity':
            conditioners[str(cond)] = IntensityConditioner(
                output_dim=output_dim,
                device=device,
                **model_args
            )
        elif model_type == 'melody':
            conditioners[str(cond)] = MelodyConditioner(
                output_dim=output_dim,
                device=device,
                **model_args
            )
        elif model_type == 'va':
            conditioners[str(cond)] = VAConditioner(
                output_dim=output_dim,
                device=device,
                **model_args
            )
        else:
            raise ValueError(f"Unrecognized conditioning model: {model_type}")
    conditioner = ConditioningProvider(
        conditioners, device=device, **condition_provider_args
    )
    return conditioner

def get_condition_fuser(cfg: omegaconf.DictConfig) -> ConditionFuser:
    """Instantiate a condition fuser object."""
    fuser_cfg = getattr(cfg, "fuser")
    fuser_methods = ["sum", "cross", "prepend", "input_interpolate"]
    fuse2cond = {k: fuser_cfg[k] for k in fuser_methods}
    kwargs = {k: v for k, v in fuser_cfg.items() if k not in fuser_methods}
    fuser = ConditionFuser(fuse2cond=fuse2cond, **kwargs)
    return fuser

def get_muticondfuser(cfg: omegaconf.DictConfig) -> MutiConditionFuser:
    """Instantiate a muticondfuser object."""
    try:
        fuser_cfg = getattr(cfg, "muticondfuser")
    except AttributeError:
        return None
    fuser_methods = ["sum"]
    fuse2cond = {k: fuser_cfg[k] for k in fuser_methods}
    kwargs = {k: v for k, v in fuser_cfg.items() if k not in fuser_methods}
    muticondfuser = MutiConditionFuser(fuse2cond=fuse2cond, **kwargs)
    return muticondfuser

def get_codebooks_pattern_provider(
    n_q: int, cfg: omegaconf.DictConfig
) -> CodebooksPatternProvider:
    """Instantiate a codebooks pattern provider object."""
    pattern_providers = {
        "parallel": ParallelPatternProvider,
        "delay": DelayedPatternProvider,
        "unroll": UnrolledPatternProvider,
        "coarse_first": CoarseFirstPattern,
        "musiclm": MusicLMPattern,
    }
    name = cfg.modeling
    kwargs = dict_from_config(cfg.get(name)) if hasattr(cfg, name) else {}
    klass = pattern_providers[name]
    return klass(n_q, **kwargs)

def get_wrapped_compression_model(
    compression_model: CompressionModel, cfg: omegaconf.DictConfig
) -> CompressionModel:
    if hasattr(cfg, "compression_model_n_q"):
        if cfg.compression_model_n_q is not None:
            compression_model.set_num_codebooks(cfg.compression_model_n_q)
    return compression_model
