# Controllable Video-to-Music Generation with Multiple Time-Varying Conditions

**This project has been accepted to ACMMM 2025! 🚀🚀🚀**

[![arXiv](https://img.shields.io/badge/arXiv-2507.20627-brightgreen.svg?style=flat-square)](https://arxiv.org/pdf/2507.20627)   [![githubio](https://img.shields.io/badge/GitHub.io-Project-blue?logo=Github&style=flat-square)](https://kita-wjx.github.io/MCV2M/)

**This is the official repository for "[Controllable Video-to-Music Generation with Multiple Time-Varying Conditions](https://arxiv.org/pdf/2507.20627)".**

## ✨ Abstract

Music enhances video narratives and emotions, driving demand for automatic video-to-music (V2M) generation. However, existing V2M methods relying solely on visual features or supplementary textual inputs generate music in a black-box manner, often failing to meet user expectations. To address this challenge, we propose a novel multi-condition guided V2M generation framework that incorporates multiple time-varying conditions for enhanced control over music generation. Our method uses a two-stage training strategy that enables learning of V2M fundamentals and audiovisual temporal synchronization while meeting users’ needs for multi-condition control. In the first stage, we introduce a fine-grained feature selection module and a progressive temporal alignment attention mechanism to ensure flexible feature alignment. For the second stage, we develop a dynamic conditional fusion module and a control-guided decoder module to integrate multiple conditions and accurately guide the music composition process. Extensive experiments demonstrate that our method outperforms existing V2M pipelines in both subjective and objective evaluations, significantly enhancing control and alignment with user expectations.

## ✨ Method
<p align="center">
  <img src="./static/images/Architecture.png" alt="method">
</p>

## 🛠️ Environment Setup

- Create Anaconda Environment:
  ```bash
  git clone https://github.com/kita-wjx/MTCV2M.git; cd MTCV2M
  ```
  
  ```bash
  conda create -n MTCV2M python=3.9
  conda activate MTCV2M
  pip install -r requirements.txt
  ```
- Install ffmpeg:
  
  ```bash
  sudo apt-get install ffmpeg
  # Or if you are using Anaconda or Miniconda
  conda install "ffmpeg<5" -c conda-forge
  ```

## 🔮 Pretrained Weights

- Please download the pretrained Audio Compression checkpoint [compression_state_dict.bin](https://huggingface.co/facebook/musicgen-medium/resolve/main/compression_state_dict.bin) to preprocess music data, put it into the directory `'./checkpoints'`.
  ```bash
  wget https://huggingface.co/facebook/musicgen-medium/resolve/main/compression_state_dict.bin -O checkpoints/compression_state_dict.bin
  ```

- We use a text-to-music generation model MusicGen as the pretrained model for our stage 1. You can download and place the pretrained weights into the appropriate folder by running:
  ```bash
  # musicgen-small
  wget https://huggingface.co/facebook/musicgen-small/resolve/main/state_dict.bin -O checkpoints/musicgen-small/state_dict.bin

  # musicgen-medium
  wget https://huggingface.co/facebook/musicgen-medium/resolve/main/state_dict.bin -O checkpoints/musicgen-medium/state_dict.bin
  ```

- If you prefer to train from scratch without using any pretrained model, simply remove the `--continue_from` argument from the training command.

- Our model weights will release soon.

## 🔥 Training

- Data preprocessing:
  
  ```bash
  # Data construction details to be released...
  # Examples in ./datasets
  ```
- Start training:

  ```bash
  # First stage: pre-training stage
  bash train_video.sh

  # Importing / Exporting models
  cd vm2m
  python load_model.py --checkpoint_path /path/checkpoint.th --output_path ../checkpoints/stage1/state_dict.bin

  # Second stage: fine-tuning stage
  cd ..
  bash train_finetune.sh
  ```

## 🎯 Infer

- We will release model weights and relevant codes soon.

## 🤗 Acknowledgement

- [Audiocraft](https://github.com/facebookresearch/audiocraft), [GVMGen](https://github.com/chouliuzuo/GVMGen), [VidMuse](https://github.com/ZeyueT/VidMuse): the codebase we built upon. Thanks for their wonderful works.

## 🚀 Citation

If you find our work useful, please consider citing:

```
@misc{wu2025controllablevideotomusicgenerationmultiple,
      title={Controllable Video-to-Music Generation with Multiple Time-Varying Conditions}, 
      author={Junxian Wu and Weitao You and Heda Zuo and Dengming Zhang and Pei Chen and Lingyun Sun},
      year={2025},
      eprint={2507.20627},
      archivePrefix={arXiv},
      primaryClass={cs.MM},
      url={https://arxiv.org/abs/2507.20627}, 
}
```
