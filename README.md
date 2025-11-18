<div align="center">

# StableTTS

Next-generation TTS model using flow-matching and DiT, inspired by [Stable Diffusion 3](https://stability.ai/news/stable-diffusion-3).

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [News](#news)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pretrained Models](#pretrained-models)
- [Usage](#usage)
  - [Inference](#inference)
  - [Training](#training)
  - [API Usage](#api-usage)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [Disclaimer](#disclaimer)

## Introduction

**StableTTS** is a fast and lightweight text-to-speech (TTS) model that combines flow-matching and Diffusion Transformer (DiT) architectures. As the first open-source TTS model to integrate these technologies, StableTTS offers high-quality speech synthesis with only 31M parameters.

### Key Highlights

- 🚀 **Fast Inference**: Efficient flow-matching decoder with configurable ODE solvers
- 🌍 **Multilingual Support**: Chinese, English, and Japanese in a single checkpoint
- 🎯 **High Quality**: CFG (Classifier-Free Guidance) support for improved audio quality
- 🔧 **Easy to Use**: Simple API and Gradio web interface
- 📦 **Lightweight**: Only 31M parameters, suitable for deployment

## Features

- **Flow-Matching Decoder**: Uses continuous normalizing flows for high-quality mel-spectrogram generation
- **Diffusion Transformer (DiT)**: Leverages transformer architecture with U-Net-like skip connections
- **Reference Audio**: Voice cloning using reference audio embeddings
- **CFG Support**: Classifier-Free Guidance for enhanced generation quality
- **Multiple ODE Solvers**: Support for various ODE solvers (Euler, Midpoint, DOPRI5, RK4, etc.)
- **Flexible Vocoders**: Compatible with Vocos and FireflyGAN vocoders
- **Distributed Training**: Built-in support for multi-GPU training

## News

**2024/10**: A new autoregressive TTS model is coming soon...

**2024/9**: 🚀 **StableTTS V1.1 Released** ⭐ Audio quality is largely improved ⭐

⭐ **V1.1 Release Highlights:**

- Fixed critical issues that cause the audio quality being much lower than expected. (Mainly in Mel spectrogram and Attention mask)
- Introduced U-Net-like long skip connections to the DiT in the Flow-matching Decoder.
- Use cosine timestep scheduler from [Cosyvoice](https://github.com/FunAudioLLM/CosyVoice)
- Add support for CFG (Classifier-Free Guidance).
- Add support for [FireflyGAN vocoder](https://github.com/fishaudio/vocoder/releases/tag/1.0.0).
- Switched to [torchdiffeq](https://github.com/rtqichen/torchdiffeq) for ODE solvers.
- Improved Chinese text frontend (partially based on [gpt-sovits2](https://github.com/RVC-Boss/GPT-SoVITS)).
- Multilingual support (Chinese, English, Japanese) in a single checkpoint.
- Increased parameters: 10M -> 31M.

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training and inference)
- PyTorch 2.0 or higher

### Step 1: Install PyTorch

Follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) to install PyTorch and torchaudio. We recommend the latest version (tested with PyTorch 2.4 and Python 3.12).

**Example for CUDA 11.8:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Example for CUDA 12.1:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install Optional Dependencies

For the Gradio web interface:
```bash
pip install gradio matplotlib
```

### Step 4: Download Pretrained Models

Download the pretrained models and place them in the appropriate directories:

- **TTS Model**: Place in `./checkpoints/`
- **Vocoder**: Place in `./vocoders/pretrained/`

See [Pretrained Models](#pretrained-models) section for download links.

## Quick Start

### Using the Web UI

1. Download the pretrained models (see [Pretrained Models](#pretrained-models))
2. Run the web interface:
```bash
python webui.py
```
3. Open your browser and navigate to the displayed URL (typically `http://127.0.0.1:7860`)

### Using Python API

```python
from api import StableTTSAPI
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = StableTTSAPI(
    tts_model_path='./checkpoints/checkpoint_0.pt',
    vocoder_model_path='./vocoders/pretrained/firefly-gan-base-generator.ckpt',
    vocoder_name='ffgan'
)
model.to(device)

# Synthesize speech
text = "Hello, this is a test."
ref_audio = './reference_audio.wav'
audio_output, mel_output = model.inference(
    text=text,
    ref_audio=ref_audio,
    language='english',
    step=25,
    temperature=1.0,
    length_scale=1.0,
    solver='dopri5',
    cfg=3.0
)
```

## Pretrained Models

### Text-To-Mel Model

Download and place the model in the `./checkpoints` directory.

| Model Name | Task Details | Dataset | Download Link |
|:----------:|:------------:|:-------------:|:-------------:|
| StableTTS | text to mel | 600 hours | [🤗 HuggingFace](https://huggingface.co/KdaiP/StableTTS1.1/resolve/main/StableTTS/checkpoint_0.pt) |

### Mel-To-Wav Model (Vocoders)

Choose a vocoder (`vocos` or `firefly-gan`) and place it in the `./vocoders/pretrained` directory.

| Model Name | Task Details | Dataset | Download Link |
|:----------:|:------------:|:-------------:|:-------------:|
| Vocos | mel to wav | 2k hours | [🤗 HuggingFace](https://huggingface.co/KdaiP/StableTTS1.1/resolve/main/vocoders/vocos.pt) |
| firefly-gan-base | mel to wav | HiFi-16kh | [FishAudio](https://github.com/fishaudio/vocoder/releases/download/1.0.0/firefly-gan-base-generator.ckpt) |

## Usage

### Inference

#### Using Jupyter Notebook

For detailed inference instructions, please refer to `inference.ipynb`.

#### Using Web UI

Run `webui.py` to launch a Gradio-based web interface:

```bash
python webui.py
```

The web UI provides:
- Text input with language selection
- Reference audio upload
- Adjustable parameters (steps, temperature, length scale, solver, CFG)
- Real-time mel-spectrogram visualization
- Audio playback

#### Using Python API

See the [Quick Start](#quick-start) section for a basic example.

**API Parameters:**

- `text` (str): Input text to synthesize
- `ref_audio` (str): Path to reference audio file for voice cloning
- `language` (str): Language code (`'chinese'`, `'english'`, or `'japanese'`)
- `step` (int): Number of ODE solver steps (1-100, recommended: 10-30)
- `temperature` (float): Controls variance of terminal distribution (0-2, default: 1.0)
- `length_scale` (float): Controls speech pace (0-5, default: 1.0, >1 slows down)
- `solver` (str): ODE solver type (`'euler'`, `'midpoint'`, `'dopri5'`, `'rk4'`, etc.)
- `cfg` (float): Classifier-Free Guidance strength (0-10, recommended: 1-4)

**Recommended Settings:**

- **Fast inference**: `step=10`, `solver='euler'`, `cfg=1.0`
- **High quality**: `step=30`, `solver='dopri5'`, `cfg=3.0`
- **Balanced**: `step=25`, `solver='dopri5'`, `cfg=3.0`

### Training

StableTTS is designed to be trained easily. You only need text and audio pairs, without any speaker ID or extra feature extraction.

#### Step 1: Prepare Your Data

1. **Generate Text and Audio Pairs**: Create a filelist in the format `audiopath | text` (one per line) and save it as `./filelists/example.txt`.

   Example format:
   ```
   /path/to/audio1.wav | This is the first sentence.
   /path/to/audio2.wav | This is the second sentence.
   ```

   Some recipes for open-source datasets can be found in `./recipes/`.

2. **Run Preprocessing**: Adjust the `DataConfig` in `preprocess.py` to set your input and output paths, then run:

   ```bash
   python preprocess.py
   ```

   This will:
   - Process audio files and extract mel-spectrograms
   - Convert text to phonemes using the specified language G2P
   - Output a JSON file with paths to mel features and phonemes

   **Note**: Process multilingual data separately by changing the `language` setting in `DataConfig`.

#### Step 2: Configure Training

In `config.py`, modify `TrainConfig` to set:
- `train_dataset_path`: Path to your preprocessed JSON filelist
- `batch_size`: Batch size (adjust based on GPU memory)
- `learning_rate`: Learning rate (default: 1e-4)
- `num_epochs`: Number of training epochs
- `model_save_path`: Directory to save checkpoints
- `log_dir`: Directory for TensorBoard logs

#### Step 3: Start Training

Launch the training script:

```bash
python train.py
```

The script will:
- Automatically detect and use all available GPUs
- Load the latest checkpoint if resuming training
- Save checkpoints at specified intervals
- Log training metrics to TensorBoard

**For Finetuning**: Download the pretrained model and place it in the `model_save_path` directory. The training script will automatically detect and load the pretrained checkpoint.

#### Step 4: Monitor Training

View training progress with TensorBoard:

```bash
tensorboard --logdir ./runs
```

### API Usage

The `StableTTSAPI` class provides a simple interface for inference:

```python
from api import StableTTSAPI
import torch

# Initialize model
model = StableTTSAPI(
    tts_model_path='./checkpoints/checkpoint_0.pt',
    vocoder_model_path='./vocoders/pretrained/vocos.pt',
    vocoder_name='vocos'  # or 'ffgan'
)
model.to('cuda')

# Get model parameters
tts_params, vocoder_params = model.get_params()
print(f'TTS params: {tts_params:.2f}M, Vocoder params: {vocoder_params:.2f}M')

# Synthesize
audio, mel = model.inference(
    text="Your text here",
    ref_audio="./reference.wav",
    language="chinese",
    step=25,
    temperature=1.0,
    length_scale=1.0,
    solver="dopri5",
    cfg=3.0
)
```

### (Optional) Vocoder Training

The `./vocoders/vocos` folder contains training and finetuning code for the Vocos vocoder.

For other vocoders, we recommend using [fishaudio vocoder](https://github.com/fishaudio/vocoder): a uniform interface for developing various vocoders. We use the same spectrogram transform, so vocoders trained with this interface are compatible with StableTTS.

## Model Architecture

<div align="center">

<p style="text-align: center;">
  <img src="./figures/structure.jpg" height="512"/>
</p>

</div>

### Key Components

1. **Text Encoder**: Converts phoneme sequences to hidden representations
2. **Reference Encoder**: Extracts speaker/style embeddings from reference audio
3. **Duration Predictor**: Predicts phoneme durations for alignment
4. **Flow-Matching Decoder**: Generates mel-spectrograms using continuous normalizing flows

### Architecture Details

- **Diffusion Convolution Transformer (DiT)**: We use the Diffusion Convolution Transformer block from [Hierspeech++](https://github.com/sh-lee-prml/HierSpeechpp), which combines the original [DiT](https://github.com/sh-lee-prml/HierSpeechpp) and [FFT](https://arxiv.org/pdf/1905.09263.pdf) (Feed Forward Transformer from FastSpeech) for better prosody.

- **FiLM Conditioning**: In the flow-matching decoder, we add a [FiLM layer](https://arxiv.org/abs/1709.07871) before the DiT block to condition timestep embeddings into the model.

- **U-Net Skip Connections**: Long skip connections similar to U-Net architecture improve gradient flow and generation quality.

## Configuration

### Model Configuration (`config.py`)

```python
@dataclass
class ModelConfig:
    hidden_channels: int = 256      # Hidden dimension
    filter_channels: int = 1024     # Filter dimension
    n_heads: int = 4                # Number of attention heads
    n_enc_layers: int = 3           # Encoder layers
    n_dec_layers: int = 6           # Decoder layers
    kernel_size: int = 3            # Convolution kernel size
    p_dropout: float = 0.1          # Dropout rate
    gin_channels: int = 256         # Speaker embedding dimension
```

### Mel Configuration

```python
@dataclass
class MelConfig:
    sample_rate: int = 44100        # Audio sample rate
    n_fft: int = 2048               # FFT window size
    hop_length: int = 512           # Hop length
    n_mels: int = 128               # Number of mel bins
    # ... other parameters
```

### Training Configuration

```python
@dataclass
class TrainConfig:
    train_dataset_path: str = 'filelists/filelist.json'
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10000
    model_save_path: str = './checkpoints'
    log_dir: str = './runs'
    warmup_steps: int = 200
```

## Troubleshooting

### Common Issues

**Q: CUDA out of memory during training**

A: Reduce `batch_size` in `TrainConfig` or use gradient accumulation.

**Q: Audio quality is poor**

A: Try:
- Increase `step` parameter (e.g., 25-30)
- Use `solver='dopri5'` for better quality
- Adjust `cfg` parameter (try 2-4)
- Ensure reference audio is clear and matches desired voice

**Q: Training loss is not decreasing**

A: Check:
- Learning rate (try 5e-5 or 1e-4)
- Data quality and preprocessing
- Ensure proper checkpoint loading

**Q: Japanese text processing fails**

A: If `pyopenjtalk` fails to download the dictionary:
1. Manually download [open_jtalk_dic_utf_8-1.11.tar.gz](https://github.com/r9y9/open_jtalk/releases/download/v1.11.1/open_jtalk_dic_utf_8-1.11.tar.gz)
2. Extract it
3. Set environment variable: `export OPEN_JTALK_DICT_DIR=/path/to/extracted/dict`

**Q: Import errors**

A: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
pip install gradio matplotlib  # for web UI
```

**Q: Slow inference**

A: 
- Use fewer steps (e.g., `step=10`)
- Use faster solver (`solver='euler'` or `solver='midpoint'`)
- Use GPU for inference
- Consider using `cfg=1.0` to disable CFG

### Performance Tips

1. **Inference Speed**:
   - Use `step=10-15` for real-time applications
   - Prefer `euler` or `midpoint` solvers for speed
   - Disable CFG (`cfg=1.0`) if quality is acceptable

2. **Training Speed**:
   - Use multiple GPUs (automatically detected)
   - Increase `batch_size` if GPU memory allows
   - Use `num_workers=4` or higher in DataLoader

3. **Memory Optimization**:
   - Reduce `batch_size` if OOM errors occur
   - Use mixed precision training (requires code modification)
   - Process shorter audio segments during preprocessing

## References

The development of our models heavily relies on insights and code from various projects. We express our heartfelt thanks to the creators of the following:

### Direct Inspirations

- [Matcha TTS](https://github.com/shivammehta25/Matcha-TTS): Essential flow-matching code.
- [Grad TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS): Diffusion model structure.
- [Stable Diffusion 3](https://stability.ai/news/stable-diffusion-3): Idea of combining flow-matching and DiT.
- [Vits](https://github.com/jaywalnut310/vits): Code style and MAS insights, DistributedBucketSampler.

### Additional References

- [plowtts-pytorch](https://github.com/p0p4k/pflowtts_pytorch): Codes of MAS in training
- [Bert-VITS2](https://github.com/Plachtaa/VITS-fast-fine-tuning): Numba version of MAS and modern PyTorch codes of Vits
- [fish-speech](https://github.com/fishaudio/fish-speech): Dataclass usage and mel-spectrogram transforms using torchaudio, Gradio webui
- [gpt-sovits](https://github.com/RVC-Boss/GPT-SoVITS): Melstyle encoder for voice clone
- [coqui xtts](https://huggingface.co/spaces/coqui/xtts): Gradio webui
- Chinese Dictionary: [Multi-langs_Dictionary](https://github.com/colstone/Multi-langs_Dictionary) and [atonyxu's fork](https://github.com/atonyxu/Multi-langs_Dictionary)

## TODO

- [x] Release pretrained models.
- [x] Support Japanese language.
- [x] User friendly preprocess and inference script.
- [x] Enhance documentation and citations.
- [x] Release multilingual checkpoint.

## Disclaimer

Any organization or individual is prohibited from using any technology in this repo to generate or edit someone's speech without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

✨ **Huggingface demo:** [🤗](https://huggingface.co/spaces/KdaiP/StableTTS1.1)

</div>
