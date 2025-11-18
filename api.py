import os
import torch
import torch.nn as nn
from typing import Optional, Tuple

from dataclasses import asdict

from utils.audio import LogMelSpectrogram
from config import ModelConfig, MelConfig
from models.model import StableTTS

from text import symbols
from text import cleaned_text_to_sequence
from text.mandarin import chinese_to_cnm3
from text.english import english_to_ipa2
from text.japanese import japanese_to_ipa2

from datas.dataset import intersperse
from utils.audio import load_and_resample_audio

def get_vocoder(model_path, model_name='ffgan') -> nn.Module:
    if model_name == 'ffgan':
        # training or changing ffgan config is not supported in this repo
        # you can train your own model at https://github.com/fishaudio/vocoder
        from vocoders.ffgan.model import FireflyGANBaseWrapper
        vocoder = FireflyGANBaseWrapper(model_path)
        
    elif model_name == 'vocos':
        from vocoders.vocos.models.model import Vocos
        from config import VocosConfig, MelConfig
        vocoder = Vocos(VocosConfig(), MelConfig())
        vocoder.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
        vocoder.eval()
        
    else:
        raise NotImplementedError(f"Unsupported model: {model_name}")
        
    return vocoder

class StableTTSAPI(nn.Module):
    def __init__(self, tts_model_path: str, vocoder_model_path: str, vocoder_name: str = 'ffgan'):
        """
        Initialize StableTTS API.
        
        Args:
            tts_model_path: Path to TTS model checkpoint
            vocoder_model_path: Path to vocoder model checkpoint
            vocoder_name: Vocoder type ('ffgan' or 'vocos')
        """
        super().__init__()
        
        if not os.path.exists(tts_model_path):
            raise FileNotFoundError(f"TTS model file not found: {tts_model_path}")
        
        if not os.path.exists(vocoder_model_path):
            raise FileNotFoundError(f"Vocoder model file not found: {vocoder_model_path}")

        self.mel_config = MelConfig()
        self.tts_model_config = ModelConfig()
        
        self.mel_extractor = LogMelSpectrogram(**asdict(self.mel_config))
        
        # text to mel spectrogram
        try:
            self.tts_model = StableTTS(len(symbols), self.mel_config.n_mels, **asdict(self.tts_model_config))
            tts_state = torch.load(tts_model_path, map_location='cpu', weights_only=True)
            self.tts_model.load_state_dict(tts_state)
            self.tts_model.eval()
        except Exception as e:
            raise RuntimeError(f"Error loading TTS model from {tts_model_path}: {e}")
        
        # mel spectrogram to waveform
        try:
            self.vocoder_model = get_vocoder(vocoder_model_path, vocoder_name)
            self.vocoder_model.eval()
        except Exception as e:
            raise RuntimeError(f"Error loading vocoder from {vocoder_model_path}: {e}")
        
        self.g2p_mapping = {
            'chinese': chinese_to_cnm3,
            'japanese': japanese_to_ipa2,
            'english': english_to_ipa2,
        }
        self.supported_languages = list(self.g2p_mapping.keys())
        
    @torch.inference_mode()
    def inference(self, text: str, ref_audio: str, language: str, step: int, 
                  temperature: float = 1.0, length_scale: float = 1.0, 
                  solver: Optional[str] = None, cfg: float = 3.0):
        """
        Synthesize speech from text using reference audio.
        
        Args:
            text: Input text to synthesize
            ref_audio: Path to reference audio file
            language: Language code ('chinese', 'english', or 'japanese')
            step: Number of ODE solver steps (1-100)
            temperature: Controls variance of terminal distribution (0-2)
            length_scale: Controls speech pace (0-5, >1 slows down)
            solver: ODE solver type ('euler', 'midpoint', 'dopri5', etc.)
            cfg: Classifier-Free Guidance strength (0-10)
            
        Returns:
            Tuple of (audio_output, mel_output) tensors on CPU
        """
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")
        
        if not os.path.exists(ref_audio):
            raise FileNotFoundError(f"Reference audio file not found: {ref_audio}")
        
        if language not in self.supported_languages:
            raise ValueError(f"Unsupported language: {language}. Supported: {list(self.supported_languages)}")
        
        if step < 1 or step > 100:
            raise ValueError(f"Step must be between 1 and 100, got {step}")
        
        if temperature < 0 or temperature > 2:
            raise ValueError(f"Temperature must be between 0 and 2, got {temperature}")
        
        if length_scale < 0 or length_scale > 5:
            raise ValueError(f"Length scale must be between 0 and 5, got {length_scale}")
        
        device = next(self.parameters()).device
        phonemizer = self.g2p_mapping.get(language)
        
        if phonemizer is None:
            raise ValueError(f"No phonemizer found for language: {language}")
        
        try:
            text = phonemizer(text)
        except Exception as e:
            raise RuntimeError(f"Error in text-to-phoneme conversion: {e}")
        
        if not text or len(text.strip()) == 0:
            raise ValueError("Phonemized text is empty")
        
        text = torch.tensor(intersperse(cleaned_text_to_sequence(text), item=0), dtype=torch.long, device=device).unsqueeze(0)
        text_length = torch.tensor([text.size(-1)], dtype=torch.long, device=device)
        
        try:
            ref_audio_tensor = load_and_resample_audio(ref_audio, self.mel_config.sample_rate)
            if ref_audio_tensor is None:
                raise ValueError(f"Failed to load audio from {ref_audio}")
            ref_audio_tensor = ref_audio_tensor.to(device)
            ref_audio_mel = self.mel_extractor(ref_audio_tensor)
        except Exception as e:
            raise RuntimeError(f"Error processing reference audio: {e}")
        
        try:
            mel_output = self.tts_model.synthesise(text, text_length, step, temperature, ref_audio_mel, length_scale, solver, cfg)['decoder_outputs']
            audio_output = self.vocoder_model(mel_output)
        except Exception as e:
            raise RuntimeError(f"Error during synthesis: {e}")
        
        return audio_output.cpu(), mel_output.cpu()
    
    def get_params(self) -> Tuple[float, float]:
        """
        Get the number of parameters in millions for TTS model and vocoder.
        
        Returns:
            Tuple of (tts_params_millions, vocoder_params_millions)
        """
        tts_param = sum(p.numel() for p in self.tts_model.parameters()) / 1e6
        vocoder_param = sum(p.numel() for p in self.vocoder_model.parameters()) / 1e6
        return tts_param, vocoder_param
    
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tts_model_path = './checkpoints/checkpoint_0.pt'
    vocoder_model_path = './vocoders/pretrained/vocos.pt'
    
    model = StableTTSAPI(tts_model_path, vocoder_model_path, 'vocos')
    model.to(device)
    
    text = '樱落满殇祈念集……殇歌花落集思祈……樱花满地集于我心……揲舞纷飞祈愿相随……'
    audio = './audio_1.wav'
    
    audio_output, mel_output = model.inference(text, audio, 'chinese', 10, solver='dopri5', cfg=3)
    print(audio_output.shape)
    print(mel_output.shape)
    
    import torchaudio
    torchaudio.save('output.wav', audio_output, MelConfig().sample_rate)
    
    
