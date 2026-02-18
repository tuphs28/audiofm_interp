
import numpy as np
import torch
from transformers import AutoModel, AutoFeatureExtractor, AutoConfig
import librosa

class FeatureExtractor:
    """Class to extract hidden states and associated audio features from a model"""

    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-base",
        random_init: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lag: int = 0, # I changed this to 0 - I had this set to 1 after the weird colab results, but now I think that may have just been a bug in the original notebook. Check this.
        hop_length: int = 320,
        n_fft: int = 512,
        n_mels: int = 80,
        n_mfcc: int = 13
    ):

        self.model_id = model_id
        self.device = device
        self.processor = AutoFeatureExtractor.from_pretrained(model_id) # NOTE - update when using more models
        self.lag = lag # NOTE - update when using more models

        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

        if random_init:
            config = AutoConfig.from_pretrained(model_id)
            self.model = AutoModel.from_config(config).to(self.device)
        else:
            self.model = AutoModel.from_pretrained(model_id).to(self.device)

        self.model.eval()


    def get_aligned_hidden_states_and_targets(
        self,
        batch: dict,
        sr: int = 16000
    ) -> tuple[dict[int, torch.Tensor], dict[str, np.ndarray]]:

        audio_array = batch["audio"]["array"]
        hidden_states = self._get_hidden_states(audio_array, sr)
        accoustic_targets = self._get_acoustic_targets(audio_array, sr)
        semantic_targets = self._get_semantic_targets(batch, accoustic_targets["stft"].shape[0])

        n_frames_hiddens = hidden_states[0].shape[0]
        n_frames_targets = accoustic_targets["stft"].shape[0]
        min_frames = min(n_frames_hiddens, n_frames_targets)

        if self.lag > 0:
            aligned_hidden_states = {layer_idx: hs[:min_frames][:-self.lag] for layer_idx, hs in hidden_states.items()}
            aligned_acoustic_targets = {k: v[:min_frames][self.lag:] for k, v in accoustic_targets.items()}
            aligned_semantic_targets = {k: v[:min_frames][self.lag:] for k, v in semantic_targets.items()}
        else:
            aligned_hidden_states = {layer_idx: hs[:min_frames] for layer_idx, hs in hidden_states.items()}
            aligned_acoustic_targets = {k: v[:min_frames] for k, v in accoustic_targets.items()}
            aligned_semantic_targets = {k: v[:min_frames] for k, v in semantic_targets.items()}

        all_targets = {**aligned_acoustic_targets, **aligned_semantic_targets}

        return aligned_hidden_states, all_targets
    

    def _get_hidden_states(
        self,
        audio_array: np.ndarray,
        sr: int = 16000
    ) -> dict[int, torch.Tensor]:

        inputs = self.processor(audio_array, sampling_rate=sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        hidden_states_dict = {
            layer_idx: hidden_state.squeeze(0).cpu() for layer_idx, hidden_state in enumerate(hidden_states)
            }
        return hidden_states_dict


    def _get_acoustic_targets(
        self,
        audio_array: np.ndarray,
        sr: int = 16000
    ) -> dict[str, np.ndarray]:

        stft_mag = np.abs(librosa.stft(audio_array, n_fft=self.n_fft, hop_length=self.hop_length))
        mel_power = librosa.feature.melspectrogram(y=audio_array, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
        
        stft_db = librosa.amplitude_to_db(stft_mag, ref=np.max)
        mel_db = librosa.power_to_db(mel_power, ref=np.max)
        
        mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=self.n_mfcc)
        
        feature_dict = {
            "stft": stft_db.T,
            "mel": mel_db.T,
            "mfcc": mfcc.T
        }

        return feature_dict

    def _get_semantic_targets(
            self,
            batch: dict,
            n_frames: int 
    ) -> dict[str, np.ndarray]:
        
        speaker_ids = np.array([batch["speaker_id"]] * n_frames).reshape(-1, 1)
        return {
            "speaker_id": speaker_ids
        }