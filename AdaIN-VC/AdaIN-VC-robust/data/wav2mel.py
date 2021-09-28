"""Wav2Mel for processing audio data."""

import torch
from torchaudio.sox_effects import apply_effects_tensor
from torchaudio.transforms import MelSpectrogram


class Wav2Mel(torch.nn.Module):
    """Transform audio file into mel spectrogram tensors."""

    def __init__(
        self,
        sox_effect: bool = True,
        resample: bool = True,
        norm_vad: bool = True,
        norm: bool = False,
        sample_rate: float = 16000,
        norm_db: float = -3.0,
        fft_window_ms: float = 50.0,
        fft_hop_ms: float = 12.5,
        n_fft: int = 2048,
        f_min: float = 50.0,
        n_mels: int = 80,
        preemph: float = 0.97,
        ref_db: float = 20.0,
        dc_db: float = 100.0,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.norm_db = norm_db
        self.fft_window_ms = fft_window_ms
        self.fft_hop_ms = fft_hop_ms
        self.n_fft = n_fft
        self.f_min = f_min
        self.n_mels = n_mels
        self.preemph = preemph
        self.ref_db = ref_db
        self.dc_db = dc_db

        # sox effect
        if sox_effect:
            self.sox_effects = SoxEffects(
                resample=resample, norm_vad=norm_vad, norm=norm, norm_db=norm_db
            )
        # mel spectrogram
        self.log_melspectrogram = LogMelspectrogram(
            sample_rate,
            fft_window_ms,
            fft_hop_ms,
            n_fft,
            f_min,
            n_mels,
            preemph,
            ref_db,
            dc_db,
        )

    def mel(self, wav_tensor: torch.Tensor) -> torch.Tensor:
        mel_tensor = self.log_melspectrogram(wav_tensor)
        return mel_tensor

    def forward(self, wav_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        # 1. sox effect
        wav_tensor = self.sox_effects(wav_tensor, sample_rate)
        if wav_tensor is None:
            return None

        # 2. log mel-spectrogram
        mel_tensor = self.log_melspectrogram(wav_tensor)
        return mel_tensor


class SoxEffects(torch.nn.Module):
    """Transform waveform tensors."""

    def __init__(
        self,
        resample: bool = True,
        norm_vad: bool = True,
        norm: bool = False,
        sample_rate: int = 16000,
        norm_db: float = -3,
    ):
        super().__init__()
        self.effects = [
            ["channels", "1"],  # convert to mono
        ]

        if resample:
            self.effects.extend([["rate", f"{sample_rate}"]])

        if norm_vad:
            # remove silence throughout the file; vad only trim beginning of the utternece
            # 1. use "reverse" to trim the end of the utterence
            # 2. norm before vad is recommended
            self.effects.extend([["norm"], ["vad"], ["reverse"], ["vad"], ["reverse"]])

        if norm:
            self.effects.extend([["norm", f"{norm_db}"]])

        # === print effect chains === #
        print(f"[INFO] sox_effects : {self.effects}")

    def forward(self, wav_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav_tensor, _ = apply_effects_tensor(wav_tensor, sample_rate, self.effects)
        if wav_tensor.numel() == 0:
            return None
        return wav_tensor


class LogMelspectrogram(torch.nn.Module):
    """Transform waveform tensors into log mel spectrogram tensors."""

    def __init__(
        self,
        sample_rate: float,
        fft_window_ms: float,
        fft_hop_ms: float,
        n_fft: int,
        f_min: float,
        n_mels: int,
        preemph: float,
        ref_db: float,
        dc_db: float,
    ):
        super().__init__()
        self.melspectrogram = MelSpectrogram(
            sample_rate=sample_rate,
            win_length=int(sample_rate * fft_window_ms / 1000),
            hop_length=int(sample_rate * fft_hop_ms / 1000),
            n_fft=n_fft,
            f_min=f_min,
            n_mels=n_mels,
        )
        self.preemph = preemph
        self.ref_db = ref_db
        self.dc_db = dc_db

    def forward(self, wav_tensor: torch.Tensor) -> torch.Tensor:
        # preemph
        wav_tensor = torch.cat(
            (
                wav_tensor[:, 0].unsqueeze(-1),
                wav_tensor[:, 1:] - self.preemph * wav_tensor[:, :-1],
            ),
            dim=-1,
        )
        mel_tensor = self.melspectrogram(wav_tensor).squeeze(0)  # (n_mels, time)
        mel_tensor = 20 * mel_tensor.clamp(min=1e-9).log10()
        mel_tensor = (mel_tensor - self.ref_db + self.dc_db) / self.dc_db
        return mel_tensor

    # def forward_batch(self, wav_tensor: torch.Tensor) -> torch.Tensor:
    #     # preemph
    #     # wav_tensor: batch, length
    #     assert len(wav_tensor.shape) == 2 and len(wav_tensor) > 1
    #     wav_tensor = torch.cat(
    #         wav_tensor[:, 0],
    #         wav_tensor[:, 1:] - self.preemph * wav_tensor[:, :-1],
    #         dim=-1,
    #     )
    #     mel_tensor = self.melspectrogram(wav_tensor)  # batch, n_mels, time
    #     mel_tensor = 20 * mel_tensor.clamp(min=1e-9).log10()
    #     mel_tensor = (mel_tensor - self.ref_db + self.dc_db) / self.dc_db
    #     return mel_tensor
