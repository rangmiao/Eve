import os
from functools import lru_cache
from typing import Optional, Union
import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from subprocess import CalledProcessError, run, Popen, PIPE
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperModel


# Copy from Qwen-aduio
def exact_div(x, y):
    assert x % y == 0
    return x // y

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


def get_T_after_cnn(L_in, dilation=1):
    for (padding, kernel_size, stride) in eval("[(1,3,1)] + [(1,3,2)] "):
        L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
        L_out = 1 + L_out // stride
        L_in = L_out
    return L_out


def load_bytesio_audio(content, sr: int = SAMPLE_RATE):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", "pipe:",
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "pipe:"
    ]
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, bufsize=-1)
    out, _ = p.communicate(input=content)
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


def trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)
    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(
        os.path.join(os.path.dirname(__file__), "mel_filters.npz") # todo
        # os.path.join("assets", "mel_filters.npz")
    ) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = N_MELS,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class WhisperAudioTower(nn.Module):
    def __init__(self, audio_tower, args, **kwargs):
        super().__init__()
        # import pdb;pdb.set_trace()

        self.is_loaded = False

        self.audio_tower_name = audio_tower
        self.args = args
        self.load_model()

    def load_model(self):
        print("WhisperAudioTower load_model")
        self.audio_encoder = WhisperModel.from_pretrained(self.audio_tower_name)
        self.audio_encoder.requires_grad_(False)
        self.audio_processor = self.process_audio
        self.is_loaded = True

    def process_audio(self, audio_urls):
        if len(audio_urls) > 0:
            audios, audio_lens, audio_span_tokens = [], [], []
            for audio_path in audio_urls:
                audio = load_audio(audio_path)
                L = (audio.shape[0] if audio.shape[0] <= 480000 else 480000)  # max_length < 30s
                mel_len = L // 160
                audio = pad_or_trim(audio.flatten())
                mel = log_mel_spectrogram(audio)
                audio_len_after_cnn = get_T_after_cnn(mel_len)
                audio_token_num = audio_len_after_cnn
                # audio_token_num = (audio_len_after_cnn - 2) // 2 + 1
                audio_len = [audio_len_after_cnn, audio_token_num]
                audios.append(mel)
                audio_lens.append(audio_len)
                audio_span_tokens.append(audio_token_num)  # do not add audio bos eos
                # audio_span_tokens.append(audio_token_num + 2)  # add audio bos eos
            input_audio_lengths = torch.IntTensor(audio_lens)
            input_audios = torch.stack(audios, dim=0)
            return {"input_audios": input_audios,
                    "input_audio_lengths": input_audio_lengths,
                    "audio_span_tokens": audio_span_tokens,
                    "audio_urls": audio_urls}
        else:
            return None

    @torch.no_grad()
    def forward(self, audios, input_audio_lengths):
        # audio_urls to audio_mels
        # audio_info = self.process_audio(audio_urls)
        # audio_mels to audio_features
        # audios = audio_info["input_audios"]
        # audio_span_tokens = audio_info["audio_span_tokens"]
        # input_audio_lengths = audio_info["input_audio_lengths"]
        audios = self.audio_encoder(audios, input_audio_lengths[0])
        # print('features', audios)
        return audios['last_hidden_state']

    @property
    def hidden_size(self):
        return 1024


if __name__ == '__main__':
    audio_tower_name = '/home/ma-user/work/project/LMM/Eve-LLaVA_lossgd/models/audio_model/whisper/medium.en'
    audio_model = WhisperAudioTower(audio_tower_name, args=None)

    audio_data = audio_model.process_audio(["/home/ma-user/work/data/VLM/audio/AnyInstruct/speech_anygpt/speech/chunk_00436/0035.mp3"])
    # 1, 80, 3000
    print("audio_data", audio_data['input_audios'].shape)
    audios, input_audio_lengths = audio_data['input_audios'], audio_data['input_audio_lengths']
    output = audio_model(audios, input_audio_lengths)
    # import pdb;pdb.set_trace()
    print("output", output.shape)
    # 1, 260, 1024