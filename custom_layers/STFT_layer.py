import torch, torchaudio
from torch import nn

class STFT_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=127, window_fn=torch.hann_window, win_length=127, hop_length=40, normalized=True)

    def forward(self, x):
        # STFT - 64x64
        x = self.spectrogram(x)
        x = torch.absolute(x) # magnitude
        x = 20 * torch.log10(x) # to dB
        # x = self.norm(x)
        x /= 10

        return x