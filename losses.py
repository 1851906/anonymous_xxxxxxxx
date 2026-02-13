import torch
import torch.nn as nn
import torch.nn.functional as F

class LogMagSTFTLoss(nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(LogMagSTFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length), persistent=False)

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        window = self.window.to(x.device)
        mag_x = torch.stft(x, self.fft_size, self.shift_size, self.win_length, window, return_complex=True,
                           pad_mode='constant').abs()
        mag_y = torch.stft(y, self.fft_size, self.shift_size, self.win_length, window, return_complex=True,
                           pad_mode='constant').abs()
        loss = F.mse_loss(torch.log1p(mag_x), torch.log1p(mag_y))

        return loss