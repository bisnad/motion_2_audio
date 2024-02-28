import torch
import torch.nn as nn
import torchaudio as ta
import torchaudio.transforms as tat
from einops import rearrange
import numpy as np

class STFT(nn.Module):
    
    def __init__(self,
                 scale, # int
                 normalized = False # bool
                 ):
        super().__init__()
        
        self.scale = scale
        
        """
        self.stft = tat.Spectrogram(
            n_fft=scale,
            win_length=scale,
            hop_length=scale // 4,
            normalized=normalized,
            center=False
        )
        """
        
        self.stft = tat.Spectrogram(
            n_fft=scale,
            win_length=scale,
            hop_length=scale // 4,
            normalized=normalized
        )
        
    def forward(self, 
                x # torch.Tensor
                ):
        
        #print("stft scale ", self.stft.n_fft)
        #print("x s ", x.shape)

        # pad audio
        # y = torch.nn.functional.pad(x, ((self.scale - x.shape[-1])//2, (self.scale - x.shape[-1])//2), "constant", 0)
        y = x
        
        #print("y1 s ", y.shape)
        
        # stft
        y = self.stft(y)
        
        #print("y2 s ", y.shape)
        
        return y

class MelScale(nn.Module):

    def __init__(self, 
                 sample_rate, # int
                 n_fft, # int
                 n_mels, # int
                 ):
        super().__init__()
        self.mel = tat.MelScale(n_mels=n_mels, 
                               sample_rate=sample_rate, 
                               n_stft=n_fft // 2 + 1)

    def forward(self, 
                x # torch.Tensor
                ):
        y = self.mel(x)
        return y
    
class MultiScaleSTFT(nn.Module):

    def __init__(self,
                 scales, # List[int]
                 sample_rate, # int
                 magnitude = True, # bool
                 normalized = False, # bool
                 num_mels = 0 # int
                 ):
        super().__init__()
        self.scales = scales
        self.magnitude = magnitude
        self.num_mels = num_mels

        self.stfts = []
        self.mel_scales = []
        for scale in scales:
            self.stfts.append(
                STFT(scale = scale,
                     normalized=normalized
                    ))
            
            if num_mels != 0:
                self.mel_scales.append(
                    MelScale(
                        sample_rate=sample_rate,
                        n_fft=scale,
                        n_mels=num_mels,
                    ))
            else:
                self.mel_scales.append(None)

        self.stfts = nn.ModuleList(self.stfts)
        self.mel_scales = nn.ModuleList(self.mel_scales)

    def forward(self, 
                x # torch.Tensor
                ):
        x = rearrange(x, "b c t -> (b c) t")
        stfts = []
        for stft, mel in zip(self.stfts, self.mel_scales):
            y = stft(x)
            if mel is not None:
                y = mel(y)
            if self.magnitude:
                y = y.abs()
            else:
                y = torch.stack([y.real, y.imag], -1)
            stfts.append(y)

        return stfts
    
def mean_difference(target, # torch.Tensor
                    value, # torch.Tensor
                    norm='L1', # str
                    relative=False # bool
                    ):
    diff = target - value
    if norm == 'L1':
        diff = diff.abs().mean()
        if relative:
            diff = diff / target.abs().mean()
        return diff
    elif norm == 'L2':
        diff = (diff * diff).mean()
        if relative:
            diff = diff / (target * target).mean()
        return diff
    else:
        raise Exception(f'Norm must be either L1 or L2, got {norm}')

    
class AudioDistance(nn.Module):

    def __init__(self, 
                 multiscale_stft, #nn.Module
                 log_epsilon #float)
                 ):
        super().__init__()
        self.multiscale_stft = multiscale_stft
        self.log_epsilon = log_epsilon

    def forward(self, 
                x, # torch.Tensor
                y, # torch.Tensor
                ):
        
        stfts_x = self.multiscale_stft(x)
        stfts_y = self.multiscale_stft(y)
        
        distance = 0.

        for x, y in zip(stfts_x, stfts_y):
            
            #print("x s ", x.shape, " y s ", y.shape)

            logx = torch.log(x + self.log_epsilon)
            logy = torch.log(y + self.log_epsilon)

            lin_distance = mean_difference(x, y, norm='L2', relative=True)
            log_distance = mean_difference(logx, logy, norm='L1')

            distance = distance + lin_distance + log_distance
            
        distance /= len(stfts_x)

        return distance

"""
mtft_scales = [2048, 1024, 512, 256, 128]
sample_rate = 16000
mtft_magnitude = True
mtft_normalized = False
mtft_num_mels = 128

multi_stft = MultiScaleSTFT(mtft_scales, sample_rate, mtft_magnitude, mtft_normalized, mtft_num_mels)

audio_dist = AudioDistance(multi_stft, 1e-4)


batch_size = 16
waveform_length = 2048

test_waveform_1 = torch.rand((batch_size, 1, waveform_length))
test_waveform_2 = torch.rand((batch_size, 1, waveform_length))

audio_dist(test_waveform_1, test_waveform_2)

MultiScaleSTFT

magnitude = True, # bool
normalized = False, # bool
num_mels = 0 # int

sample_rate = 16000
fft_scale = 1024
fft_normalized = False
mel_nmels = 128


stft = STFT(fft_scale, fft_normalized)
mel = MelScale(sample_rate, fft_scale, mel_nmels)
    
batch_size = 16
waveform_length = 256

test_waveform = torch.rand((batch_size, waveform_length))

print("test_waveform s ", test_waveform.shape)

test_spec = stft(test_waveform)

print("test_spec s ", test_spec.shape)

test_mel = mel(test_spec)

print("test_mel s ", test_mel.shape)
"""

def get_beta_kl(step, warmup, min_beta, max_beta):
    if step > warmup: return max_beta
    t = step / warmup
    min_beta_log = np.log(min_beta)
    max_beta_log = np.log(max_beta)
    beta_log = t * (max_beta_log - min_beta_log) + min_beta_log
    return np.exp(beta_log)


def get_beta_kl_cyclic(step, cycle_size, min_beta, max_beta):
    return get_beta_kl(step % cycle_size, cycle_size // 2, min_beta, max_beta)


def get_beta_kl_cyclic_annealed(step, cycle_size, warmup, min_beta, max_beta):
    min_beta = get_beta_kl(step, warmup, min_beta, max_beta)
    return get_beta_kl_cyclic(step, cycle_size, min_beta, max_beta)