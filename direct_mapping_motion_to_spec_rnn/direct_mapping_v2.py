"""
same as direct_mapping.py 
but with translating a single dance pose into multiple mel spectrograms
doesn't work very well, so avoid for the time being
"""

"""
Imports
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict
import scipy.linalg as sclinalg

import math
import os, sys, time, subprocess
import numpy as np
import csv
import matplotlib.pyplot as plt

# audio specific imports

import torchaudio
import torchaudio.transforms as transforms
import simpleaudio as sa

# mocap specific imports

from common import utils
from common import bvh_tools as bvh
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, slerp
from common.pose_renderer import PoseRenderer

"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Mocap Settings
"""

mocap_data_path = "D:/data/mocap/Motion2Audio/improvisation_mocap.bvh"
mocap_fps = 50
mocap_dim = -1 # automatically determined when loading mocap data

"""
Audio Settings
"""

audio_file = "D:/data/audio/Motion2Audio/improvisation_audio_48khz.wav"
audio_sample_rate = 48000
audio_n_mels = 128
audio_n_mel_count = 4
audio_n_fft = 2048
audio_dim = audio_n_mels * audio_n_mel_count
audio_n_hop_length = audio_sample_rate // ( mocap_fps * audio_n_mel_count)
audio_n_win_length = audio_n_hop_length * 2
audio_griffinlim_iterations = 10

"""
Dataset Settings
"""

valid_frame_start = 1500
valid_frame_end = 29180
batch_size = 32
test_percentage = 0.1

"""
Model Settings
"""

sequence_length = 128
m2a_rnn_layer_count = 2
m2a_rnn_layer_size = 512
m2a_dense_layer_sizes = [ 512 ]

save_models = False
save_tscript = False
save_weights = True

# load model weights
load_weights = False
m2a_weights_file = "results_v2/weights/m2a_weights_epoch_600"

"""
Training Settings
"""

learning_rate = 1e-4
model_save_interval = 50
load_weights = False
save_weights = True
transformer_load_weights_path = "results_v3/weights/transformer_weights_epoch_200"
epochs = 600

"""
Load Data - Mocap
"""

# load mocap data
bvh_tools = bvh.BVH_Tools()
mocap_tools = mocap.Mocap_Tools()

bvh_data = bvh_tools.load(mocap_data_path)
mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])

# get mocap data
mocap_sequence = mocap_data["motion"]["rot_local"].astype(np.float32)

# normalize mocap sequence
mocap_mean = np.mean(mocap_sequence)
mocap_std = np.std(mocap_sequence)

mocap_sequence = (mocap_sequence - mocap_mean) / mocap_std

total_mocap_sequence_length = mocap_sequence.shape[0]
joint_count = mocap_sequence.shape[1]
joint_dim = mocap_sequence.shape[2]
pose_dim = joint_count * joint_dim
mocap_dim = pose_dim

mocap_sequence = mocap_sequence.reshape(total_mocap_sequence_length, mocap_dim)
mocap_sequence = mocap_sequence[valid_frame_start:valid_frame_end,...]
mocap_sequence = torch.tensor(mocap_sequence)

total_mocap_sequence_length = mocap_sequence.shape[0]

offsets = mocap_data["skeleton"]["offsets"].astype(np.float32)
parents = mocap_data["skeleton"]["parents"]
children = mocap_data["skeleton"]["children"]

# create edge list
def get_edge_list(children):
    edge_list = []

    for parent_joint_index in range(len(children)):
        for child_joint_index in children[parent_joint_index]:
            edge_list.append([parent_joint_index, child_joint_index])
    
    return edge_list

edge_list = get_edge_list(children)

poseRenderer = PoseRenderer(edge_list)

"""
Load Data - Audio
"""

waveform_data, _ = torchaudio.load(audio_file)

# play audio file
waveform_data_np = waveform_data.numpy().flatten()
play_obj = sa.play_buffer(waveform_data_np[audio_sample_rate * 60:], 1, 4, audio_sample_rate)
play_obj.stop()

# forward and inverse spectral conversions

mel_transform = torchaudio.transforms.MelSpectrogram(audio_sample_rate, n_fft=audio_n_fft, n_mels=audio_n_mels, win_length=audio_n_win_length, hop_length=audio_n_hop_length)
inv_mel_transform = torchaudio.transforms.InverseMelScale(sample_rate=audio_sample_rate, n_stft=audio_n_fft // 2 + 1, n_mels=audio_n_mels)
griffin_lim = torchaudio.transforms.GriffinLim(n_fft=audio_n_fft, win_length=audio_n_win_length, hop_length=audio_n_hop_length, n_iter=audio_griffinlim_iterations)

# compute mel spectra of entire waveform
mel_data = mel_transform(waveform_data)

# create audio data (swap axes and standardize)
audio_data = mel_data.squeeze(axis=0)
audio_data = torch.permute(audio_data, (1, 0))
audio_data_mean = torch.mean(audio_data, dim=0, keepdim=True)
audio_data_std = torch.std(audio_data, dim=0, keepdim=True)
audio_data = (audio_data - audio_data_mean) / audio_data_std

# create audio sequence

audio_sequence = audio_data[valid_frame_start * audio_n_mel_count:valid_frame_end * audio_n_mel_count,...]
audio_sequence = audio_sequence.reshape(-1, audio_dim)

total_audio_sequence_length = audio_sequence.shape[0]


print("total_mocap_sequence_length ", total_mocap_sequence_length, " total_audio_sequence_length ", total_audio_sequence_length)

assert total_mocap_sequence_length == total_audio_sequence_length

total_sequence_length = total_mocap_sequence_length

"""
Create Dataset
"""


mocap_excerpts = []
audio_excerpts = []

for pI in range(total_sequence_length - sequence_length):
    
    mocap_excerpt = mocap_sequence[pI:pI+sequence_length]
    mocap_excerpts.append(mocap_excerpt.reshape((1, sequence_length, mocap_dim)))
    
    audio_excerpt = audio_sequence[pI:pI+sequence_length]
    audio_excerpts.append(audio_excerpt.reshape((1, sequence_length, audio_dim)))


mocap_excerpts = torch.cat(mocap_excerpts, dim=0)
audio_excerpts = torch.cat(audio_excerpts, dim=0)

print("mocap_excerpts s ", mocap_excerpts.shape)
print("audio_excerpts s ", audio_excerpts.shape)


class MocapAudio(Dataset):
    def __init__(self, mocap_excerpts, audio_excerpts):
        self.mocap_excerpts = mocap_excerpts
        self.audio_excerpts = audio_excerpts
    
    def __len__(self):
        return self.mocap_excerpts.shape[0]
    
    def __getitem__(self, idx):
        return self.mocap_excerpts[idx, ...], self.audio_excerpts[idx, ...]
    
full_dataset = MocapAudio(mocap_excerpts, audio_excerpts)

mocap_item, audio_item = full_dataset[0]

print("mocap_item s ", mocap_item.shape)
print("audio_item s ", audio_item.shape)

test_size = int(test_percentage * len(full_dataset))
train_size = len(full_dataset) - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

mocap_batch, audio_batch = next(iter(train_loader))

print("mocap_batch s ", mocap_batch.shape)
print("audio_batch s ", audio_batch.shape)

"""
create models
"""

# create mocap 2 audio model
    
class Mocap2AudioModel(nn.Module):
    def __init__(self, sequence_length, mocap_dim, audio_dim, rnn_layer_count, rnn_layer_size, dense_layer_sizes):
        super(Mocap2AudioModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.mocap_dim = mocap_dim
        self.audio_dim = audio_dim
        self.rnn_layer_count = rnn_layer_count
        self.rnn_layer_size = rnn_layer_size 
        self.dense_layer_sizes = dense_layer_sizes
    
        # create recurrent layers
        rnn_layers = []
        rnn_layers.append(("m2a_rnn_0", nn.LSTM(self.mocap_dim, self.rnn_layer_size, self.rnn_layer_count, batch_first=True)))
        
        self.rnn_layers = nn.Sequential(OrderedDict(rnn_layers))
        
        # create dense layers
        
        dense_layers = []
        
        dense_layers.append(("m2a_dense_0", nn.Linear(self.rnn_layer_size, self.dense_layer_sizes[0])))
        dense_layers.append(("m2a_dense_relu_0", nn.ReLU()))
        
        dense_layer_count = len(self.dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("m2a_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("m2a_dense_relu_{}".format(layer_index), nn.ReLU()))

        dense_layers.append(("pm2a_dense_{}".format(len(self.dense_layer_sizes)), nn.Linear(self.dense_layer_sizes[-1], self.audio_dim)))
        
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        
    def forward(self, x):
        
        #print("x 1 ", x.shape)
        
        x, (_, _) = self.rnn_layers(x)
        
        #print("x 2 ", x.shape)
        
        x = x.reshape(-1, x.shape[2])
        
        #print("x 3 ", x.shape)
        
        x = self.dense_layers(x)
        
        #print("x 4 ", x.shape)
        
        x = x.reshape(-1, self.sequence_length, self.audio_dim)
        
        #print("x 5 ", x.shape)
 
        return x
    
m2a = Mocap2AudioModel(sequence_length, mocap_dim, audio_dim, m2a_rnn_layer_count, m2a_rnn_layer_size, m2a_dense_layer_sizes).to(device)

print(m2a)

m2a_in = mocap_batch.to(device)
m2a_out = m2a(m2a_in)

print("m2a_in s ", m2a_in.shape)
print("m2a_out s ", m2a_out.shape)

if load_weights and m2a_weights_file:
    m2a.load_state_dict(torch.load(m2a_weights_file, map_location=device))

"""
Training
"""

m2a_optimizer = torch.optim.Adam(m2a.parameters(), lr=learning_rate)
m2a_scheduler = torch.optim.lr_scheduler.StepLR(m2a_optimizer, step_size=200, gamma=0.5) # reduce the learning every 20 epochs by a factor of 10

l1_loss = nn.L1Loss()

def audio_loss(y_audio, yhat_audio):
    
    _loss = l1_loss(yhat_audio, y_audio)
    
    return _loss

def m2a_train_step(batch_mocap, batch_audio):
    
    target_audio = batch_audio
    
    # predict audio
    pred_audio = m2a(batch_mocap)
    
    _loss = audio_loss(target_audio, pred_audio) 
    m2a_optimizer.zero_grad()
    _loss.backward()
    
    m2a_optimizer.step()
    
    return _loss

def m2a_test_step(batch_mocap, batch_audio):
    
    with torch.no_grad():
        target_audio = batch_audio
        
        # predict audio
        pred_audio = m2a(batch_mocap)
        
        _loss = audio_loss(target_audio, pred_audio) 
    
    return _loss


def train(train_dataloader, test_dataloader, epochs):
    
    loss_history = {}
    loss_history["m2a train"] = []
    loss_history["m2a test"] = []
    
    for epoch in range(epochs):

        start = time.time()
        
        m2a_train_loss_per_epoch = []
        
        for train_batch_mocap, train_batch_audio in train_dataloader:
            
            train_batch_mocap = train_batch_mocap.to(device)
            train_batch_audio = train_batch_audio.to(device)
            
            _m2a_loss = m2a_train_step(train_batch_mocap, train_batch_audio)
            _m2a_loss = _m2a_loss.detach().cpu().numpy()
            m2a_train_loss_per_epoch.append(_m2a_loss)
            
        m2a_train_loss_per_epoch = np.mean(np.array(m2a_train_loss_per_epoch))
        
        m2a_test_loss_per_epoch = []
        
        for test_batch_mocap, test_batch_audio in test_dataloader:
            
            test_batch_mocap = test_batch_mocap.to(device)
            test_batch_audio = test_batch_audio.to(device)
            
            _m2a_loss = m2a_test_step(test_batch_mocap, test_batch_audio)
            
            _m2a_loss = _m2a_loss.detach().cpu().numpy()
            m2a_test_loss_per_epoch.append(_m2a_loss)

        m2a_test_loss_per_epoch = np.mean(np.array(m2a_test_loss_per_epoch))
        
        if epoch % model_save_interval == 0 and save_weights == True:
            torch.save(m2a.state_dict(), "results/weights/m2a_weights_epoch_{}".format(epoch))
        
        loss_history["m2a train"].append(m2a_train_loss_per_epoch)
        loss_history["m2a test"].append(m2a_test_loss_per_epoch)
        
        print ('epoch {} : m2a train: {:01.4f} m2a test: {:01.4f} time {:01.2f}'.format(epoch + 1, m2a_train_loss_per_epoch, m2a_test_loss_per_epoch, time.time()-start))
    
        m2a_scheduler.step()
        
    return loss_history

# fit model
loss_history = train(train_loader, test_loader, epochs)

# save history
utils.save_loss_as_csv(loss_history, "results/histories/history_{}.csv".format(epochs))
utils.save_loss_as_image(loss_history, "results/histories/history_{}.png".format(epochs))

# save model weights
torch.save(m2a.state_dict(), "results/weights/m2a_weights_epoch_{}".format(epochs))

"""
Inference
"""


poseRenderer = PoseRenderer(edge_list)

# visualization settings
view_ele = 90.0
view_azi = -90.0
view_line_width = 1.0
view_size = 4.0


def forward_kinematics(joint_rotations, root_positions):
    

    joint_rotations = joint_rotations.reshape(joint_rotations.shape[0], joint_rotations.shape[1], joint_count, 4)

    toffsets = torch.tensor(offsets).to(device)
    
    positions_world = []
    rotations_world = []

    expanded_offsets = toffsets.expand(joint_rotations.shape[0], joint_rotations.shape[1], offsets.shape[0], offsets.shape[1])

    # Parallelize along the batch and time dimensions
    for jI in range(offsets.shape[0]):
        if parents[jI] == -1:
            positions_world.append(root_positions)
            rotations_world.append(joint_rotations[:, :, 0])
        else:
            positions_world.append(qrot(rotations_world[parents[jI]], expanded_offsets[:, :, jI]) \
                                   + positions_world[parents[jI]])
            if len(children[jI]) > 0:
                rotations_world.append(qmul(rotations_world[parents[jI]], joint_rotations[:, :, jI]))
            else:
                # This joint is a terminal node -> it would be useless to compute the transformation
                rotations_world.append(None)

    return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)

def create_ref_audio_excerpt(excerpt_index):
    mel_excerpt_norm = audio_excerpts[excerpt_index]
    mel_excerpt_norm = torch.permute(mel_excerpt_norm, (1, 0))
    mel_excerpt = mel_excerpt_norm * audio_data_std + audio_data_mean

    audio_specs = inv_mel_transform(mel_excerpt)
    audio_excerpt = griffin_lim(audio_specs)
    audio_excerpt = audio_excerpt.unsqueeze(dim=0)

    torchaudio.save("ref_audio_excerpt_{}.wav".format(excerpt_index), audio_excerpt, audio_sample_rate)
    
def create_pred_audio_excerpt(excerpt_index):
    
    m2a.eval()
    mocap_excerpt_norm = mocap_excerpts[excerpt_index]
    mocap_excerpt_norm = mocap_excerpt_norm.unsqueeze(0).to(device)
    mel_excerpt_norm = m2a(mocap_excerpt_norm)
    mel_excerpt_norm = mel_excerpt_norm.detach().cpu().squeeze(0)
    mel_excerpt_norm = torch.permute(mel_excerpt_norm, (1, 0))
    mel_excerpt = mel_excerpt_norm * audio_data_std + audio_data_mean
    audio_specs = inv_mel_transform(mel_excerpt)
    audio_excerpt = griffin_lim(audio_specs)
    audio_excerpt = audio_excerpt.unsqueeze(dim=0)
    
    torchaudio.save("pred_audio_excerpt_{}.wav".format(test_excerpt_index), audio_excerpt, audio_sample_rate)
    
    m2a.train()
    
def create_mocap_anim(start_frame, end_frame):

    mocap_region_norm = mocap_sequence[start_frame:end_frame]
    mocap_region = mocap_region_norm * mocap_std + mocap_mean
    mocap_region = mocap_region.reshape(1, mocap_region.shape[0], joint_count, joint_dim).to(device)
    
    zero_trajectory = torch.tensor(np.zeros((1, mocap_region.shape[1], 3), dtype=np.float32)).to(device)
    
    mocap_region_jointpos = forward_kinematics(mocap_region, zero_trajectory)
    
    mocap_region_jointpos = mocap_region_jointpos.detach().cpu().squeeze(0).numpy()
    view_min, view_max = utils.get_equal_mix_max_positions(mocap_region_jointpos)
    mocap_region_images = poseRenderer.create_pose_images(mocap_region_jointpos, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    
    mocap_region_images[0].save("mocap_region_{}-{}.gif".format(start_frame, end_frame), save_all=True, append_images=mocap_region_images[1:], optimize=False, duration=33.0, loop=0) 
    

def create_pred_audio(start_frame, end_frame, window_offset):

    m2a.eval()
    
    frame_count = end_frame - start_frame
    corr_frame_count = (frame_count - sequence_length) // window_offset
    corr_frame_count = corr_frame_count * window_offset + sequence_length
    end_frame = start_frame + corr_frame_count
    
    mocap_region_norm = mocap_sequence[start_frame:end_frame]
    audio_region = torch.zeros(int(corr_frame_count / mocap_fps * audio_sample_rate), dtype=torch.float32)
    
    for fI in range(0, end_frame - start_frame - sequence_length, window_offset):
        
        sI = int(fI / mocap_fps * audio_sample_rate)
        
        print("fI ", fI, " sI ", sI)
        
        mocap_excerpt_norm = mocap_region_norm[fI:fI + sequence_length]
        mocap_excerpt_norm = mocap_excerpt_norm.unsqueeze(0).to(device)
        
        print("mocap_excerpt_norm s ", mocap_excerpt_norm.shape)
        
        mel_excerpt_norm = m2a(mocap_excerpt_norm)
        
        print("mel_excerpt_norm s ", mel_excerpt_norm.shape)
        
        mel_excerpt_norm = mel_excerpt_norm.detach().cpu().squeeze(0)
        mel_excerpt_norm = torch.permute(mel_excerpt_norm, (1, 0))
        mel_excerpt = mel_excerpt_norm * audio_data_std + audio_data_mean
        
        print("mel_excerpt s ", mel_excerpt.shape)
        
        audio_specs = inv_mel_transform(mel_excerpt)
        audio_excerpt = griffin_lim(audio_specs)
        
        print("audio_excerpt s ", audio_excerpt.shape)
    
        audio_window_env = torch.from_numpy(np.hanning(audio_excerpt.shape[0])).to(torch.float32)
    
        print("audio_window_env s ", audio_window_env.shape)
        
        audio_region[sI:sI + audio_excerpt.shape[0]] += audio_excerpt * audio_window_env
        
        
    audio_region = audio_region.unsqueeze(dim=0)
    
    torchaudio.save("pred_audio_region_{}-{}.wav".format(test_start_frame, test_end_frame), audio_region, audio_sample_rate)  
        

# create ref audio excerpt    

test_excerpt_index = 1000
    
create_ref_audio_excerpt(test_excerpt_index)

# create pred audio excerpt    

test_excerpt_index = 1000
    
create_pred_audio_excerpt(test_excerpt_index)

# create mocap region
    
test_start_frame = 1000
test_end_frame = 2000  

create_mocap_anim(test_start_frame, test_end_frame)

# create pred audio region

test_start_frame = 1000
test_end_frame = 2000  
window_offset = 64

create_pred_audio(test_start_frame, test_end_frame, window_offset)

