"""
simple rnn that maps a sequence of dance poses into sequence of audio encodings
the number of encodings and dance poses must be identical with each encoding having a corresponding dance pose
the encodings are created by a pre-trained audio autoencoder
the autoencoder is: audio/audio_autoencoder/aae_resnet
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
import audio_loss as al
import autoencoder as ae

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

mocap_data_path = "D:/data/mocap/Motion2Audio/stocos/Take_3_50fps_crop.bvh"
mocap_fps = 50
mocap_dim = -1 # automatically determined when loading mocap data

"""
Audio Settings
"""

audio_file = "D:/Data/audio/Motion2Audio/stocos/Take3_RO_37-4-1_HQ_audio_crop_32khz.wav"
audio_sample_rate = 32000
audio_channels = 1

audio_windows_per_second = 50
audio_window_length = audio_sample_rate // audio_windows_per_second * 2
audio_standardise = True
audio_encoding_standardise = True

"""
Dataset Settings
"""

valid_frame_start = 0
valid_frame_end = 12350
batch_size = 32
test_percentage = 0.1

"""
Model Settings
"""

# audio autoencoder model

audio_ae_in_channels = audio_channels # Number of input channels
audio_ae_channels = 16 # Number of base channels 
audio_ae_multipliers = [1, 1, 2, 2] # Channel multiplier between layers (i.e. channels * multiplier[i] -> channels * multiplier[i+1])
audio_ae_factors = [8, 4, 4] # Downsampling/upsampling factor per layer
audio_ae_num_blocks = [2, 2, 2] # Number of resnet blocks per layer
audio_ae_patch_size = 1 # ??
audio_ae_resnet_groups = 8 # ??
audio_ae_out_channels = audio_ae_in_channels # ??
audio_ae_bottleneck = ae.TanhBottleneck
audio_ae_bottleneck_channels = None # 

audio_ae_encoder_weights_file = "results/weights/encoder_weights_epoch_400"
audio_ae_decoder_weights_file = "results/weights/decoder_weights_epoch_400"

# motion to audio encoding model

sequence_length = 128
m2a_rnn_layer_count = 2
m2a_rnn_layer_size = 512
m2a_dense_layer_sizes = [ 512 ]

save_models = False
save_tscript = False
save_weights = True

# load model weights
load_weights = False
m2a_weights_file = "results/weights/m2a_weights_epoch_200"


"""
Training Settings
"""

learning_rate = 1e-4
model_save_interval = 50
save_weights = True
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
total_mocap_sequence_length = mocap_sequence.shape[0]
joint_count = mocap_sequence.shape[1]
joint_dim = mocap_sequence.shape[2]
pose_dim = joint_count * joint_dim
mocap_dim = pose_dim

mocap_sequence = mocap_sequence.reshape(total_mocap_sequence_length, mocap_dim)
mocap_sequence = mocap_sequence[valid_frame_start:valid_frame_end,...]
mocap_sequence = torch.tensor(mocap_sequence)

# normalize mocap sequence
mocap_mean = torch.mean(mocap_sequence, dim=0, keepdim=True).to(device)
mocap_std = torch.std(mocap_sequence, dim=0, keepdim=True).to(device)


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

# normalise audio
audio_mean = torch.mean(waveform_data).to(device)
audio_std = torch.std(waveform_data).to(device)

# create sequence of audio excerpts (each excerpt has the length required for the autoencoder to work with)
audio_sequence = []

for aI in range(0, waveform_data.shape[1] - audio_window_length, audio_sample_rate // audio_windows_per_second):
    audio_excerpt = waveform_data[:, aI:aI+audio_window_length]
    audio_sequence.append(audio_excerpt)
    
audio_sequence = torch.concat(audio_sequence, dim=0)
audio_sequence = audio_sequence[valid_frame_start:valid_frame_end,...]


"""
Create Audio Autoencoder
"""

audio_encoder = ae.Encoder1d(
    in_channels=audio_ae_in_channels,
    out_channels=audio_ae_bottleneck_channels,
    channels=audio_ae_channels,
    multipliers=audio_ae_multipliers,
    factors=audio_ae_factors,
    num_blocks=audio_ae_num_blocks,
    patch_size=audio_ae_patch_size,
    resnet_groups=audio_ae_resnet_groups,
    bottleneck=audio_ae_bottleneck()).to(device)

audio_encoder.load_state_dict(torch.load(audio_ae_encoder_weights_file))

audio_encoder_in = torch.randn(1, audio_channels, audio_window_length).to(device)
audio_encoder_out = audio_encoder(audio_encoder_in)
latent_dim = audio_encoder_out.shape[1]

audio_encoding_dim = audio_encoder_out.shape[1]
audio_encoding_count = audio_encoder_out.shape[2]

print("audio_encoder_in s ", audio_encoder_in.shape)
print("audio_encoder_out s ", audio_encoder_out.shape)

audio_decoder = ae.Decoder1d(
    in_channels=audio_ae_bottleneck_channels,
    out_channels=audio_ae_out_channels,
    channels=audio_ae_channels,
    multipliers=audio_ae_multipliers[::-1],
    factors=audio_ae_factors[::-1],
    num_blocks=audio_ae_num_blocks[::-1],
    patch_size=audio_ae_patch_size,
    resnet_groups=audio_ae_resnet_groups).to(device)

audio_decoder.load_state_dict(torch.load(audio_ae_decoder_weights_file))

audio_decoder_in = audio_encoder_out
audio_decoder_out = audio_decoder(audio_decoder_in)

print("audio_decoder_in s ", audio_decoder_in.shape)
print("audio_decoder_out s ", audio_decoder_out.shape)


"""
# test audio autoencoder

def create_autoencoder_pred_sonification(waveform, file_name):
    
    if audio_standardise:
        waveform = (waveform - audio_mean) / audio_std

    grain_env = torch.hann_window(audio_window_length)
    grain_offset = audio_window_length // 2
    
    waveform = waveform[0,:]
    predict_start_window = 0
    predict_window_count = int(waveform.shape[0] // grain_offset)

    pred_audio_sequence_length = (predict_window_count - 1) * grain_offset + audio_window_length
    pred_audio_sequence = torch.zeros((pred_audio_sequence_length), dtype=torch.float32)

    for i in range(predict_window_count - 1):
        target_audio = waveform[predict_start_window * audio_window_length + i*grain_offset:predict_start_window * audio_window_length + i*grain_offset + audio_window_length]
        
        #target_audio = torch.from_numpy(target_audio).to(device)
        target_audio = target_audio.to(device)
        target_audio = target_audio.reshape((1, 1, -1))
        
        with torch.no_grad():
            
            #print("target_audio s ", target_audio.shape)
            
            encoder_output = audio_encoder(target_audio)
            pred_audio = audio_decoder(encoder_output)
            

            
        pred_audio = torch.flatten(pred_audio.detach().cpu())
    
        pred_audio = pred_audio * grain_env

        pred_audio_sequence[i*grain_offset:i*grain_offset + audio_window_length] = pred_audio_sequence[i*grain_offset:i*grain_offset + audio_window_length] + pred_audio

    if audio_standardise:
        pred_audio_sequence = audio_mean.cpu() + pred_audio_sequence * audio_std.cpu()


    torchaudio.save(file_name, torch.reshape(pred_audio_sequence.detach().cpu(), (1, -1)), audio_sample_rate)
    
    

audio_encoder.eval()
audio_decoder.eval()

test_waveform_data, _ = torchaudio.load(audio_file)

create_autoencoder_pred_sonification(test_waveform_data[:, audio_sample_rate*100:audio_sample_rate*110].to(device), "autoencoder_test.wav")
"""

# create audio encoding sequence

audio_encoder.eval()

audio_encoding_sequence = []

for audio_excerpt in audio_sequence:
    
    audio_excerpt_norm = (audio_excerpt.to(device) - audio_mean) // audio_std 
    audio_excerpt_norm = audio_excerpt_norm.reshape(1, 1, -1)
    audio_encoding =  audio_encoder(audio_excerpt_norm)
    
    audio_encoding = audio_encoding.detach().cpu()
    
    audio_encoding_sequence.append(audio_encoding)
    
audio_encoding_sequence = torch.concat(audio_encoding_sequence, dim=0)
    
print("audio_encoding_sequence s ", audio_encoding_sequence.shape)

# normalise audio encodings

tmp_audio_encoding_sequence = torch.transpose(audio_encoding_sequence, 1, 2).reshape(-1, audio_encoding_sequence.shape[1])
audio_encoding_mean = torch.mean(tmp_audio_encoding_sequence, dim=0).reshape(1, -1, 1).to(device)
audio_encoding_std = torch.std(tmp_audio_encoding_sequence, dim=0).reshape(1, -1, 1).to(device)


"""
Create Dataset
"""

total_sequence_length = mocap_sequence.shape[0]

mocap_excerpts = []
audio_excerpts = []

for pI in range(total_sequence_length - sequence_length):
    
    mocap_excerpt = mocap_sequence[pI:pI+sequence_length]
    mocap_excerpts.append(mocap_excerpt.reshape((1, sequence_length, mocap_dim)))
    
    audio_excerpt = audio_sequence[pI:pI+sequence_length]
    audio_excerpts.append(audio_excerpt.reshape((1, sequence_length, audio_window_length)))

mocap_excerpts = torch.cat(mocap_excerpts, dim=0)
audio_excerpts = torch.cat(audio_excerpts, dim=0)

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
Create Mocap 2 Audio Encoding Model
"""

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
    

    
m2a = Mocap2AudioModel(sequence_length, mocap_dim, audio_encoding_dim * audio_encoding_count, m2a_rnn_layer_count, m2a_rnn_layer_size, m2a_dense_layer_sizes).to(device)

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
m2a_scheduler = torch.optim.lr_scheduler.StepLR(m2a_optimizer, step_size=100, gamma=0.5) # reduce the learning every 20 epochs by a factor of 10

l1_loss = nn.L1Loss()

def audio_loss(y_audio, yhat_audio):
    
    _loss = l1_loss(yhat_audio, y_audio)
    
    return _loss

def audio_encoding_loss(y_enc, yhat_env):
    
    _loss = l1_loss(yhat_env, y_enc)
    
    return _loss

def m2a_train_step(batch_mocap, batch_audio):
    
    #print("batch_mocap s ", batch_mocap.shape)
    #print("batch_audio s ", batch_audio.shape)
    
    target_audio = batch_audio
    
    # normalise mocap
    batch_mocap_norm = (batch_mocap - mocap_mean.unsqueeze(0)) / mocap_std.unsqueeze(0)
    batch_mocap_norm = torch.nan_to_num(batch_mocap_norm)
    
    #print("batch_mocap_norm s ", batch_mocap_norm.shape)
    
    # normalise audio
    batch_audio_norm = (batch_audio - audio_mean) / audio_std
    batch_audio_norm = torch.nan_to_num(batch_audio_norm)
    target_audio_norm = batch_audio_norm
    
    #print("target_audio_norm s ", target_audio_norm.shape)
    
    # create target audio encodings
    with torch.no_grad():
        target_audio_norm = target_audio_norm.reshape(-1, 1, audio_window_length)
        
        #print("target_audio_norm 2 s ", target_audio_norm.shape)
        
        target_audio_encoding = audio_encoder(target_audio_norm)
        
        #print("target_audio_encoding s ", target_audio_encoding.shape)
        
        target_audio_encoding = torch.transpose(target_audio_encoding, 1, 2).reshape(-1, audio_encoding_dim)

        #print("target_audio_encoding 2 s ", target_audio_encoding.shape)
    
    # predict audio encodings
    pred_audio_encoding_norm = m2a(batch_mocap_norm)
    
    #print("pred_audio_encoding_norm s ", pred_audio_encoding_norm.shape)
    
    pred_audio_encoding_norm = pred_audio_encoding_norm.reshape(-1, audio_encoding_dim, audio_encoding_count)
    
    #print("pred_audio_encoding_norm 2 s ", pred_audio_encoding_norm.shape)
    
    pred_audio_encoding = pred_audio_encoding_norm * audio_encoding_std + audio_encoding_mean
    
    #print("pred_audio_encoding 3 s ", pred_audio_encoding.shape)
    
    # create pred audio
    with torch.no_grad():
        pred_audio_norm = audio_decoder(pred_audio_encoding)
        
        #print("pred_audio_norm 3 s ", pred_audio_norm.shape)
        
        pred_audio = pred_audio_norm * audio_std + audio_mean
    
    
    #print("pred_audio_encoding s ", pred_audio_encoding.shape)

    # calculate audio encoding loss
    pred_audio_encoding = torch.transpose(pred_audio_encoding, 1, 2).reshape(-1, audio_encoding_dim)
    
    #print("target_audio_encoding s ", target_audio_encoding.shape)
    #print("pred_audio_encoding s ", pred_audio_encoding.shape)
    _loss_encoding = audio_encoding_loss(target_audio_encoding, pred_audio_encoding) 
    
    # calculate audio loss
    target_audio = target_audio.reshape(-1, audio_window_length)
    pred_audio = pred_audio.reshape(-1, audio_window_length)
    
    #print("target_audio s ", target_audio.shape)
    #print("pred_audio s ", pred_audio.shape)
    _loss_audio = audio_loss(target_audio, pred_audio) 
    
    # calculate total loss
    _loss = _loss_encoding + _loss_audio
    
    m2a_optimizer.zero_grad()
    _loss.backward()
    
    m2a_optimizer.step()
    
    return _loss

m2a_train_step(mocap_batch.to(device), audio_batch.to(device))

def m2a_test_step(batch_mocap, batch_audio):
    
    target_audio = batch_audio
    
    # normalise mocap
    batch_mocap_norm = (batch_mocap - mocap_mean.unsqueeze(0)) / mocap_std.unsqueeze(0)
    batch_mocap_norm = torch.nan_to_num(batch_mocap_norm)

    # normalise audio
    batch_audio_norm = (batch_audio - audio_mean) / audio_std
    batch_audio_norm = torch.nan_to_num(batch_audio_norm)
    target_audio_norm = batch_audio_norm

    with torch.no_grad():
        # create target audio encodings
        target_audio_norm = target_audio_norm.reshape(-1, 1, audio_window_length)
        target_audio_encoding = audio_encoder(target_audio_norm)
        target_audio_encoding = torch.transpose(target_audio_encoding, 1, 2).reshape(-1, audio_encoding_dim)

        # predict audio encodings
        pred_audio_encoding_norm = m2a(batch_mocap_norm)
        pred_audio_encoding_norm = pred_audio_encoding_norm.reshape(-1, audio_encoding_dim, audio_encoding_count)
        pred_audio_encoding = pred_audio_encoding_norm * audio_encoding_std + audio_encoding_mean

        # create pred audio
        pred_audio_norm = audio_decoder(pred_audio_encoding)
        pred_audio = pred_audio_norm * audio_std + audio_mean
    
        # calculate audio encoding loss
        pred_audio_encoding = torch.transpose(pred_audio_encoding, 1, 2).reshape(-1, audio_encoding_dim)
        _loss_encoding = audio_encoding_loss(target_audio_encoding, pred_audio_encoding) 
    
        # calculate audio loss
        target_audio = target_audio.reshape(-1, audio_window_length)
        pred_audio = pred_audio.reshape(-1, audio_window_length)
        _loss_audio = audio_loss(target_audio, pred_audio) 
        
        # calculate total loss
        _loss = _loss_encoding + _loss_audio
    
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

def create_ref_mocap_excerpt(excerpt_index, file_name):
    mocap_excerpt = mocap_excerpts[excerpt_index].to(device)
    mocap_excerpt = mocap_excerpt.reshape(1, sequence_length, joint_count, joint_dim)
    
    zero_trajectory = torch.tensor(np.zeros((1, sequence_length, 3), dtype=np.float32)).to(device)
    
    mocap_excerpt_jointpos = forward_kinematics(mocap_excerpt, zero_trajectory)
    
    mocap_excerpt_jointpos = mocap_excerpt_jointpos.detach().cpu().squeeze(0).numpy()
    view_min, view_max = utils.get_equal_mix_max_positions(mocap_excerpt_jointpos)
    mocap_excerpt_images = poseRenderer.create_pose_images(mocap_excerpt_jointpos, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    
    mocap_excerpt_images[0].save(file_name, save_all=True, append_images=mocap_excerpt_images[1:], optimize=False, duration=33.0, loop=0) 

excerpt_index = 1000
    
create_ref_mocap_excerpt(excerpt_index, "ref_mocap_excerpt_{}.gif".format(excerpt_index))  

def create_ref_audio_excerpt(excerpt_index, file_name):
    
    audio_excerpt = audio_excerpts[excerpt_index]
    audio_amp_window = torch.hann_window(audio_window_length, dtype=torch.float32)
    
    audio_window_offset_size = audio_window_length // 2
    
    result_audio_length = audio_window_length + (audio_excerpt.shape[0] - 1) * audio_window_offset_size
    
    result_audio = torch.zeros((result_audio_length), dtype=torch.float32)
    
    for aI in range(sequence_length):
        
        audio_excerpt_window = audio_excerpt[aI]
        
        insert_start = aI * audio_window_offset_size
        insert_end = insert_start + audio_window_length
        
        result_audio[insert_start:insert_end] += audio_excerpt_window * audio_amp_window

    result_audio = result_audio.unsqueeze(dim=0)

    torchaudio.save(file_name, result_audio, audio_sample_rate)
    
excerpt_index = 1000
    
create_ref_audio_excerpt(excerpt_index, "ref_audio_excerpt_{}.wav".format(excerpt_index))  

def create_ae_audio_excerpt(excerpt_index, file_name):
    
    audio_excerpt = audio_excerpts[excerpt_index].to(device)
    
    # normalise audio excerpt
    audio_excerpt_norm = (audio_excerpt - audio_mean) / audio_std
    audio_excerpt_norm = torch.nan_to_num(audio_excerpt_norm)
    audio_excerpt_norm = audio_excerpt_norm.reshape(sequence_length, 1, audio_window_length)
    
    #print("audio_excerpt_norm s ", audio_excerpt_norm.shape)
    
    audio_ae_norm = audio_decoder(audio_encoder(audio_excerpt_norm))
    
    #print("audio_ae_norm s ", audio_ae_norm.shape)
    
    audio_ae_norm = audio_ae_norm.detach()
    audio_ae_norm = audio_ae_norm.reshape(sequence_length, audio_window_length)
    audio_ae = audio_ae_norm * audio_std + audio_mean
    
    audio_ae = audio_ae.cpu()
    
    audio_amp_window = torch.hann_window(audio_window_length, dtype=torch.float32)
    
    audio_window_offset_size = audio_window_length // 2
    
    result_audio_length = audio_window_length + (audio_excerpt.shape[0] - 1) * audio_window_offset_size
    
    result_audio = torch.zeros((result_audio_length), dtype=torch.float32)
    
    for aI in range(sequence_length):
        
        audio_excerpt_window = audio_ae[aI]
        
        insert_start = aI * audio_window_offset_size
        insert_end = insert_start + audio_window_length
        
        result_audio[insert_start:insert_end] += audio_excerpt_window * audio_amp_window

    result_audio = result_audio.unsqueeze(dim=0)

    torchaudio.save(file_name, result_audio, audio_sample_rate)
    
excerpt_index = 1000
    
create_ae_audio_excerpt(excerpt_index, "ae_audio_excerpt_{}.wav".format(excerpt_index))  

def create_m2a_audio_excerpt(excerpt_index, file_name):
    
    m2a.eval()
    
    mocap_excerpt = mocap_excerpts[excerpt_index].to(device)
    
    # normalise mocap
    mocap_excerpt_norm = (mocap_excerpt - mocap_mean) / mocap_std
    mocap_excerpt_norm = torch.nan_to_num(mocap_excerpt_norm)
    
    # predict audio encodings
    pred_audio_encoding_norm = m2a(mocap_excerpt_norm.unsqueeze(0))
    pred_audio_encoding_norm = pred_audio_encoding_norm.squeeze(0)
    
    # denormalise audio encodings
    pred_audio_encoding_norm = pred_audio_encoding_norm.reshape(-1, audio_encoding_dim, audio_encoding_count)
    pred_audio_encoding = pred_audio_encoding_norm * audio_encoding_std + audio_encoding_mean
    
    # create pred audio
    audio_m2a_norm = audio_decoder(pred_audio_encoding)
    audio_m2a_norm = audio_m2a_norm.detach()
    audio_m2a_norm = audio_m2a_norm.reshape(sequence_length, audio_window_length)
    audio_m2a = audio_m2a_norm * audio_std + audio_mean
    audio_m2a = audio_m2a.cpu()
    
    audio_amp_window = torch.hann_window(audio_window_length, dtype=torch.float32)
    
    audio_window_offset_size = audio_window_length // 2
    
    result_audio_length = audio_window_length + (audio_excerpt.shape[0] - 1) * audio_window_offset_size
    
    result_audio = torch.zeros((result_audio_length), dtype=torch.float32)
    
    for aI in range(sequence_length):
        
        audio_excerpt_window = audio_m2a[aI]
        
        insert_start = aI * audio_window_offset_size
        insert_end = insert_start + audio_window_length
        
        result_audio[insert_start:insert_end] += audio_excerpt_window * audio_amp_window

    result_audio = result_audio.unsqueeze(dim=0)

    torchaudio.save(file_name, result_audio, audio_sample_rate)
 
    m2a.train()
    
excerpt_index = 1000
    
create_m2a_audio_excerpt(excerpt_index, "m2a_audio_excerpt_{}.wav".format(excerpt_index))  
    
def create_ref_audio(start_frame, end_frame, file_name):
    
    audio_excerpt = audio_sequence[start_frame:end_frame, :]
    audio_amp_window = torch.hann_window(audio_window_length, dtype=torch.float32)
    
    audio_window_offset_size = audio_window_length // 2

    result_audio_length = audio_window_length + (audio_excerpt.shape[0] - 1) * audio_window_offset_size
    
    result_audio = torch.zeros((result_audio_length), dtype=torch.float32)
    
    for aI in range(audio_excerpt.shape[0]):
        
        audio_excerpt_window = audio_excerpt[aI]
        
        insert_start = aI * audio_window_offset_size
        insert_end = insert_start + audio_window_length
        
        result_audio[insert_start:insert_end] += audio_excerpt_window * audio_amp_window
    
    result_audio = result_audio.unsqueeze(dim=0)
    
    torchaudio.save(file_name, result_audio, audio_sample_rate)

start_frame = 1000
end_frame = 2000
    
create_ref_audio(start_frame, end_frame, "ref_audio_{}-{}.wav".format(start_frame, end_frame))  
    
def create_mocap_anim(start_frame, end_frame, file_name):

    mocap_region = mocap_sequence[start_frame:end_frame].to(device)
    mocap_region = mocap_region.reshape(1, mocap_region.shape[0], joint_count, joint_dim).to(device)
    
    zero_trajectory = torch.tensor(np.zeros((1, mocap_region.shape[1], 3), dtype=np.float32)).to(device)
    
    mocap_region_jointpos = forward_kinematics(mocap_region, zero_trajectory)
    
    mocap_region_jointpos = mocap_region_jointpos.detach().cpu().squeeze(0).numpy()
    view_min, view_max = utils.get_equal_mix_max_positions(mocap_region_jointpos)
    mocap_region_images = poseRenderer.create_pose_images(mocap_region_jointpos, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    
    mocap_region_images[0].save(file_name, save_all=True, append_images=mocap_region_images[1:], optimize=False, duration=33.0, loop=0) 
 
start_frame = 1000
end_frame = 2000
    
create_mocap_anim(start_frame, end_frame, "ref_mocap_{}-{}.gif".format(start_frame, end_frame))  

def create_ae_audio(start_frame, end_frame, file_name):
    
    audio_excerpt = audio_sequence[start_frame:end_frame, :].to(device)
    
    # normalise audio excerpt
    audio_excerpt_norm = (audio_excerpt - audio_mean) / audio_std
    audio_excerpt_norm = torch.nan_to_num(audio_excerpt_norm)
    audio_excerpt_norm = audio_excerpt_norm.reshape(audio_excerpt.shape[0], 1, audio_window_length)

    #print("audio_excerpt_norm s ", audio_excerpt_norm.shape)
    
    audio_ae_norm = audio_decoder(audio_encoder(audio_excerpt_norm))
    
    #print("audio_ae_norm s ", audio_ae_norm.shape)

    audio_ae_norm = audio_ae_norm.detach()
    audio_ae_norm = audio_ae_norm.reshape(audio_ae_norm.shape[0], audio_window_length)
    audio_ae = audio_ae_norm * audio_std + audio_mean

    audio_ae = audio_ae.cpu()
    
    audio_amp_window = torch.hann_window(audio_window_length, dtype=torch.float32)
    
    audio_window_offset_size = audio_window_length // 2
    
    result_audio_length = audio_window_length + (audio_excerpt.shape[0] - 1) * audio_window_offset_size
    
    result_audio = torch.zeros((result_audio_length), dtype=torch.float32)
    
    for aI in range(audio_ae.shape[0]):
        
        audio_excerpt_window = audio_ae[aI]
        
        insert_start = aI * audio_window_offset_size
        insert_end = insert_start + audio_window_length
        
        result_audio[insert_start:insert_end] += audio_excerpt_window * audio_amp_window

    result_audio = result_audio.unsqueeze(dim=0)

    torchaudio.save(file_name, result_audio, audio_sample_rate)
 
start_frame = 1000
end_frame = 2000
    
create_ae_audio(start_frame, end_frame, "ae_audio_{}-{}.wav".format(start_frame, end_frame))         

mocap_mean.shape

def create_m2a_audio(start_frame, end_frame, file_name):
    
    m2a.eval()
    
    mocap_window_offset_size = sequence_length // 2
    audio_window_offset_size = audio_window_length // 2
    
    print("mocap_window_offset_size ", mocap_window_offset_size)
    print("audio_window_offset_size ", audio_window_offset_size)
    
    frame_count = end_frame - start_frame
    corr_frame_count = (frame_count - sequence_length) // mocap_window_offset_size
    corr_frame_count = corr_frame_count * mocap_window_offset_size + sequence_length
    end_frame = start_frame + corr_frame_count
    
    mocap_region = mocap_sequence[start_frame:end_frame].to(device)
    
    print("mocap_region s ", mocap_region.shape)
    
    # normalise mocap
    mocap_region_norm = (mocap_region - mocap_mean) / mocap_std
    mocap_region_norm = torch.nan_to_num(mocap_region_norm)
    
    # prepare result audio
    result_audio_length = audio_window_length + (corr_frame_count - 1) * audio_window_offset_size
    result_audio = torch.zeros((result_audio_length), dtype=torch.float32)
    
    print("result_audio s ", result_audio.shape)
    
    audio_amp_window = torch.hann_window(audio_window_length, dtype=torch.float32)

    for fI in range(0, mocap_region.shape[0] - sequence_length, mocap_window_offset_size):  

        mocap_excerpt_norm = mocap_region_norm[fI:fI+sequence_length]
        
        # predict audio encodings
        pred_audio_encoding_norm = m2a(mocap_excerpt_norm.unsqueeze(0))
        pred_audio_encoding_norm = pred_audio_encoding_norm.squeeze(0)
        
        # denormalise audio encodings
        pred_audio_encoding_norm = pred_audio_encoding_norm.reshape(-1, audio_encoding_dim, audio_encoding_count)
        pred_audio_encoding = pred_audio_encoding_norm * audio_encoding_std + audio_encoding_mean
        
        # create pred audio
        audio_m2a_norm = audio_decoder(pred_audio_encoding)
        audio_m2a_norm = audio_m2a_norm.detach()
        audio_m2a_norm = audio_m2a_norm.reshape(sequence_length, audio_window_length)
        audio_m2a = audio_m2a_norm * audio_std + audio_mean
        audio_m2a = audio_m2a.cpu()
        
        print("audio_m2a s ", audio_m2a.shape)
        
        for aI in range(sequence_length):
            
            print("aI ", aI)
            
            audio_excerpt_window = audio_m2a[aI]
            
            print("audio_excerpt_window s ", audio_excerpt_window.shape)
            
            insert_start = fI * audio_window_offset_size + aI * audio_window_offset_size
            insert_end = insert_start + audio_window_length
            
            print("insert_start ", insert_start, " insert_end ", insert_end)
        
            result_audio[insert_start:insert_end] += audio_excerpt_window * audio_amp_window
    

    result_audio = result_audio.unsqueeze(dim=0)

    torchaudio.save(file_name, result_audio, audio_sample_rate)
 
    m2a.train()
    

start_frame = 1000
end_frame = 2000
    
create_m2a_audio(start_frame, end_frame, "m2a_audio_{}-{}.wav".format(start_frame, end_frame))         
    

    




# test with different mocap file

def diff_create_ref_mocap(diff_mocap_sequence, start_frame, end_frame, file_name):

    mocap_region = diff_mocap_sequence[start_frame:end_frame].to(device)
    mocap_region = mocap_region.reshape(1, mocap_region.shape[0], joint_count, joint_dim).to(device)
    
    zero_trajectory = torch.tensor(np.zeros((1, mocap_region.shape[1], 3), dtype=np.float32)).to(device)
    
    mocap_region_jointpos = forward_kinematics(mocap_region, zero_trajectory)
    
    mocap_region_jointpos = mocap_region_jointpos.detach().cpu().squeeze(0).numpy()
    view_min, view_max = utils.get_equal_mix_max_positions(mocap_region_jointpos)
    mocap_region_images = poseRenderer.create_pose_images(mocap_region_jointpos, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    
    mocap_region_images[0].save(file_name, save_all=True, append_images=mocap_region_images[1:], optimize=False, duration=33.0, loop=0) 
     
def diff_create_m2a_audio(diff_mocap_sequence, start_frame, end_frame, file_name):
    
    m2a.eval()
    
    mocap_window_offset_size = sequence_length // 2
    audio_window_offset_size = audio_window_length // 2
    
    print("mocap_window_offset_size ", mocap_window_offset_size)
    print("audio_window_offset_size ", audio_window_offset_size)
    
    frame_count = end_frame - start_frame
    corr_frame_count = (frame_count - sequence_length) // mocap_window_offset_size
    corr_frame_count = corr_frame_count * mocap_window_offset_size + sequence_length
    end_frame = start_frame + corr_frame_count
    
    mocap_region = diff_mocap_sequence[start_frame:end_frame].to(device)
    
    print("mocap_region s ", mocap_region.shape)
    
    # normalise mocap
    mocap_region_norm = (mocap_region - mocap_mean) / mocap_std
    mocap_region_norm = torch.nan_to_num(mocap_region_norm)
    
    # prepare result audio
    result_audio_length = audio_window_length + (corr_frame_count - 1) * audio_window_offset_size
    result_audio = torch.zeros((result_audio_length), dtype=torch.float32)
    
    print("result_audio s ", result_audio.shape)
    
    audio_amp_window = torch.hann_window(audio_window_length, dtype=torch.float32)

    for fI in range(0, mocap_region.shape[0] - sequence_length, mocap_window_offset_size):  

        mocap_excerpt_norm = mocap_region_norm[fI:fI+sequence_length]
        
        # predict audio encodings
        pred_audio_encoding_norm = m2a(mocap_excerpt_norm.unsqueeze(0))
        pred_audio_encoding_norm = pred_audio_encoding_norm.squeeze(0)
        
        # denormalise audio encodings
        pred_audio_encoding_norm = pred_audio_encoding_norm.reshape(-1, audio_encoding_dim, audio_encoding_count)
        pred_audio_encoding = pred_audio_encoding_norm * audio_encoding_std + audio_encoding_mean
        
        # create pred audio
        audio_m2a_norm = audio_decoder(pred_audio_encoding)
        audio_m2a_norm = audio_m2a_norm.detach()
        audio_m2a_norm = audio_m2a_norm.reshape(sequence_length, audio_window_length)
        audio_m2a = audio_m2a_norm * audio_std + audio_mean
        audio_m2a = audio_m2a.cpu()
        
        print("audio_m2a s ", audio_m2a.shape)
        
        for aI in range(sequence_length):
            
            print("aI ", aI)
            
            audio_excerpt_window = audio_m2a[aI]
            
            print("audio_excerpt_window s ", audio_excerpt_window.shape)
            
            insert_start = fI * audio_window_offset_size + aI * audio_window_offset_size
            insert_end = insert_start + audio_window_length
            
            print("insert_start ", insert_start, " insert_end ", insert_end)
        
            result_audio[insert_start:insert_end] += audio_excerpt_window * audio_amp_window
    

    result_audio = result_audio.unsqueeze(dim=0)

    torchaudio.save(file_name, result_audio, audio_sample_rate)
 
    m2a.train()

        
#diff_mocap_data_path = "D:/data/mocap/Motion2Audio/anna_improvisation.bvh"
#diff_mocap_data_path = "D:/data/mocap/Motion2Audio/zachary_improvisation.bvh"
#diff_mocap_data_path = "D:/data/mocap/Motion2Audio/cristel_improvisation.bvh"
#diff_mocap_data_path = "D:/data/mocap/Motion2Audio/anna_improvisation.bvh"
#diff_mocap_data_path = "D:/data/mocap/Motion2Audio/madeline_improvisation.bvh"

diff_mocap_data_path = "D:/data/mocap/Motion2Audio/stocos/Take_1_50fps_crop.bvh"

diff_bvh_data = bvh_tools.load(diff_mocap_data_path)
diff_mocap_data = mocap_tools.bvh_to_mocap(diff_bvh_data)
diff_mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(diff_mocap_data["motion"]["rot_local_euler"], diff_mocap_data["rot_sequence"])
diff_mocap_sequence = diff_mocap_data["motion"]["rot_local"].astype(np.float32)
#diff_mocap_sequence = (diff_mocap_sequence - mocap_mean) / mocap_std
diff_mocap_sequence = diff_mocap_sequence.reshape(-1, mocap_dim)
diff_mocap_sequence = torch.tensor(diff_mocap_sequence)

test_start_frame = 1000
test_end_frame = 2000  

diff_create_ref_mocap(diff_mocap_sequence, test_start_frame, test_end_frame, "Take_1_diff_ref_mocap_{}-{}.gif".format(test_start_frame, test_end_frame))
diff_create_m2a_audio(diff_mocap_sequence, test_start_frame, test_end_frame, "Take_1_diff_m2a_audio_{}-{}.wav".format(test_start_frame, test_end_frame))
