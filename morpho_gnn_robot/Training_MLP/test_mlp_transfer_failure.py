import torch
import torch.nn as nn
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'core')))
from robot_env_bullet import RobotEnvBullet
import os
import glob
import time
import numpy as np

class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, action_dim))
        self.critic = nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1))
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))

    def forward(self, obs):
        mean = self.actor(obs)
        val = self.critic(obs)
        return (mean, val)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../morpho_gnn_robot/
HEXAPOD_URDF = os.path.join(BASE_DIR, '..', 'URDFs', 'hexapod_anymal.urdf')
MLP_CHECKPOINT_DIR = os.path.join(BASE_DIR, 'Training_MLP', 'checkpoints')
print('=====================================================')
print('  MULTI-LAYER PERCEPTRON (MLP) MORPHOLOGY TRANSFER   ')
print('=====================================================')
checkpoints = glob.glob(os.path.join(MLP_CHECKPOINT_DIR, '*.pt'))
if not checkpoints:
    print(f'No checkpoints found in {MLP_CHECKPOINT_DIR}')
    exit(1)
latest_checkpoint = max(checkpoints, key=os.path.getmtime)
print(f'-> Selected trained Quadruped MLP Checkpoint: {os.path.basename(latest_checkpoint)}')
print(f'-> Connecting to Hexapod PyBullet Environment...')
env = RobotEnvBullet(urdf_path=HEXAPOD_URDF, render_mode=None)
num_joints = 18
HEXAPOD_OBS_DIM = num_joints * 2 + 3 + 3 + 4 + 3
print(f'-> Hexapod mathematically dictates an Observation Dimension of: {HEXAPOD_OBS_DIM}')
print(f'-> Hexapod mathematically dictates an Action Dimension of     : {num_joints}')
print(f'-> Instantiating MLP to control the Hexapod...')
hexapod_mlp = MLPActorCritic(obs_dim=HEXAPOD_OBS_DIM, action_dim=num_joints)
QUADRUPED_OBS_DIM = 37
QUADRUPED_ACT_DIM = 12
print('\n-----------------------------------------------------')
print(f'ATTEMPTING ZERO-SHOT TRANSFER (Quadruped -> Hexapod)...')
print('-----------------------------------------------------')
state_dict = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
if 'agent' in state_dict:
    state_dict = state_dict['agent']
elif 'model_state_dict' in state_dict:
    state_dict = state_dict['model_state_dict']
try:
    hexapod_mlp.load_state_dict(state_dict, strict=True)
    print('SUCCESS: MLP Generalized successfully! (This should never happen)')
except Exception as e:
    print(f'❌ TRANSFER FAILED! ❌')
    print(f'The Dense MLP architecture is inherently rigid and shattered upon topology transfer.')
    print('\nDetailed PyTorch Shape Mismatch Error:')
    print(str(e).replace('size', 'DIMENSION').replace('match', 'MATCH'))
    print('\n-----------------------------------------------------')
    print('WHY DID THIS HAPPEN?')
    print("The Quadruped MLP's very first weight matrix is permanently fused to a 37-dimensional input.")
    print('The Hexapod generates 49 sensory signals. Because an MLP expects a flat, fixed-width vector,')
    print('the new 12 signals (middle leg positions/velocities) have nowhere to plug in.')
    print("Unlike the GNN, the MLP has no concept of 'joints' or 'edges'—only a hardcoded flat array.")
    print('To make the MLP work for the Hexapod, you would have to delete all weights and retrain from step 0.')
    print('=====================================================\n')