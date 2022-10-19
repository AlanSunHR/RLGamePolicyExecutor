# RLGameInference

## Introduction
The package is used to execute the policy trained using [rl_games](https://github.com/Denys88/rl_games) without initializing simulation environments, given the observation vectors.

## Pre-request
1. Install rl_games. Please refer to https://github.com/Denys88/rl_games.
2. You should have a config file containing the training configurations. For example in [OmniIsaacGymEnvs](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs), a typical training config should look like [AllegroHandPPO.yaml](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/blob/main/omniisaacgymenvs/cfg/train/AllegroHandPPO.yaml).
3. You should have a trained pth file, which contains the weight of the policy network.
4. Install gym, yaml:
```bash
    pip install gym pyyaml
```

## Installation
TODO

## Basic Usage
please refer to the example in test/test_policy_executor.py
