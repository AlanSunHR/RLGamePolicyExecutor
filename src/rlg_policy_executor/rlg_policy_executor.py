# MIT License

# Copyright (c) 2022 Haoran Sun

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from gym import spaces
from rl_games.torch_runner import Runner, _restore
import numpy as np
import yaml

class RLGPolicyExecutor(object):
    def __init__(self, 
                 train_config, 
                 trained_pth, 
                 num_actions, 
                 num_obs, 
                 is_deterministic=True,
                 device=None,
                 seed=None,
                 num_agents=1) -> None:
        '''
            @Params:
            train_config: the path to the training config file
            trained_pth: the pth file trained following the pipeline of rl_games
            num_actions: number of actions of your agent
            num_obs: number of observations of your agent
            is_deterministic: True if your policy is deterministic, keep it the same as that in your simulation
            device: inference device in string format, i.e. 'cuda:0' referes to 
            your first CPU, or 'cpu' if you want to run on your CPU. (NOTE: the inference
            device should be the same as the rl_device to train your agent's policy); if it is None, value defined
            in train_config will be used (NOTE: for IsaacGymEnvs/OmniIsaacGymEnvs, please assign the device name, the default
            {...rl_device} string will not be parsed. )
            seed: random seed for np.random and torch.random
            num_agents: number of agents in your environment, should be the same as your simulation settings
        '''
        self.train_config = train_config
        self.trained_pth = trained_pth
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.is_deterministic = is_deterministic
        self.device = device
        self.seed = seed
        self.num_agents = num_agents

        # Config obs and action space
        self.env_info = self._getEnvInfo()

        # initialize runner and player
        self._fake_runner = Runner(None)
        self._fake_player = None
        with open(self.train_config, 'r') as config_stream:
            config = yaml.safe_load(config_stream)

            # Append necessary config to the end of the config
            config['params']['config'] = {**config['params']['config'], 
                                          **self.env_info,
                                          'reward_shape': {}}
            config['params']['seed'] = self.seed
            if self.device is not None:
                config['params']['config']['device'] = device
                config['params']['config']['device_name'] = device

            self._fake_runner.load(config)
            play_args = {
            'train': False,
            'play': True,
            'checkpoint': self.trained_pth,
            'sigma': None
            }
            self._fake_player = self._fake_runner.create_player()
            _restore(self._fake_player, play_args)
            # In OmniIsaacGymEnvs, has_batch_dimension is always True
            self._fake_player.has_batch_dimension = True

    def _getEnvInfo(self):
        info = {}
        info['action_space'] = spaces.Box(np.ones(self.num_actions) * -1.0, 
                                                              np.ones(self.num_actions) * 1.0)
        info['observation_space'] = spaces.Box(np.ones(self.num_obs) * -np.Inf, 
                                               np.ones(self.num_obs) * np.Inf)

        info['agents'] = self.num_agents

        return {'env_info': info}

    def getAction(self, obs_tensor):
        '''
            @Params:
            obs_tensor: the input observation tensor to be fed into the policy
            network, should be of the same type as that of your simulation env.
        '''
        return self._fake_player.get_action(obs_tensor, self.is_deterministic)

        