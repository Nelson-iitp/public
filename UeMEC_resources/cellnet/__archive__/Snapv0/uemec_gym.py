from UEMEC.core import UeMEC, PARAMS
n_IOT, n_UAV, n_BSV = 1, 1, 1
max_episode_steps=200
import numpy as np
import gym
import gym.spaces
class UeMEC_GYM(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.uemec = UeMEC(device='cpu', params=PARAMS(n_BSV, n_UAV, n_IOT) , cap=0, meed=None, seed=None, logging="", frozen=False)

        self.action_space = gym.spaces.Box(low=np.zeros(self.uemec.nA)-1, high=np.zeros(self.uemec.nA)+1, dtype= np.float32)
        self.observation_space = gym.spaces.Box(low=np.zeros(self.uemec.nS)-np.inf, high=np.zeros(self.uemec.nS)+np.inf)
        self.max_episode_steps = max_episode_steps
        
    def reset(self):
        self.uemec.start()
        return self.uemec.state().numpy()
    
    def step(self, action):
        self.uemec.act( action )
        done = self.uemec.step()
        reward = self.uemec.R.item()
        return self.uemec.state().numpy(), reward, done, {}



        



import torch.nn as nn
global_policy_kwargs = dict(activation_fn=nn.LeakyReLU,
                                net_arch=[dict(pi=[256, 256, 256, 256], vf=[256, 256, 256, 256])]) #vf = value function


import argparse
import UEMEC.gymrl as gymrl
callers = {
    'main_ddpg': gymrl.main_ddpg,
    'main_ppo': gymrl.main_ppo,
    'main_sac': gymrl.main_sac,
    'main_a2c': gymrl.main_a2c,
    'main_td3': gymrl.main_td3,
}



if __name__ == '__main__':

    # get the args
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_timesteps', type=int,      default=100, help='Total timesteps in training')
    parser.add_argument('--log_interval', type=int,         default=10, help='log after these many steps')
    parser.add_argument('--save_as', type=str,              default='pie', help='name of policy file')

    parser.add_argument('--main', type=str,                 default='main_ppo', help='caller function name')
    parser.add_argument('--test', type=int,                 default=2, help='0 for training, 1 for test mode')
    
    args = parser.parse_args()

    #env = AIMAV() #gym.make("mav-v0")
    env = UeMEC_GYM()
    gymrl.check_env(env)
    model = callers[args.main](env, args.total_timesteps, args.log_interval, args.save_as, args.test, global_policy_kwargs)

    if args.test>1:
        import UEMEC.game as game
        game.play(env.uemec, model)

#   uemec_gym.py --total_timesteps=10_000 --log_interval=100 --save_as=pie --main=main_ppo --test=0