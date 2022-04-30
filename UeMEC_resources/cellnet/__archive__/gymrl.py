#=============================================================================================
# Impoter
#=============================================================================================

import numpy as np
from numpy.random import default_rng
#import matplotlib.pyplot as plt
from math import inf

import gym
from gym import Env, spaces

from stable_baselines3.common.env_checker import check_env
#from stable_baselines3.common.env_util import make_vec_env




#=============================================================================================
# uses - global_policy_kwargs
#=============================================================================================

def main_a2c(env, total_timesteps, log_interval, save_as, test, global_policy_kwargs):
    from stable_baselines3 import A2C

    if not test:
        model = A2C("MlpPolicy", env,  policy_kwargs=global_policy_kwargs, verbose=1)
        rewart = main_test(env, model, total_timesteps)
        print('Pre-Training. Total-Reward =', rewart)
        model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
        model.save(save_as)
        print('Training is finished')
        rewart = main_test(env, model, total_timesteps)
        print('Post-Training. Total-Reward =', rewart)
    else:
        model = A2C.load(save_as)
        rewart = main_test(env, model, total_timesteps)
        print('Testing is Finished. Total-Reward =', rewart)
    return model

def main_ppo(env, total_timesteps, log_interval, save_as, test, global_policy_kwargs):

    from stable_baselines3 import PPO

    if not test:
        model = PPO("MlpPolicy", env, policy_kwargs=global_policy_kwargs, verbose=1)
        rewart = main_test(env, model, total_timesteps)
        print('Pre-Training. Total-Reward =', rewart)
        model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
        model.save(save_as)
        print('Training is finished')
        rewart = main_test(env, model, total_timesteps)
        print('Post-Training. Total-Reward =', rewart)
    else:    
        model = PPO.load(save_as)
        rewart = main_test(env, model, total_timesteps)
        print('Testing is Finished. Total-Reward =', rewart)
    return model

#=============================================================================================
# uses - none
#=============================================================================================

def main_sac(env, total_timesteps, log_interval, save_as, test, global_policy_kwargs=None):
    from stable_baselines3 import SAC

    if not test:
        model = SAC("MlpPolicy", env, verbose=1)

        rewart = main_test(env, model, total_timesteps)
        print('Pre-Training. Total-Reward =', rewart)

        model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
        model.save(save_as)
        print('Training is finished')

        rewart = main_test(env, model, total_timesteps)
        print('Post-Training. Total-Reward =', rewart)
    else:
        model = SAC.load(save_as)
        rewart = main_test(env, model, total_timesteps)
        print('Testing is Finished. Total-Reward =', rewart)
   
    return model
#=============================================================================================
# uses - noise
#=============================================================================================

def main_ddpg(env, total_timesteps, log_interval, save_as, test, global_policy_kwargs=None):

    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
    
    if not test:
        # The noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions), theta=0.15, dt=0.01, initial_noise=None)
        # NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        
        model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)

        rewart = main_test(env, model, total_timesteps)
        print('Pre-Training. Total-Reward =', rewart)

        model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
        model.save(save_as)
        print('Training is finished.')

        rewart = main_test(env, model, total_timesteps)
        print('Post-Training. Total-Reward =', rewart)
    else:
        model = DDPG.load(save_as)
        rewart = main_test(env, model, total_timesteps)
        print('Testing is Finished. Total-Reward =', rewart)
    return model
    

def main_td3(env, total_timesteps, log_interval, save_as, test, global_policy_kwargs=None):
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

    if not test:
        # The noise objects for TD3
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = TD3("MlpPolicy", env, action_noise=action_noise,  verbose=1)

        rewart = main_test(env, model, total_timesteps)
        print('Pre-Training. Total-Reward =', rewart)

        model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
        model.save(save_as)

        print('Training is finished')

        rewart = main_test(env, model, total_timesteps)
        print('Post-Training. Total-Reward =', rewart)
    else:
        model = TD3.load(save_as)
        rewart = main_test(env, model, total_timesteps)
        print('Testing is Finished. Total-Reward =', rewart)
    
    return model
   
    

#=============================================================================================
# Testing
#=============================================================================================
def main_test(env, model, max_ts):
    obs = env.reset()
    rewart, done, ts = 0, False, 0
    while not done and ts<max_ts:
        ts+=1
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        #print(obs, reward, done, info)
        rewart+=reward
    return rewart



#=============================================================================================
# Args __main__ Caller
#=============================================================================================

    
def main_check(env):
    check_env(env)