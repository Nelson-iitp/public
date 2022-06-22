import os
from math import inf
import numpy as np
import gym.spaces
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import torch as tt
from .common import observation_key, action_key, reward_key, done_key, step_key, default_spaces
from .common import MEM, MLP, MLP2, load_model, save_model, clone_model, REMAP
import datetime
now = datetime.datetime.now


class MAVPIE:
    def __init__(self, observation_space, action_space) -> None:
        self.observation_space=observation_space
        self.action_space=action_space
        self.predict = self.predict_random
        self.policy_theta=None

    def predict_random(self, state):
        return self.action_space.sample()

    @tt.no_grad()
    def predict_policy(self, state):
        #return np.clip(self.policy_theta(state).numpy().astype(self.action_space.dtype), self.action_space.low, self.action_space.high)
        return self.policy_theta(tt.tensor(state, dtype=tt.float32)).numpy().astype(self.action_space.dtype)

    def load(self, path):
        if path:
            self.policy_theta = load_model(path)
            self.predict = self.predict_policy
        else:
            self.predict = self.predict_random

    def save_pie(self, path):
        if not (self.policy_theta is None):
            save_model(path, self.policy_theta)

    def new_pie(self, layers, actF, actL):
        self.policy_theta = MLP(
            in_dim=self.observation_space.shape[0],
            layer_dims=layers,
            out_dim=self.action_space.shape[0],
            actF=actF, actL=actL
        )
        self.predict = self.predict_policy

    def set_pie(self, theta):
        self.policy_theta = theta
        self.predict = self.predict_policy

    def has_pie(self):
        return not (self.policy_theta is None)

class MAVBOT:
    state_dtype = np.float32
    action_dtype = np.float64
    battery_scan_time = 1000
    initial_wait_time = (battery_scan_time/1000) + 0.1
    battery_min_val = 100

    def __init__(self, robot, time_step_multiplier):
        self.robot = robot
        self.name = robot.getName()
        self.timestep = int(  robot.getBasicTimeStep() * time_step_multiplier )
        self.build_spaces()
        self.pie = MAVPIE(self.observation_space, self.action_space)

    def initialize(self):
        #self.keyboard = self.robot.getKeyboard()
        # get devices ----------------------------------
        #print('[@] - Initialize Robot Devices')
        self.imu = self.robot.getDevice('imu')
        self.gps = self.robot.getDevice('gps')
        self.gyro = self.robot.getDevice('gyro')

        self.led_dLAND = self.robot.getDevice('led_DSEN_LAND')
        self.dLAND = self.robot.getDevice('DSEN_LAND')

        # get motors 
        self.front_left_motor = self.robot.getDevice('flp')
        self.front_right_motor = self.robot.getDevice('frp')
        self.rear_left_motor = self.robot.getDevice('rlp')
        self.rear_right_motor = self.robot.getDevice('rrp')
        self.motors = (self.front_left_motor, self.front_right_motor, self.rear_left_motor, self.rear_right_motor)

        # enable devices ----------------------------------
        self.imu.enable(self.timestep)
        self.gps.enable(self.timestep)
        self.gyro.enable(self.timestep)
        self.dLAND.enable(self.timestep)
        # enable battery ----------------------------------
        # <--- note: battery is not a 'device' its just a 'field' with 3 values - Current Level, Max level, Charging Rate
        self.robot.batterySensorEnable(self.battery_scan_time) #<---- argument is the battery sampling period in ms
        # initialize motor position and velocity (max 576)
        for m in self.motors:
            m.setPosition(inf)
            m.setVelocity(0.0)

    def build_spaces(self):
        self.observation_space = gym.spaces.Box(shape=(11,), low=-np.inf, high=np.inf, dtype=self.state_dtype)
        self.S = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        self.roll, self.pitch, self.yaw =       self.S[0:1], self.S[1:2], self.S[2:3]
        self.gX, self.gY, self.gZ =             self.S[3:4], self.S[4:5], self.S[5:6]
        self.rollA, self.pitchA, self.yawA =    self.S[6:7], self.S[7:8], self.S[8:9]
        self.dis2land, self.battery_level =     self.S[9:10], self.S[10:11]


        #self.min_v = np.zeros(len(self.motors), dtype=self.action_dtype)
        #self.max_v = np.zeros(len(self.motors), dtype=self.action_dtype) + 576
        self.action_space = gym.spaces.Box(shape=(4,), low=-1, high=1,  dtype=self.action_dtype)
        self.action_mapper = REMAP(Input_Range=(-1, 1), Mapped_Range=(0, 576))

    def reset(self):
        #print('Reset Called <-----')
        self.robot.simulationResetPhysics()
        self.robot.simulationReset()
        self.robot.step(self.timestep)
        #print('Trying initializing devices')
        self.initialize()
        self.robot.step(self.timestep)
        #print('Initial wait <------------')

        while self.robot.step(self.timestep) != -1:
            if (self.robot.getTime() > self.initial_wait_time):
                break   
        # can be used to record initializ information
        #print('Initial Sensor reading <------------')
        self.initial_gps_location = -1*np.array(self.gps.getValues())
        
        self._dis2land = float(self.dis2land)

        #print(f'Initial BATTERY: [{self.initial_battery_level}]')
        self.read()
        #print(self.S)
        return False

    def read(self):
        # reads the state from sensor data
        # either use a build in state or return a new state from readings
        self.gps_read =  self.gps.getValues()
        self.S[0:3] =                           self.imu.getRollPitchYaw()
        self.S[3:6] =                           self.initial_gps_location + self.gps_read
        self.S[6:9] =                           self.gyro.getValues()
        self.S[9:10] =                          self.dLAND.getValue()
        self.S[10:11] =                         self.robot.batterySensorGetValue()
        return self.S, self._done()
    def _done(self):
        # checks if the sensors read an invalid state 
        # - which causes feed back loop termination
        if self.battery_level<self.battery_min_val:
            self._stop_motor()
            #print('Stopping :: Battery End!!')
            return True #<---- very low battery = 10 cycle left
        if self.gZ<=0.0657594:
            if self.roll < -0.7 or self.roll > 0.7 or self.pitch < -0.7 or self.pitch > 0.7:
                self._stop_motor()
                #print('Stopping :: Crashed!!')
                return True #<---- drone has crashed
        return False

    def step(self, action):
        # write actuator inputs
        #print(self.roll, self.rollA, self.pitch, self.pitchA, self.yaw, self.yawA)
        self._step_motor(self.action_mapper.in2map(action))
        
        return (self.robot.step(self.timestep) == -1)

    def reward(self):
       # _dis2land = float(self.dis2land)
        r = (self.dis2land - self._dis2land)*0.001 + abs(self.gps_read[2] - 1.0)  - (abs(self.rollA) + abs(self.pitchA))
        
        return float(r)

    def _stop_motor(self):
        self._step_motor([0,0,0,0])
    def _step_motor(self, velocities):
        for m, v in zip(self.motors, velocities):
            m.setVelocity(v)

    def episode(self):
        if not self.pie.has_pie():
            print('Policy missing! using random actions.')
            

        #zero_action = self.action_space.sample()*0
        timeout = self.reset() # this indicates the bot to get ready
        done = False
        ts = 0
        terminate = (timeout or done)
        # feedback loop: step simulation until receiving an exit event
        #print('Starting feedback loop')
        while not (terminate):
            ts+=1
            # read sensors outputs #<--- implement as observation_space
            state, done = self.read() #<--- this updates bot.state with updated data / returns a state
            #print(f'{ts=}, {done=}, {state=}')
            if not done: # process behavior
                action = self.pie.predict(state)
                timeout = self.step(action) # write actuators inputs
            else:
                action = None
                timeout = False
                
            terminate = (timeout or done)
            #print(f'->>{action}:{timeout=}:{done=}:{terminate=}')
        return ts
        
def train_ddpg(bot):
    """ Trains using deep deterministic policy gradients 
    
        bot :   instance of MAVBOT
    """

    dtype=tt.float32
    device='cpu'

    observation_space = bot.observation_space
    action_space = bot.action_space
    
    memory_capacity = 50000
    memory_seed = None

    total_steps = 100_000
    pie_lr = 0.003
    val_lr = 0.001

    # > training parameters here
    training_frequency = 1000 # steps - trains after every (training_frequency) steps
    
    learn_times = 3
    batch_size = 512
    start_steps = batch_size*5 #<--- explored sperated (independent of epochs)
    
    train_value_iters = 80
    polyak_val = 0.1
    polyak_pie = 0.1
    # discounting 
    gamma = 0.9999
    # for noisy exploration
    action_noise_mean = 0 
    action_noise_sdev = 0.1

    

    #-----------------------------------------------------------------------------------------
    # [1] setup policy and value networks
    #-----------------------------------------------------------------------------------------
    pie =  MLP(
        in_dim=observation_space.shape[0],
        layer_dims=[256,256,256,256],
        out_dim=action_space.shape[0],
        actF=tt.nn.ReLU, actL=tt.nn.Tanh).to(device, dtype)
    pie_ = clone_model(pie, detach=True)

    val = MLP2(
        in_dim_s=observation_space.shape[0],
        in_dim_a=action_space.shape[0],
        layer_dims=[256,256,256,256],
        out_dim=1, actF=tt.nn.ReLU, actL=None).to(device, dtype)
    val_ = clone_model(val, detach=True)

    pie_.eval()
    val_.eval()

    # setup optimizer
    pie_opt = tt.optim.Adam(pie.parameters(), lr = pie_lr, weight_decay=0.0)
    pie_lrs = tt.optim.lr_scheduler.LinearLR(pie_opt, start_factor=1.0, end_factor=0.5, total_iters=total_steps)

    val_opt = tt.optim.Adam(val.parameters(), lr = val_lr, weight_decay=0.0)
    val_lrs = tt.optim.lr_scheduler.LinearLR(val_opt,  start_factor=1.0, end_factor=0.5, total_iters=total_steps)
    val_loss = tt.nn.MSELoss()

    mem = MEM(observation_space, action_space, 
                capacity=memory_capacity, seed=memory_seed)



    zero_action = action_space.sample()*0
    epsilon = NormalActionNoise(
                mean= action_noise_mean * np.ones_like(zero_action), 
                sigma= action_noise_sdev * np.ones_like(zero_action))



    #-----------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------------
    # [2] training - Training happens in between simulation
    #-----------------------------------------------------------------------------------------

    @tt.no_grad()
    def take_action(observation):
        return pie(tt.tensor(observation, dtype=dtype, device=device)).numpy().astype(action_space.dtype) 

    @tt.no_grad()
    def take_noisy_action(observation):
        return np.clip(pie(tt.tensor(observation, dtype=dtype, device=device)).numpy().astype(action_space.dtype) + epsilon(), 
                    a_min=action_space.low, a_max=action_space.high)

    @tt.no_grad()
    def update_target(theta, theta_, polyak):
        for target_params, model_params in zip(theta_.parameters(), theta.parameters()):
            target_params*=(polyak)
            target_params+=((1-polyak)*model_params)


    #-----------------------------------------------------------------------------------------
    _start_time = now()
    terminate=True
    tr = None
    # starting loop
    for _ in range(start_steps):
        if terminate:
            # just reset it-------------------------------------------
            timeout = bot.reset() # this indicates the bot to get ready
            done = False
            ts = 0
            tr=0.0
            terminate = (timeout or done)
            # --------------------------------------------------------

        ts+=1 # stepping ... (running timestep)
        # observe a state and select action
        state, done = bot.read() #<--- this updates bot.state with updated data / returns a state
        #print(f'{ts=}, {done=}, {state=}')
        if not done: # process behavior
            action = action_space.sample()
            timeout = bot.step(action) # write actuators inputs
            reward = bot.reward()
            tr+=reward
        else:
            action = zero_action
            timeout, reward = False, 0.0
        terminate = (timeout or done)
        #print(f'->>{action}:{timeout=}:{done=}:{terminate=}')
        mem.snap(mask=not (terminate), observation=state, action=action , reward=reward, done=terminate, step=ts ) 

    print('Starting Steps Explored: {}'.format(mem.length()))


    # training loop
    step = 0
    episode = 0
    train_hist, train_count = [], 0
    
    while step<=total_steps:
        step+=1 # running epoch
        
        if terminate:
            # just reset it-------------------------------------------
            print(f'Last-Episodic-Reward:[{episode=}]::[{tr=}]')
            timeout = bot.reset() # this indicates the bot to get ready
            done = False
            ts = 0
            tr = 0.0
            terminate = (timeout or done)
            episode+=1
            # --------------------------------------------------------

        ts+=1 # stepping ... (running timestep)
        # observe a state and select action
        state, done = bot.read() #<--- this updates bot.state with updated data / returns a state
        #print(f'{ts=}, {done=}, {state=}')
        if not done: # process behavior
            action = take_noisy_action(state) 
            timeout = bot.step(action) # write actuators inputs
            reward = bot.reward()
            tr+=reward
        else:
            action = zero_action
            timeout, reward = False, 0.0
        terminate = (timeout or done)
        #print(f'->>{action}:{timeout=}:{done=}:{terminate=}')
        mem.snap(mask=not (terminate), observation=state, action=action , reward=reward, done=terminate, step=ts ) 

        if step%training_frequency==0: # .to(device, dtype)
            print(f'[Training] :: {step = } of {total_steps} :: {episode = } :: {train_count = }')
            # time for training 
            pie.train(True)
            val.train(True)
            # ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ -
            for _ in range(learn_times):
                #  batch_size, dtype, device, discrete_action, replace=False
                pick, samples = mem.sample_random_(size=batch_size, replace=False)
                batch = mem.readkeis(
                (samples,          samples+1,        samples,     samples,     samples+1,     ), #samples      ), 
                (observation_key,  observation_key,  action_key,  reward_key,  done_key,      ), #step_key     ), 
                ('cS',             'nS',             'A',         'R',         'D',           )) #'T'          ))
                # return pick, cS, nS, A, R, D, T
                
                cS = tt.tensor(batch['cS'], dtype=dtype, device=device)
                nS = tt.tensor(batch['nS'], dtype=dtype, device=device)
                A =  tt.tensor(batch['A'], dtype=(dtype), device=device)
                R =  tt.tensor(batch['R'], dtype=dtype, device=device).unsqueeze(-1)
                D =  tt.tensor(batch['D'], dtype=dtype, device=device).unsqueeze(-1)
                #T = tt.tensor(batch['T'], dtype=dtype, device=device)

                with tt.no_grad(): #<=====================================
                    pie_ns = pie_(nS)
                    target_val = val_(nS, pie_ns)
                    target = R + gamma * target_val * (1 - D)
                #<========================================================
                #print(f'{pie_ns.shape = }, {target_val.shape = }, {target.shape = }')
                
                for _ in range(train_value_iters):
                    val_opt.zero_grad()
                    pred = val(cS, A)
                    #print(f'{pred.shape = }, {target.shape = }')
                    qloss =  val_loss(pred, target)
                    qloss.backward()
                    val_opt.step()
                
            
                pie_opt.zero_grad()
                pred_actions = pie(cS)
                ploss = -(val(cS, pred_actions).mean())
                ploss.backward()
                pie_opt.step()

                update_target(val, val_, polyak_val)
                update_target(pie, pie_, polyak_pie)
                
                val_.eval()
                pie_.eval()

            # ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ -
            #loss_hist.append((ploss.item(), qloss.item()))
            train_hist.append( (pie_lrs.get_last_lr()[-1], val_lrs.get_last_lr()[-1])  )
            pie_lrs.step()
            val_lrs.step()
            train_count+=1

            pie.train(False)
            val.train(False)
        # training end
    # epochs end
    _end_time=now()
    print('Training Finished: [{}]'.format(_end_time - _start_time ))
    bot.robot.simulationSetMode(0)
    return train_count, train_hist[-5:]
        
            

        