
import UEMEC.rl as rl
from UEMEC.core import UeMEC, PARAMS
import UEMEC.basic as kp
import torch as tt
import random
import matplotlib.pyplot as plt 
max_episode_steps_train=2000
n_IOT, n_UAV, n_BSV = 5, 4, 2
fixed_move=10
layers = [128,64,64,64,64,64,64,64,64,64]
relu_act = False
if __name__=='__main__':
    print('Starting at:[{}]\n\n'.format( kp.now()))


    env = UeMEC('cpu', PARAMS(n_BSV, n_UAV, n_IOT) , cap=100_000, meed=None, seed=None, logging="", fixed_move=fixed_move, frozen=False, discrete_action=True)
    #env.reset()
    #fig=env.render('Initial-Train', True, True, True)
    #fig.savefig('train.png')


    pie = rl.DQN(env.nS, layers, env.nA, tt.optim.Adam, tt.nn.MSELoss, lr=0.000025, double=True, tuf=20, device='cpu', dtype=tt.float32, relu_act=relu_act )


    lx = []
    ex = []
    eps=1.0
    epochs = 10_000
    report_at = 100
    learn_times = 2
    learn_batch = 256
    learn_min_memory = learn_batch*10
    # 0 , 2999 ---> 1.0, 0.0   d1 = 2999, d2= 1.0
    # l = e 
    rfc = kp.RCONV((0, epochs), (1.0, 0.0))
    for e in range(epochs):
        done = env.start()
        #_ = env.render("", True, True, True)
        rr = []
        eps = rfc.in2map(e)
        ex.append(eps)
        ts=0
        while not done and ts<max_episode_steps_train:
            ts+=1
            a =  (random.randint(0, env.nA-1) ) if random.random()<eps else (pie.predict(env.state()))
            
            #print('Action:', a)
            #aa.append(a)

            env.act(a)
            done = env.next()
            reward = env.R.item()
            rr.append(reward)
            #if reward!=0:
            #    print('REward:', reward)
        #_ = env.render("", True, True, True)
        if env.memory.count() > learn_min_memory:
            for _ in range(learn_times):
                loss = pie.learn(env.memory, learn_batch, lr=0.0, dis=1)
                lx.append(loss)

                
        if e%report_at==0:
            print('epoch:[{}], Eps:[{}], Rew:[{}], Loss:[{}]'.format(e, eps, sum(rr),  (lx[-1] if lx else None)))


    fig, ax = plt.subplots(2, 1, figsize=(12,6))
    ax[0].plot(lx, label='loss')
    ax[1].plot(ex, label='epsilon')
    fig.savefig('train_result.png')

        #plt.figure()
        #plt.hist(aa, color='black')
        #plt.hist(rr, color='red')
        #plt.show()


    pie.save_external('dqn.pie')

    print('\n\nEnding at:[{}], saved @ dqn.pie'.format( kp.now()))