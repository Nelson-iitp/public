
import UEMEC.rl as rl
from UEMEC.core import UeMEC, PARAMS
import UEMEC.printer as kp
import torch as tt
import random
import matplotlib.pyplot as plt 


if __name__=='__main__':
    from train import n_IOT, n_UAV, n_BSV, fixed_move, layers, relu_act, max_episode_steps_train
    max_episode_steps_test = int(max_episode_steps_train* 0.5)

    # test
    env = UeMEC('cpu', PARAMS(n_BSV, n_UAV, n_IOT) , cap=0, meed=None, seed=None, logging="", fixed_move=fixed_move, frozen=True)
    pie = rl.DQN(env.nS, layers, env.nA, tt.optim.Adam, tt.nn.MSELoss, lr=0.00025, double=False, tuf=0, device='cpu', dtype=tt.float32, relu_act=relu_act )
    pie.load_external('dqn.pie')
    done = env.start()
    fig=env.render('Initial-Test', True, True, True)
    fig.savefig('test_initial.png')
    aa, rr = [], []
    ts=0
    lastrew=0
    while not done and ts<max_episode_steps_test:
        ts+=1
        a = (pie.predict(env.state())) # random.randint(1, env.nA-1) #
        aa.append(a)
        msg = env.act(a)
        done = env.next()
        reward = env.R.item()
        rr.append(reward+lastrew)
        lastrew = rr[-1]
        print('TS:[{}], A:[{}/{}], R:[{}]'.format(ts, a, msg, reward))

    fig=env.render('Final-Test', True, True, True)
    fig.savefig('test_final.png')
    print('\ntotal-reward:', sum(rr))

    histo = tt.zeros(env.nA)
    for a in aa:
        histo[a] += 1

    fig, ax = plt.subplots(2, 1, figsize=(12,6))
    ax[0].set_xticks([ i for i in range(env.nA) ])
    #ax[0].hist(aa, bins=env.nA, range=(0, env.nA+1), color='black', label='action-dist')
    ax[0].bar(tt.arange(0, env.nA, 1), histo, color='tab:blue', label='actions')
    ax[1].plot(rr, color='green', label='reward')
    plt.legend()
    fig.savefig('test_result.png')
    #plt.show()
