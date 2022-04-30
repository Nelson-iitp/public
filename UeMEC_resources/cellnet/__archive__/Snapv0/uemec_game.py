from UEMEC.core import UeMEC, PARAMS

n_IOT, n_UAV, n_BSV = 20, 10, 4

env = UeMEC(device='cpu', params=PARAMS(n_BSV, n_UAV, n_IOT) , cap=10_000, meed=None, seed=None, logging="m.txt", frozen=False)
env.reset()
#_=env.render('Initial', True, True, True)


import UEMEC.game as game

game.play(env)

#env.memory.render_all()
