from UEMEC.core import UeMEC, PARAMS

n_IOT, n_UAV, n_BSV = 7, 4, 2

env = UeMEC('cpu', PARAMS(n_BSV, n_UAV, n_IOT) , cap=10_000, meed=None, seed=None, logging="", fixed_move=20, frozen=False  )
env.reset()
#_=env.render('Initial', True, True, True)


import UEMEC.game as game

game.play(env)

env.memory.render_all()