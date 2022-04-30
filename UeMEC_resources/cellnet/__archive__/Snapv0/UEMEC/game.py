#import numpy as np
#import matplotlib.pyplot as plt
#import torch as tt
import pygame as pg
import os
import threading
from numpy.random import default_rng
pg.init() # load pygame modules

#~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~#
# Game Module for Grid-World
# Implements graphics and most of pygame related subs 
#~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~#
        
class GAME:

    def __init__(self, env):
        self.env = env
        self.XR, self.YR = self.env.params.XR, self.env.params.YR
        self.n_IOT, self.n_BSV, self.n_UAV = self.env.params.n_IOT, self.env.params.n_BSV, self.env.params.n_UAV 

        # -for- pygame
        # size
        
        self.size_ratio = 0.35 # ratio of each cell size
        self.width = (self.XR[1]-self.XR[0])*self.size_ratio   # Window width
        self.height = (self.YR[1]-self.YR[0])*self.size_ratio  # Window height

        #print('size:', self.width, self.height, self.XR, self.YR )
        
        # images
        self_art_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)) , 'art')
        self.bsv_img = lambda : pg.image.load(os.path.join(self_art_dir, 'bsv.jpg'))
        self.uav_img = lambda : pg.image.load(os.path.join(self_art_dir, 'uav.jpg'))
        self.iot_img = lambda : pg.image.load( os.path.join(self_art_dir, 'iot.jpg'))
        self.img_wid, self.img_hig = 16, 16
        
        
        self.font_family = 'consolas'
        self.font_size = 12

        self.bgcol=(255,255,255)

        self.lcX=(0,0,0) # x-axis
        self.lcY=(0,0,0) # y-axis
        self.axis_line_width = 1

        self.concol = (0,0,0) # connection 
        self.con_line_width = 1

        self.iot_col, self.uav_col, self.bsv_col = (0,255,0), (0,0,255), (255,0,0)        
        self.sel_iot = 0
        self.sel_bsv = 0
        self.sel_uav = 0

        return

    def show_axis(self):
        pg.draw.line(self.screen, 
                color=self.lcX,
                start_pos=self.ccv (self.XR[0], 0),  #(0, self.height/2), 
                end_pos=self.ccv(self.XR[1], 0), 
                width=self.axis_line_width)

        pg.draw.line(self.screen, 
                color=self.lcY,
                start_pos=self.ccv(0, self.YR[0]), 
                end_pos=self.ccv(0, self.YR[1]), 
                width=self.axis_line_width)

    def make_screen(self):
        """ Create game screen {OTC} """
        #self.reset_hlast()
        # define font 
        # SysFont(name, size, bold=False, italic=False)
        self.font = pg.font.SysFont(self.font_family, min(self.font_size, 20 )) # after pygame is initialized
        
        # create a window or screen (pygame.display)
        self.screen = pg.display.set_mode((self.width ,  self.height))  # screen display object

        self.screen.fill(self.bgcol)   # fill back ground color
        #pg.display.set_caption("UeMEC")
        rot = self.env.ROT.item()
        cost= self.env.COST.item()
        pg.display.set_caption('Rew:[{}], RoT:[{}], Cost:[{}]'.format(self.env.R.item(), rot, cost))
        
        #self.show_lines()              # create grid-lines ( one time call )
        #self.show_grid()               # create grid-cells ( one time call )
        
        pg.display.update()
        return

    def make_update(self):
        self.update_world()
        self.update_iot()
        self.update_uav()
        #self.update_sel()       
        self.show_axis() 
        pg.display.update()
        return
        
    def ccv(self, x, y):
        xx, yy = float(x), float(y)
        xx = (x + 1000) * self.size_ratio
        yy = (y + 1000) * self.size_ratio
        return int(xx), int(yy)

    def update_world(self):
        self.screen.fill(self.bgcol)   # fill back ground color
        for envdev,imgdev in [(self.env.iot, self.iot_img), (self.env.uav, self.uav_img), (self.env.bsv, self.bsv_img)]:
            for dev in envdev: #<--- for each IOT
                ii = dev.loc
                x, y =self.ccv(ii[0].item(), ii[1].item())
                self.screen.blit(imgdev(),(x-self.img_wid,y-self.img_hig) )  # (ii[0], ii[1]) 

    def update_sel(self):
        iot = self.env.iot[self.sel_iot]
        loc = iot.loc
        x, y = self.ccv(loc[0].item(), loc[1].item())
        pg.draw.ellipse(self.screen, self.iot_col, 
            pg.Rect(x-self.img_wid/2,y-self.img_hig/2, self.img_wid, self.img_hig ))

        uav = self.env.uav[self.sel_uav]
        loc = uav.loc
        x, y = self.ccv(loc[0].item(), loc[1].item())
        pg.draw.ellipse(self.screen, self.uav_col, 
            pg.Rect(x-self.img_wid/2,y-self.img_hig/2, self.img_wid, self.img_hig ))

        bsv = self.env.bsv[self.sel_bsv]
        loc = bsv.loc
        x, y = self.ccv(loc[0].item(), loc[1].item())
        pg.draw.ellipse(self.screen, self.bsv_col, 
            pg.Rect(x-self.img_wid/2,y-self.img_hig/2, self.img_wid, self.img_hig ))

    def update_uav(self):
        for dev in self.env.uav:
            loc = dev.loc
            x, y = self.ccv(loc[0].item(), loc[1].item())
            
            # draw transmission radius
            dtx = int(dev.dtx[0].item()*self.size_ratio)
            pg.draw.ellipse(self.screen, self.uav_col, 
                pg.Rect(x-dtx,y-dtx, dtx*2, dtx*2),1)

            # draw connections
            parent = int(dev.parent[0].item())
            if parent>=0:
                pi = self.env.bsv[parent].loc
                px, py = pi[0].item(), pi[1].item()

                pg.draw.line(self.screen, 
                        color=self.concol,
                        start_pos=self.ccv(px, py), 
                        end_pos=(x,y), 
                        width=self.con_line_width)

    def update_iot(self):
        for dev in self.env.iot:
            loc = dev.loc
            x, y = self.ccv(loc[0].item(), loc[1].item())
            
            # draw transmission radius
            dtx = int(dev.dtx[0].item()*self.size_ratio)
            pg.draw.ellipse(self.screen, self.iot_col, 
                pg.Rect(x-dtx,y-dtx, dtx*2, dtx*2),1)

            # draw connections
            parent = int(dev.parent[0].item())
            if parent>=0:
                pi = self.env.uav[parent].loc
                px, py = pi[0].item(), pi[1].item()

                pg.draw.line(self.screen, 
                        color=self.concol,
                        start_pos=self.ccv(px, py), 
                        end_pos=(x,y), 
                        width=self.con_line_width)
                                
            ioff = int(dev.off[0].item())
            if ioff==0:
                assert(parent>=0)
                pg.draw.rect(self.screen, self.uav_col, 
                    pg.Rect(x-self.img_wid,y-self.img_hig, self.img_wid*2, self.img_hig*2),3)
            elif ioff==1:
                assert(parent>=0)
                pg.draw.rect(self.screen, self.bsv_col, 
                    pg.Rect(x-self.img_wid,y-self.img_hig, self.img_wid*2, self.img_hig*2),3)
            else:
                pg.draw.rect(self.screen, self.iot_col, 
                    pg.Rect(x-self.img_wid,y-self.img_hig, self.img_wid*2, self.img_hig*2),3)

    def main_kb(self):
        """ main game loop - run in a seprate thread/process and read the hlast dict for scores """
        # perpare
        done = self.env.start()
        rng = default_rng(None)
        self.make_screen() # will call pg.display.update()
        #actv = self.env.spaces['ACT'].zeros()
        #absv = actv[0: 2*self.n_BSV] #self.env.spaces['ACT_BSV'].zeros()
        #auav = actv[2*self.n_BSV: 2*self.n_BSV + 2*self.n_UAV]
        #aiot = actv[2*self.n_BSV + 2*self.n_UAV : 2*self.n_BSV + 2*self.n_UAV + self.n_IOT] #self.env.spaces['ACT_IOT'].zeros()
        #default_move = 0.05
        going = True
        while going:
            # ~ Check Events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    going = False
                else:
                    if event.type == pg.KEYDOWN:
                        # any keypress to reset game in 'done' state
                        if event.key == pg.K_ESCAPE:
                            going =  False
                        else:
                            if event.key == pg.K_x:
                                print("reset")
                                done = self.env.start()
                                self.make_screen()
                            else:
                                if not done:
                                    actv = self.env.spaces['A'].sample(rng)
                                    pg.display.set_caption(str(actv))
                                    print('actv', actv)
                                    self.env.act(actv)
                                    done = self.env.next()
                                    reward = self.env.R.item()
                                    rot = self.env.ROT.item()
                                    cost= self.env.COST.item()
                                    pg.display.set_caption('Rew:[{}], RoT:[{}], Cost:[{}]'.format(reward, rot, cost))
                                else:
                                    pg.display.set_caption('Done!, press x to restart')
                                

            if going:
                self.make_update()   # ~ Render
            
            # ~ wait thread
            # pg.time.wait(1)
        
        # assert(going==False) #<-- test
        pg.display.quit()      # display quit here <-----------
        # pg.quit()            # unload pygame modules
        return

    def main(self, agent):
        """ main game loop - run in a seprate thread/process and read the hlast dict for scores """
        # perpare
        print('sarting with agent: [{}]'.format(agent))
        done = self.env.start()
        #rng = default_rng(None)
        self.make_screen() # will call pg.display.update()
        #actv = self.env.spaces['ACT'].zeros()
        #absv = actv[0: 2*self.n_BSV] #self.env.spaces['ACT_BSV'].zeros()
        #auav = actv[2*self.n_BSV: 2*self.n_BSV + 2*self.n_UAV]
        #aiot = actv[2*self.n_BSV + 2*self.n_UAV : 2*self.n_BSV + 2*self.n_UAV + self.n_IOT] #self.env.spaces['ACT_IOT'].zeros()
        #default_move = 0.05
        going = True
        while going:
            # ~ Check Events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    going = False
                else:
                    
                    if not done:
                        actv = agent.predict(self.env.state())[0]
                        #pg.display.set_caption(str(actv))
                        #print('actv', type(actv), actv)
                        self.env.act(actv)
                        
                        done = self.env.next()
                        ts = self.env.STEP.item()
                        reward = self.env.R.item()
                        rot = self.env.ROT.item()
                        cost= self.env.COST.item()
                        pg.display.set_caption('Rew:[{}], RoT:[{}], Cost:[{}] Step:[{}]'.format(reward, rot, cost, ts))
                    else:
                        pg.display.set_caption('Done!')
                                

            if going:
                self.make_update()   # ~ Render
            
            # ~ wait thread
            # pg.time.wait(1)
        
        # assert(going==False) #<-- test
        pg.display.quit()      # display quit here <-----------
        # pg.quit()            # unload pygame modules
        return

def play(env, agent=None):
    """ Sample Game play in pygame window """
    game = GAME(env)
    
    th = (threading.Thread(target=game.main_kb) if agent is None else threading.Thread(target=game.main, args=(agent,)))
    th.start()
    th.join()
    print('Done!')
    return