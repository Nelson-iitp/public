import numpy as np
import matplotlib.pyplot as plt
import pygame as pg
import os
import threading
from .printer import strA
#from rlgw.model import WORLD, ENV
pg.init() # load pygame modules

#~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~#
# Game Module for Grid-World
# Implements graphics and most of pygame related subs 
#~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~#
        
class GAME:

    def __init__(self, env):
        self.env = env
        self.XR, self.YR = self.env.params.XR, self.env.params.YR

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
                start_pos=self.ccv (self.env.params.XR[0], 0),  #(0, self.height/2), 
                end_pos=self.ccv(self.env.params.XR[1], 0), 
                width=self.axis_line_width)

        pg.draw.line(self.screen, 
                color=self.lcY,
                start_pos=self.ccv(0, self.env.params.YR[0]), 
                end_pos=self.ccv(0, self.env.params.YR[1]), 
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
        pg.display.set_caption("UeMEC")
        
        #self.show_lines()              # create grid-lines ( one time call )
        #self.show_grid()               # create grid-cells ( one time call )
        
        pg.display.update()
        return

    def make_update(self):
        self.update_world()
        self.update_iot()
        self.update_uav()
        self.update_sel()       
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

        #self.screen.blit(self.posi_image, (self.env.POSi[1]*self.size_ratio+self.dy, self.env.POSi[0]*self.size_ratio+self.dx))
        #self.screen.blit(self.posf_image, (self.env.POSF[1]*self.size_ratio+self.dy, self.env.POSF[0]*self.size_ratio+self.dx))


    def main(self):
        """ main game loop - run in a seprate thread/process and read the hlast dict for scores """
        # perpare
        done = self.env.start()
        steps, total_rew = 0, 0
        self.make_screen() # will call pg.display.update()
        self.sel_iot=0
        sel_uav=0
        sel_bsv=0

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
                            if done:
                                done = self.env.start()
                                steps,total_rew = 0,0
                                self.make_screen()
                            else:
                                a=-1
                                if event.key == pg.K_i:
                                    self.sel_iot = (self.sel_iot+1)%self.env.params.n_IOT
                                    pg.display.set_caption('iot:'+ str(self.sel_iot))
                                elif event.key == pg.K_u:
                                    self.sel_uav = (self.sel_uav+1)%self.env.params.n_UAV
                                    pg.display.set_caption('uav:'+ str(self.sel_uav))
                                elif event.key == pg.K_b:
                                    self.sel_bsv = (self.sel_bsv+1)%self.env.params.n_BSV
                                    pg.display.set_caption('bsv:'+ str(self.sel_bsv))
                                elif event.key == pg.K_p:
                                    self.env.render('x', True, True, True)
                                    print(strA(self.env.iot))



                                elif event.key == pg.K_KP0:
                                    a = 1+self.env.params.n_IOT*0+ self.sel_iot
                                elif event.key == pg.K_KP1:
                                    a = 1+self.env.params.n_IOT*1 + self.sel_iot

                                elif event.key == pg.K_DOWN:
                                    a = 1+self.env.params.n_IOT*2 + 0*self.env.params.n_UAV + self.sel_uav
                                elif event.key == pg.K_RIGHT:
                                    a = 1+self.env.params.n_IOT*2 + 1*self.env.params.n_UAV + self.sel_uav
                                elif event.key == pg.K_UP:
                                    a = 1+self.env.params.n_IOT*2 + 2*self.env.params.n_UAV + self.sel_uav
                                elif event.key == pg.K_LEFT:
                                    a = 1+self.env.params.n_IOT*2 + 3*self.env.params.n_UAV + self.sel_uav


                                elif event.key == pg.K_KP5:
                                    a = 1+self.env.params.n_IOT*2 +self.env.params.n_UAV*4 + 0*self.env.params.n_BSV + self.sel_bsv
                                elif event.key == pg.K_KP6:
                                    a = 1+self.env.params.n_IOT*2 +self.env.params.n_UAV*4 + 1*self.env.params.n_BSV + self.sel_bsv
                                elif event.key == pg.K_KP8:
                                    a = 1+self.env.params.n_IOT*2 +self.env.params.n_UAV*4 + 2*self.env.params.n_BSV + self.sel_bsv
                                elif event.key == pg.K_KP4:
                                    a = 1+self.env.params.n_IOT*2 +self.env.params.n_UAV*4 + 3*self.env.params.n_BSV + self.sel_bsv
                                else:
                                    a=0
                                    
                                if a>=0 and a < self.env.nA:
                                    #print('action', a)
                                #if a>=0: # only on valid actions
                                    msg = self.env.act(a)
                                    pg.display.set_caption(str(a) + '::' + msg)
                                    done = self.env.step()
                                    reward = self.env.R.item()
                                    total_rew+=reward
                                    steps+=1
                                #else:
                                #    print('invalid action')
                                    #self.update_hlast(total_rew, reward, done, steps) # update scores
                                    #if self.print_scores:
                                    #    if done:
                                    #        print(self.score_str)
            if going:
                self.make_update()   # ~ Render
            
            # ~ wait thread
            # pg.time.wait(1)
        
        # assert(going==False) #<-- test
        pg.display.quit()      # display quit here <-----------
        # pg.quit()            # unload pygame modules
        return




def play(env):
    """ Sample Game play in pygame window """
    game = GAME(env)
    
    th = threading.Thread(target=game.main)
    th.start()
    th.join()
    print('Done!')
    return