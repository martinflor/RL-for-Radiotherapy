# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 09:36:01 2022

@author: Florian Martin


Plotting module for RL radiotherapy

"""

import imageio.v2 as imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw 
import os



import matplotlib 
from mpl_toolkits import mplot3d
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation

class Animation:
    
    def __init__(self):
        
        self.bar_idx = 0
        
    def bar3D(self, xsize, ysize, dz, TICK):
        
        xpos, ypos = np.mgrid[0:xsize, 0:ysize]
        xpos, ypos = xpos.ravel(), ypos.ravel()
        zpos = np.zeros(xsize*ysize)
        
        dx, dy = np.ones((xsize,ysize))*0.5, np.ones((ysize,xsize))*0.5
        
        matplotlib.rc('xtick', labelsize=35) 
        matplotlib.rc('ytick', labelsize=35) 
        fig = plt.figure(figsize=(50,50))
        ax = fig.add_subplot(projection='3d')
        cmap = cm.get_cmap('jet')
        max_height = np.max(dz)
        min_height = np.min(dz)
        rgba = [cmap((k-min_height)/max_height) for k in dz.ravel()]
        ax.bar3d(xpos, ypos, zpos, 0.5, 0.5, dz.ravel(), color=rgba)
        
        
        ax.set_zticks([0,1,2,3,4,5,6,7])
        ax.zaxis.set_major_locator(MaxNLocator(integer=True))
    
        ax.set_title(f"\n Instant t = {TICK}", fontsize=50)
        
        plt.savefig(f"animate/distribution/{self.bar_idx}.jpg")
        self.bar_idx += 1 
        plt.show()

        
    def smooth_bar3D(self, counts, counts_old, number, list_dz, xsize, ysize, TICK):
        """
        counts     : the current occurence
        counts_old : the old occurence
        number     : number of frame to smooth the transition between counts_old and counts
        all_dz     : list of dz used in 3dBarPlot of matplotlib
        """
        
        counts_itr = np.copy(counts_old)
        add_count = (counts-counts_old)/number
        for i in range(0, number):
            counts_itr += add_count
            list_dz.append(counts_itr)
            self.bar3D(xsize, ysize, list_dz[-1], TICK)
            
    def animate_3dBarPlot(self, nb, fps=30):
        # nb is the number of image saved in the directory "animate/distribution"
        # To save the image use the function bar3D in a loop
        # .jpg to save memory 
        
        frame = []
        for index in range(nb):
            img = imageio.imread(f"animate/distribution/{index}.jpg")
            frame.append(Image.fromarray(img))
    
        imageio.mimwrite(os.path.join('./animate/', f'3d_bar_plot_{fps}fps.gif'), frame, fps=60)

    def animate_3dBarPlot2(self, xsize, ysize, all_dz):
        """
        xsize  : size of the grid for the x-axis
        ysize  : size of the grid for the y-axis
        all_dz : list of dz used in 3dBarPlot of matplotlib
        """
        
        xpos, ypos = np.mgrid[0:xsize, 0:ysize]
        xpos, ypos = xpos.ravel(), ypos.ravel()
        zpos = np.zeros(xsize*ysize)
        
        dx, dy = np.ones((xsize,ysize))*0.5, np.ones((ysize,xsize))*0.5
        
        matplotlib.rc('xtick', labelsize=35) 
        matplotlib.rc('ytick', labelsize=35) 
        fig = plt.figure(figsize=(50,50))
        ax = fig.add_subplot(projection='3d')
        ax.zaxis.set_major_locator(MaxNLocator(integer=True))
        
        def update(frame):
            
            dz = all_dz[frame]
        
            ax.set_title(f"\n Instant t = {self.bar_idx}", fontsize=50)
            cmap = cm.get_cmap('jet')
            max_height = np.max(dz)
            min_height = np.min(dz)
            rgba = [cmap((k-min_height)/max_height) for k in dz.ravel()]
            ax.bar3d(xpos, ypos, zpos, 0.5, 0.5, dz.ravel(), color=rgba)
            
            return ax


        anim = FuncAnimation(fig, update, frames=len(all_dz), interval=10)
        anim.save('animate/3dbarPlot.gif')


"""
frame = []
nb=960
for index in range(nb):
    img = imageio.imread(f"animate/{index}.png")
    frame.append(Image.fromarray(img))
    
imageio.mimwrite(os.path.join('./animate/', 'env_example_60fps.gif'), frame, fps=60)
"""

from grid import Grid
from cell import HealthyCell, CancerCell, OARCell
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np


def count_cells(grid):
    return [[len(grid.cells[i][j]) for j in range(grid.ysize)] for i in range(grid.xsize)]

def counts_barPlot(grid, xsize, ysize):
    
    counts = np.zeros((xsize, ysize))
        
    for i in range(xsize):
        for j in range(ysize):
            counts[i][j] = grid.cells[i][j].size
            
    return counts.astype(float)

                
def patch_type_color(patch):
    if len(patch) == 0:
        return 102, 0, 204
    else:
        return patch[0].cell_color()

def grid_plot(grid, xsize, ysize, radius = None, imshow = True):
    plt.figure(figsize=(xsize,ysize))
    
    if imshow:
        plt.figure(figsize=(xsize,ysize))
        plt.imshow([[patch_type_color(grid.cells[i][j]) for j in range(grid.ysize)] for i in range(grid.xsize)])
        plt.show()
        
    else:
        ax = plt.axes()
        ax.set_facecolor("darkslateblue")
        plt.grid()
        plt.xticks(np.arange(xsize), [" "]*xsize)
        plt.yticks(np.arange(ysize), [" "]*ysize)
        
        x_healthy_cells = []
        y_healthy_cells  = []
        x_cancer_cells = []
        y_cancer_cells  = []
        
        for i in range(xsize):
                    for j in range(ysize):
                        if grid.cells[i][j].size > 0:
                            if len(grid.cells[i][j].healthy_cells) > 0:
                                x_healthy_cells.append(i)
                                y_healthy_cells.append(j)
                                
                            elif len(grid.cells[i][j].cancer_cells) > 0:
                                x_cancer_cells.append(i)
                                y_cancer_cells.append(j)
        
        plt.scatter(x_healthy_cells, y_healthy_cells, color="green", s=500)
        plt.scatter(x_cancer_cells, y_cancer_cells, color="red", s=500)
        
        if radius is not None:
            circle = plt.Circle(xy=(grid.center_x, grid.center_y), radius=radius, color="gold", alpha=0.75)
            ax.add_patch(circle)
        
        
        plt.show()
    
        #plt.savefig("figures/Init_grid.svg")
            
def irradiate(grid, dose):
    """Irradiate the tumour"""
    radius = grid.irradiate(dose)
    grid_plot(radius = radius, imshow=False)

xsize=50 
ysize=50 
hcells=1000
sources=100
grid = Grid(xsize, ysize, sources)

cancer_cell = CancerCell(random.randint(0, 3))
grid.cells[xsize//2, ysize//2].append(cancer_cell)

prob = hcells / (xsize * ysize)
for i in range(xsize):
            for j in range(ysize):
                if random.random() < prob:
                    new_cell = HealthyCell(random.randint(0, 4))
                    grid.cells[i, j].append(new_cell)
"""
for i in range(100):
    print(i)
    go(grid, 10)
    for _ in range(35):
        radius = irradiate(grid, 2)
        go(grid, 24)
"""
TICK = 0
anim = Animation()
index = 0
all_dz = []

counts_old = counts_barPlot(grid, xsize, ysize)
for i in range(100):
    print(i)
    grid.fill_source(130, 4500)
    grid.cycle_cells()
    grid.diffuse_glucose(0.2)
    grid.diffuse_oxygen(0.2)
    
    counts = counts_barPlot(grid, xsize, ysize)
    print((counts_old == counts).sum()/counts.size == 1)
    anim.smooth_bar3D(counts=counts, 
                      counts_old=counts_old, 
                      number=5, 
                      list_dz=all_dz,
                      xsize=xsize,
                      ysize=ysize,
                      TICK=TICK)
    counts_old = counts[:]
    
        
    TICK += 1
    if TICK % 24 == 0:
        grid.compute_center()
        grid_plot(grid, xsize=xsize, ysize=ysize, imshow=False)
    

#anim.animate_3dBarPlot(nb=100)



