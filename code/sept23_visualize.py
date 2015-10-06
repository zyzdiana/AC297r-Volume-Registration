import matplotlib.pyplot as plt
import cPickle as pickle
import scipy.special
import ghalton
import numpy as np
import time
import os
import sys
sys.path.insert(1,'/Users/zyzdiana/GitHub/AC297r-Volume-Registration/code')
from utils import res_to_rad,ax_to_w,rep_to_angle
from visualize import plot_cost_func, plot_cost_func_t,plot_volume

#########################################################
# Plot Cost Functions
#########################################################
axes_dict = pickle.load(open('/Users/zyzdiana/Dropbox/THESIS/for_cluster/axes_dict.p','rb'))
rot_axes = ['xz', 'y', 'yz', 'xy', 'x', 'z']
ax_to_idx = {}
for rot_ax in rot_axes:
    ax_to_idx[rot_ax] = rot_axes.index(rot_ax)

def plot_results(cost_dict, res, interpolation,step_size = 0.1):
    ranges = ['0_5_to_2_5','3_0_to_5_0']
    for idx, rot_ax in enumerate( ['xz', 'y', 'yz', 'xy', 'x', 'z']):
        for rot_range in ranges:
            for loop in xrange(6):
                plt.figure(figsize = [40,4])
                for i in xrange(1,6):
                    rep = i + loop * 6
                    rot_angle = rep_to_angle(rep,rot_range)
                    deg = rot_angle[0]
                    if interpolation == 'cubic':
                        thetas = np.arange(deg-3,deg+3,step_size)
                    else:
                        thetas = np.arange(deg-3,deg+3,0.01)
                    rot = '%s_%s' % (str(rot_angle[0]),rot_ax)
                    plt.subplot(1,6,i+1)
                    plot_cost_func(thetas,cost_dict[idx][rot_angle],res,interpolation,rot)
                    plt.xlabel('Angles (degrees)')
                    plt.ylabel('SSD Cost Function')
                plt.show() 

def plot_result_trans(cost_dict, res, interpolation):
    ranges = ['0_5_to_2_5','3_0_to_5_0']
    trans = np.arange(-1,1,0.01)
    for idx, rot_ax in enumerate( ['xz', 'y', 'yz', 'xy', 'x', 'z']):
        for rot_range in ranges:
            for loop in xrange(1,6):
                plt.figure(figsize = [40,4])
                for i in xrange(6):
                    rep = i + loop * 6
                    rot_angle = rep_to_angle(rep,rot_range)
                    step = rot_angle[1]
                    rot = '%s_%s' % (str(step)+'mm',axes_dict[rot_ax])
                    plt.subplot(1,6,i+1)
                    plot_cost_func_t(trans,cost_dict[rot_ax][rot_angle],res,rot)
                plt.suptitle('%s interpolation' % interpolation)
                plt.show() 

#########################################################
# Plot Rotations Results
#########################################################

# Function that does scatter plot of results
def scatter_plot(cost_dict, res):
    colors = ['red','blue','green']
    ranges = ['0_5_to_2_5','3_0_to_5_0']
    count = 0
    for idx, rot_ax in enumerate(['x', 'y', 'z']):
        count = 1
        for rot_range in ranges:
            for loop in xrange(6):
                for i in xrange(1,6):
                    rep = i + loop * 6
                    if(count == 1): 
                        label = rot_ax
                        count = 0
                    else: label = None
                    rot_angle = rep_to_angle(rep,rot_range)
                    deg = rot_angle[0]
                    thetas = np.arange(-deg-2,deg+2,0.01)
                    cost = cost_dict[idx][rot_angle]
                    y = abs(thetas[np.argmin(cost,axis=0)])
                    plt.scatter(deg, y, s = 80, c = colors[idx],alpha = 0.6,marker='x',label = label)
    plt.plot([0,6],[0,6],c='black')
    plt.xlim([0,6])
    plt.ylim([0,6])
    plt.legend(loc=4)
    plt.xlabel('True Rotations (Degrees)')
    plt.ylabel('Search Results from Registration')
    plt.title('Search Results for %s Resolution' %res)

#########################################################
# Plot Linear Results
#########################################################
# Function that does scatter plot of results
def scatter_plot_linear(cost_dict, res, axes):
    colors = ['red','blue','green','orange','brown','purple']
    ranges = ['0_5_to_2_5','3_0_to_5_0']
    count = 0
    for idx, rot_ax in enumerate(axes):
        count = 1
        for rot_range in ranges:
            for loop in xrange(6):
                for i in xrange(1,6):
                    rep = i + loop * 6
                    if(count == 1): 
                        label = rot_ax
                        count = 0
                    else: label = None
                    rot_angle = rep_to_angle(rep,rot_range)
                    deg = rot_angle[0]
                    thetas = np.arange(deg-3,deg+3,0.01)
                    cost = cost_dict[ax_to_idx[rot_ax]][rot_angle]
                    y = abs(thetas[np.argmin(cost,axis=0)])
                    plt.scatter(deg, y, lw=0,s = 30, c = colors[ax_to_idx[rot_ax]],alpha = 0.6,marker='o',label = label)
    plt.plot([0,6],[0,6],c='black')
    plt.xlim([0,6])
    plt.ylim([0,6])
    plt.legend(loc=4)
    plt.xlabel('True Rotations (Degrees)')
    plt.ylabel('Search Results from Registration')
    plt.title('Search Results for %s Resolution' %res)
    
def scatter_plot_all_linear(axes, cost_dict_10, cost_dict_8, cost_dict_6_4,figsize = [12,4]):
    plt.figure(figsize = figsize)
    plt.subplot(1,3,1)
    scatter_plot_linear(cost_dict_10,'10mm',axes)
    plt.subplot(1,3,2)
    scatter_plot_linear(cost_dict_8,'8mm',axes)
    plt.subplot(1,3,3)
    scatter_plot_linear(cost_dict_6_4,'6.4mm',axes)
    plt.suptitle('Linear Interpolation')
    plt.show() 


def error_in_time(cost_dict, res, step_size = 0.01):
    colors = ['red','blue','green','orange','brown','purple']
    ranges = ['0_5_to_2_5','3_0_to_5_0']
    errors = []
    for idx, rot_ax in enumerate(axes_dict.keys()):
        plt.figure(figsize = [15,4])
        for ii, rot_range in enumerate(ranges):
            plt.subplot(1,2,ii+1)
            for loop in xrange(6):
                for i in xrange(1,6):
                    rep = i + loop * 6
                    rot_angle = rep_to_angle(rep,rot_range)
                    deg = rot_angle[0]
                    thetas = np.arange(deg-3,deg+3,step_size)
                    cost = cost_dict[ax_to_idx[rot_ax]][rot_angle]
                    y = abs(thetas[np.argmin(cost,axis=0)])
                    plt.scatter(rep, abs(y-deg), lw=0,s = 50, c = colors[ax_to_idx[rot_ax]],alpha = 0.6,marker='o')
            plt.title('%s, %s, rot_%s, trans_%s' % (res,rot_range,rot_ax,axes_dict[rot_ax]))
            plt.xlabel('Repetitions')
            plt.ylabel('Absolute error')
        plt.show()

#########################################################
# Plot Tricubic Results
#########################################################
def scatter_plot_cubic(cost_dict, res, axes, step_size = 0.1):
    colors = ['red','blue','green','orange','brown','purple']
    ranges = ['0_5_to_2_5','3_0_to_5_0']
    count = 0
    for idx, rot_ax in enumerate(axes):
        count = 1
        for rot_range in ranges:
            #print rot_range
            for loop in xrange(6):
                for i in xrange(1,6):
                    rep = i + loop * 6
                    if(count == 1): 
                        label = rot_ax
                        count = 0
                    else: label = None
                    rot_angle = rep_to_angle(rep,rot_range)
                    deg = rot_angle[0]
                    thetas = np.arange(deg-3,deg+3,step_size)
                    cost = cost_dict[ax_to_idx[rot_ax]][rot_angle]
                    y = abs(thetas[np.argmin(cost,axis=0)])
                    plt.scatter(deg, y, lw=0,s = 30, c = colors[ax_to_idx[rot_ax]],alpha = 0.6,marker='o',label = label)
    plt.plot([0,6],[0,6],c='black')
    plt.xlim([0,6])
    plt.ylim([0,6])
    plt.legend(loc=4)
    plt.xlabel('True Rotations (Degrees)')
    plt.ylabel('Search Results from Registration')
    plt.title('Search Results for %s Resolution' %res)
    
def scatter_plot_all_cubic(axes, cost_dict_10, cost_dict_8, cost_dict_6_4,figsize = [12,4],step_size = 0.1):
    plt.figure(figsize = figsize)
    plt.subplot(1,3,1)
    scatter_plot_linear(cost_dict_10,'10mm',axes)
    plt.subplot(1,3,2)
    scatter_plot_linear(cost_dict_8,'8mm',axes)
    plt.subplot(1,3,3)
    scatter_plot_linear(cost_dict_6_4,'6.4mm',axes)
    plt.suptitle('Tricubic Interpolation')
    plt.show() 


#########################################################
# Plot Translation Results
#########################################################

# Function that does scatter plot of results
def scatter_plot_trans(cost_dict, res, axes):
    colors = ['red','blue','green','orange','brown','purple']
    ranges = ['0_5_to_2_5','3_0_to_5_0']
    count = 0
    trans = np.arange(-1,1,0.01)
    for idx, rot_ax in enumerate(axes):
        count = 1
        for rot_range in ranges:
            for loop in xrange(1,6):
                for i in xrange(6):
                    rep = i + loop * 6
                    if(count == 1): 
                        label = rot_ax+'_'+axes_dict[rot_ax]
                        count = 0
                    else: label = None
                    rot_angle = rep_to_angle(rep,rot_range)
                    step = rot_angle[1]
                    cost = cost_dict[rot_ax][rot_angle]
                    y = abs(trans[np.argmin(cost,axis=0)])*float(res[:-2])
                    plt.scatter(step, y, s = 80, c = colors[ax_to_idx[rot_ax]],alpha = 0.3,marker='o',lw=0,label = label)
    plt.plot([0,6],[0,6],c='black')
    plt.xlim([0,6])
    plt.ylim([0,6])
    plt.legend(loc='best')
    plt.xlabel('True Translations (mm)')
    plt.ylabel('Search Results from Registration')
    plt.title('Search Results for %s Resolution' %res)
    
def scatter_plot_all_trans(axes, cost_dict_10, cost_dict_8, cost_dict_6_4,figsize = [12,4], interp = 'Linear'):
    plt.figure(figsize = figsize)
    plt.subplot(1,3,1)
    scatter_plot_trans(cost_dict_10,'10mm',axes)
    plt.subplot(1,3,2)
    scatter_plot_trans(cost_dict_8,'8mm',axes)
    plt.subplot(1,3,3)
    scatter_plot_trans(cost_dict_6_4,'6.4mm',axes)
    plt.suptitle('%s Interpolation'% interp)
    plt.show() 

def error_in_time_trans(cost_dict, res):
    colors = ['red','blue','green','orange','brown','purple']
    ranges = ['0_5_to_2_5','3_0_to_5_0']
    trans = np.arange(-1,1,0.01)
    for idx, rot_ax in enumerate(axes_dict.keys()):
        plt.figure(figsize = [15,4])
        for ii, rot_range in enumerate(ranges):
            plt.subplot(1,2,ii+1)
            for loop in xrange(1,6):
                for i in xrange(6):
                    rep = i + loop * 6
                    rot_angle = rep_to_angle(rep,rot_range)
                    step = rot_angle[1]
                    cost = cost_dict[rot_ax][rot_angle]
                    y = abs(trans[np.argmin(cost,axis=0)])*float(res[:-2])
                    plt.scatter(rep, abs(y-step), lw=0,s = 50, c = colors[ax_to_idx[rot_ax]],alpha = 0.6,marker='o')
            plt.title('%s, %s, rot_%s, trans_%s' % (res,rot_range,rot_ax,axes_dict[rot_ax]))
            plt.xlabel('Repetitions')
        plt.ylabel('abs(search results - true translation)')
        plt.show()