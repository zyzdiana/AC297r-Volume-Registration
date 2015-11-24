from bokeh.plotting import *
from bokeh.models import ColumnDataSource
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

def plot_results_offax(cost_dict1,cost_dict2, res, axes, interpolation = 'Cubic',step_size = 0.1):
    ranges = ['0_5_to_2_5','3_0_to_5_0']
    for idx, rot_ax in enumerate(axes):
        for rot_range in ranges:
            for loop in xrange(6):
                plt.figure(figsize = [40,4])
                for i in xrange(1,6):
                    rep = i + loop * 6
                    rot_angle = rep_to_angle(rep,rot_range)
                    deg = rot_angle[0]
                    thetas = np.arange(-3,3,step_size)
                    rot = '%s_%s' % (str(rot_angle[0]),rot_ax)
                    plt.subplot(1,6,i+1)
                    plot_cost_func(thetas,cost_dict1[idx][rot_angle],res,interpolation,rot,alpha = 0.5,lw=2)
                    plot_cost_func(thetas,cost_dict2[idx][rot_angle],res,interpolation,rot,alpha = 0.5,lw=2)
                    plt.xlabel('Angles (degrees)')
                    plt.ylabel('SSD Cost Function')
                plt.show() 

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


def error_in_time(cost_dict, res, step_size, axes, ax_to_idx_dict=ax_to_idx):
    colors = ['red','blue','green','orange','brown','purple']
    ranges = ['0_5_to_2_5','3_0_to_5_0']
    errors = []
    for idx, rot_ax in enumerate(axes):
        plt.figure(figsize = [15,4])
        for ii, rot_range in enumerate(ranges):
            plt.subplot(1,2,ii+1)
            for loop in xrange(6):
                for i in xrange(1,6):
                    rep = i + loop * 6
                    rot_angle = rep_to_angle(rep,rot_range)
                    deg = rot_angle[0]
                    thetas = np.arange(deg-1,deg+1,step_size)
                    cost = cost_dict[ax_to_idx_dict[rot_ax]][rot_angle]
                    y = abs(thetas[np.argmin(cost,axis=0)])
                    plt.scatter(rep, deg-y, lw=0,s = 50, c = colors[ax_to_idx[rot_ax]],alpha = 0.6,marker='o')
            plt.hlines([0.05,-0.05],-5,40,'black')
            plt.xlim([-5,40])
            plt.title('%s, %s, rot_%s, trans_%s' % (res,rot_range,rot_ax,axes_dict[rot_ax]))
            plt.xlabel('Repetitions')
            plt.ylabel('Absolute error')
    plt.show()

def error_plot(p, cost_dict, col, res, step_size, axes, ax_to_idx_dict=ax_to_idx,theta_range = 1,xlim = [0,7]):
    colors = ['red','blue','green','orange','brown','purple']
    ranges = ['0_5_to_2_5','3_0_to_5_0']
    errors = []
    for idx, rot_ax in enumerate(axes):
        for ii, rot_range in enumerate(ranges):
            for loop in xrange(6):
                for i in xrange(1,6):
                    rep = i + loop * 6
                    rot_angle = rep_to_angle(rep,rot_range)
                    deg = rot_angle[0]
                    thetas = np.arange(deg-theta_range,deg+theta_range,step_size)
                    cost = cost_dict[ax_to_idx_dict[rot_ax]][rot_angle]
                    y = abs(thetas[np.argmin(cost,axis=0)])
                    errors.append(deg-y)
                    p.circle(col,deg-y, radius=0.03, fill_color=colors[ax_to_idx[rot_ax]], fill_alpha=0.2, line_color=None,legend=rot_ax)
    return errors
#########################################################
# Plot Tricubic Results
#########################################################
def preprocess(cost_dict1,cost_dict2):
    ranges = ['0_5_to_2_5','3_0_to_5_0']
    cost_dict = []
    for i in xrange(len(cost_dict1)):
        dict_ = {}
        dict_.update(cost_dict1[i])
        dict_.update(cost_dict2[i])
        cost_dict.append(dict_)
    return cost_dict

def load_pickle(res,rotation,path = '/Users/zyzdiana/Dropbox/THESIS/Pickled_Results/',filtered=False, offax = False):
    if offax:
        filename1 = path + 'oct13_offax1_tricubic_%s_%s_rotation_0.p' % (res, rotation)
        cost_dict_1 = pickle.load(open(filename1,'rb'))
        filename2 = path + 'oct13_offax1_tricubic_%s_%s_rotation_1.p' % (res, rotation)
        cost_dict_2 = pickle.load(open(filename2,'rb'))

        filename3 = path + 'oct13_offax2_tricubic_%s_%s_rotation_0.p' % (res, rotation)
        cost_dict_3 = pickle.load(open(filename3,'rb'))
        filename4 = path + 'oct13_offax2_tricubic_%s_%s_rotation_1.p' % (res, rotation)
        cost_dict_4 = pickle.load(open(filename4,'rb'))
        return preprocess(cost_dict_1,cost_dict_2),preprocess(cost_dict_3,cost_dict_4)
    elif filtered:
        filename1 = path + 'oct13_filtered_tricubic_%s_%s_rotation_0.p' % (res, rotation)
        cost_dict_1 = pickle.load(open(filename1,'rb'))
        filename2 = path + 'oct13_filtered_tricubic_%s_%s_rotation_1.p' % (res, rotation)
        cost_dict_2 = pickle.load(open(filename2,'rb'))
        return preprocess(cost_dict_1,cost_dict_2)
    else:       
        filename1 = path + 'oct13_tricubic_%s_%s_rotation_0.p' % (res, rotation)
        cost_dict_1 = pickle.load(open(filename1,'rb'))
        filename2 = path + 'oct13_tricubic_%s_%s_rotation_1.p' % (res, rotation)
        cost_dict_2 = pickle.load(open(filename2,'rb'))
        return preprocess(cost_dict_1,cost_dict_2)

def load_pickle_filtered(res,rotation,path = '/Users/zyzdiana/Dropbox/THESIS/Pickled_Results/'):
    filename1 = path + 'oct13_filtered_tricubic_%s_%s_rotation_0.p' % (res, rotation)
    cost_dict_1 = pickle.load(open(filename1,'rb'))
    filename2 = path + 'oct13_filtered_tricubic_%s_%s_rotation_1.p' % (res, rotation)
    cost_dict_2 = pickle.load(open(filename2,'rb'))
    return preprocess(cost_dict_1,cost_dict_2)

def split_plot(cost_dict_1, cost_dict_2,res,ax_to_idx_dict):
    # Split the axes to see the results more clearly
    plt.figure(figsize = [18,12])
    for idx, ax in enumerate(['x','y','z']):
        plt.subplot(2,3,idx+1)
        axes = [ax]
        scatter_plot_cubic(cost_dict_1, res, axes,0.01,ax_to_idx_dict,alpha = 0.4,s=50)
    plt.show()
    plt.figure(figsize = [18,12])
    for idx, ax in enumerate(['xy','yz','xz']):
        plt.subplot(2,3,idx+1)
        axes = [ax]
        scatter_plot_cubic(cost_dict_2, res, axes,0.01,ax_to_idx_dict,alpha = 0.4,s=50)
    plt.show() 

def split_plot_all(cost_dict_1, cost_dict_2,cost_dict_1_shifted, cost_dict_2_shifted,cost_dict_1_filtered, cost_dict_2_filtered,res,ax_to_idx_dict):
    # Split the axes to see the results more clearly
    plt.figure(figsize = [18,12])
    for idx, ax in enumerate(['x','y','z']):
        plt.subplot(2,3,idx+1)
        axes = [ax]
        scatter_plot_cubic(cost_dict_1, res, axes,0.01,ax_to_idx_dict,alpha = 0.4,s=50)
        scatter_plot_cubic(cost_dict_1_shifted, res, axes,0.01,ax_to_idx_dict,alpha = 0.4,s = 100,marker='v',
            label_arg = ' shifted center',colors = ['orange','brown','purple','red','blue','green'])
        scatter_plot_cubic(cost_dict_1_filtered, res, axes,0.01,ax_to_idx_dict,alpha = 0.4,s = 100,marker='*',
            label_arg = ' filtered',colors = ['red','purple','orange','brown','green','blue'])
    plt.show()
    plt.figure(figsize = [18,12])
    for idx, ax in enumerate(['xy','yz','xz']):
        plt.subplot(2,3,idx+1)
        axes = [ax]
        scatter_plot_cubic(cost_dict_2, res, axes,0.01,ax_to_idx_dict,alpha = 0.4,s=50)
        scatter_plot_cubic(cost_dict_2_shifted, res, axes,0.01,ax_to_idx_dict,alpha = 0.4,s = 100, marker='v',
            label_arg = ' shifted center',colors = ['orange','brown','purple','red','blue','green'])
        scatter_plot_cubic(cost_dict_2_filtered, res, axes,0.01,ax_to_idx_dict,alpha = 0.4,s = 100,marker='*',
            label_arg = ' filtered',colors = ['red','purple','orange','brown','green','blue'])
    plt.show()   

def scatter_plot_cubic(p, cost_dict, res, axes, step_size=0.01, ax_to_idx_dict=ax_to_idx,alpha = 0.3, s = 0.05, marker = 'o',label_arg = '',colors = ['red','blue','green','orange','brown','purple']):
    #colors = ['red','blue','green','orange','brown','purple']
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
                        label = rot_ax+label_arg
                        count = 0
                    else: label = None
                    rot_angle = rep_to_angle(rep,rot_range)
                    deg = rot_angle[0]
                    thetas = np.arange(deg-1,deg+1,step_size)
                    cost = cost_dict[ax_to_idx_dict[rot_ax]][rot_angle]
                    y = abs(thetas[np.argmin(cost,axis=0)])
                    p.circle(deg,y, radius=s, fill_color=colors[ax_to_idx[rot_ax]], fill_alpha=alpha, line_color=None,legend = label)

def scatter_plot_cubic1(p, cost_dict, res, axes, step_size=0.01, ax_to_idx_dict=ax_to_idx,alpha = 0.3, s = 0.05, marker = 'o',label_arg = ''):
    colors = ['red','blue','green','orange','brown','purple']
    ranges = ['0_5_to_2_5','3_0_to_5_0']
    count = 0
    #for idx, rot_ax in enumerate(axes):
    rot_ax = 'xy'
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
                thetas = np.arange(deg-1,deg+1,step_size)
                cost = cost_dict[rot_angle]
                y = abs(thetas[np.argmin(cost,axis=0)])
                p.circle(deg,y, radius=s, fill_color=colors[ax_to_idx[rot_ax]], fill_alpha=alpha, line_color=None,legend = label)
    
def scatter_plot_all_cubic(axes, cost_dict_10, cost_dict_8, cost_dict_6_4,figsize = [12,4],step_size = 0.01):
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
                    plt.scatter(step, y, s = 60, c = colors[ax_to_idx[rot_ax]],alpha = 0.3,marker='o',lw=0,label = label)
    plt.plot([0,6],[0,6],c='black')
    plt.xlim([0,6])
    plt.ylim([0,6])
    plt.legend(loc='best')
    plt.xlabel('True Translations (mm)')
    plt.ylabel('Search Results from Registration')
    plt.title('Search Results for %s Resolution' %res)

def scatter_plot_trans1(cost_dict, res, step_range=2):
    ranges = ['0_5_to_2_5','3_0_to_5_0']
    count = 0
    #trans = np.arange(-1,1,0.01)
    rot_ax = 'yz'
    for rot_range in ranges:
        count = 1
        for loop in xrange(1,6):
            for i in xrange(6):
                rep = i + loop * 6
                if(count == 1): 
                    label = rot_ax+'_'+axes_dict[rot_ax]
                    count = 0
                else: label = None
                rot_angle = rep_to_angle(rep,rot_range)
                cost = cost_dict[rot_angle]
                step = float(rot_angle[1])*5./float('.'.join(res[:-2].split('_')))
                trans = np.arange(-np.floor(step)-step_range,-np.floor(step)+step_range,0.01)
                y = abs(trans[np.argmin(cost,axis=0)])*float('.'.join(res[:-2].split('_')))
                plt.scatter(float(rot_angle[1])*5., y, s = 60, c = 'green',alpha = 0.3,marker='o',lw=0,label = label)
    plt.plot([0,30],[0,30],c='black')
    plt.xlim([0,30])
    plt.ylim([0,30])
    plt.legend(loc='best')
    plt.xlabel('True Translations (mm)')
    plt.ylabel('Search Results from Registration')
    plt.title('Search Results for %s Resolution' %res)

def error_plot_trans1(cost_dict, col, res,step_range=2):
    colors = ['red','blue','green','orange','brown','purple']
    ranges = ['0_5_to_2_5','3_0_to_5_0']
    rot_ax = 'yz'
    for ii, rot_range in enumerate(ranges):
        #plt.subplot(1,2,ii+1)
        for loop in xrange(1,6):
            for i in xrange(6):
                rep = i + loop * 6
                rot_angle = rep_to_angle(rep,rot_range)
                cost = cost_dict[rot_angle]
                step = float(rot_angle[1])*5./float('.'.join(res[:-2].split('_')))
                trans = np.arange(-np.floor(step)-step_range,-np.floor(step)+step_range,0.01)
                y = abs(trans[np.argmin(cost,axis=0)])#*float('.'.join(res[:-2].split('_')))
                plt.scatter(col,y-step, lw=0,s = 30, c = colors[ax_to_idx[rot_ax]],alpha = 0.1,marker='o')
            #plt.hlines([0.05,-0.05],-5,40,'black')
            plt.xlim([0,7])
            plt.ylim([-1.0,3.0])
            #plt.title('%s, %s, rot_%s, trans_%s' % (res,rot_range,rot_ax,axes_dict[rot_ax]))
            plt.title('Error Plot for Translations',fontsize = 18)
            #plt.xlabel('Data Set')
            plt.ylabel('Errors (mm)',fontsize = 15)    
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

def error_plot_trans(cost_dict, col, res):
    colors = ['red','blue','green','orange','brown','purple']
    ranges = ['0_5_to_2_5','3_0_to_5_0']
    trans = np.arange(-1,1,0.01)
    for idx, rot_ax in enumerate(axes_dict.keys()):
        #plt.figure(figsize = [15,4])
        for ii, rot_range in enumerate(ranges):
            #plt.subplot(1,2,ii+1)
            for loop in xrange(1,6):
                for i in xrange(6):
                    rep = i + loop * 6
                    rot_angle = rep_to_angle(rep,rot_range)
                    step = rot_angle[1]
                    cost = cost_dict[rot_ax][rot_angle]
                    y = abs(trans[np.argmin(cost,axis=0)])*float(res[:-2])
                    plt.scatter(col,y-step, lw=0,s = 30, c = colors[ax_to_idx[rot_ax]],alpha = 0.1,marker='o')
            #plt.hlines([0.05,-0.05],-5,40,'black')
            plt.xlim([0,7])
            plt.ylim([-1.0,1.0])
            #plt.title('%s, %s, rot_%s, trans_%s' % (res,rot_range,rot_ax,axes_dict[rot_ax]))
            plt.title('Error Plot for Translations',fontsize = 18)
            #plt.xlabel('Data Set')
            plt.ylabel('Errors (mm)',fontsize = 15)
    #plt.show()