from pyadlml.dataset.plot.util import heatmap, annotate_heatmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def forceAspect(ax,aspect):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
    
def mean_image(images, devices, figsize=(10,10)):
    """ plots the mean oimage of all images
    Parameters
    ----------
    images: 3d np array (n x win_size x dev_count)
        a 3d batch of images with window size and the feature/devices as last axis
    devices: list 
        a list of all device labels
    
    """
    title='Mean image'
    
    mean_image = images.mean(axis=0).T
    
    fig, ax = plt.subplots(figsize=figsize)
    h, w = mean_image.shape
    
    im = ax.imshow(mean_image, extent=[0,w,0,h], vmin=0, vmax=1)
    
    
    divider = make_axes_locatable(ax)      
    cax = divider.append_axes("right", size='5%', pad=0.1)
    cbar = cax.figure.colorbar(im, cax)
    #plt.colorbar(im, cax=cax)
    
    
    ax.set_xticks(np.arange(w) + 0.5)
    ax.set_yticks(np.arange(h) + 0.5)
    
    ax.set_xticklabels(np.arange(w)+1)
    ax.set_yticklabels(devices[::-1]) # reverse because of transpose of values
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    
    #forceAspect(ax, aspect=1.5)
    ax.set_title(title)
    fig.tight_layout()
    plt.show()



def mean_image_per_activity(X, y, devices, figsize=(14,18)):
    """ plots the mean oimage of all images
    Parameters
    ----------
    X: 3d np array (n x win_size x dev_count)
        a 3d batch of images with window size and the feature/devices as last axis
    y: 1d array (n,) of strings
        activity labels 
    devices: list 
        a list of all device labels
    """
    title='Mean image per Activity'
    num_plts_x = 3
    
    activities = np.unique(y)
    
    mi_lst = []
    for act in activities:
        # get the index where the activity matches and compute mean
        idxs = np.where(y == act)[0]    
        mi_lst.append(X[idxs].mean(axis=0).T)
    
    len_acts = len(activities)
    num_plts_y = int(np.ceil(len_acts/num_plts_x))
    
    
    # plotting
    fig, axs = plt.subplots(num_plts_y, num_plts_x, 
                             figsize=figsize)
    #plt.title('asdf', y=1.08)
    
    h, w = mi_lst[0].shape
    k = 0
    pcm_list = []
    for i in range(num_plts_y):
        for j in range(num_plts_x):
            if k >= len(activities):
                axs[i,j].axis('off')
                continue
                
            ax = axs[i,j]
            im = ax.imshow(mi_lst[k],  vmin=0, vmax=1)#, extent=[0,w,0,h],)
            #pcm = ax.pcolormesh(mi_lst[k], cmap='viridis')
            #pcm_list.append(pcm)
            
            #if k+1 == len(activities):
            ax.set_yticks(np.arange(h))
            if j == 0:                                
                ax.set_yticklabels(devices)
            else:
                ax.set_yticklabels([])
            ax.set_xticks(np.arange(w))
            ax.set_xticklabels(np.arange(w)+1)
            
            ax.set_title(activities[k])
            k += 1
    
    # make cosmetic changes
    plt.subplots_adjust(wspace=0.1, hspace=-.6)
    plt.suptitle(title,fontsize=20, y=0.8, va='top')
    plt.show()