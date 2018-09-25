from __future__ import print_function, division

import numpy as np
import time
import scipy
import json
import Additive_mixing_layers_extraction
from scipy.spatial import ConvexHull, Delaunay
import scipy.sparse
from numpy import *
import PIL.Image as Image


def RGBXY_extraction(filepath, palette_rgb, origin_image, mask=None, prefix=None, SAVE=True):
    ### data shape is row*col*3.
    
    M=len(palette_rgb)
    if mask is None: ### normal use
        img_copy=origin_image
        img=img_copy.copy() ### do not modify img_copy
        X,Y=np.mgrid[0:img.shape[0], 0:img.shape[1]]
    else: ### for masked foreground and background
        X,Y=np.where(mask==1)
        img_copy=origin_image[X,Y].reshape((1,-1,3))
        img=img_copy.copy() ### do not modify img_copy
    
    
    XY=np.dstack((X*1.0/origin_image.shape[0],Y*1.0/origin_image.shape[1]))

    data=np.dstack((img, XY))
    # print data.shape
    
    start=time.time()
    data_hull=ConvexHull(data.reshape((-1,5)))
    # print len(data_hull.vertices)
    
    ### RGB weights using star triangulation.
    print ("using star triangulation now!")
    mixing_weights_1=Additive_mixing_layers_extraction.Get_ASAP_weights_using_Tan_2016_triangulation_and_then_barycentric_coordinates(img.reshape((-1,3))[data_hull.vertices].reshape((-1,1,3)), palette_rgb, "None", order=0)
   
    #### RGBXY weights
    mixing_weights_2=Additive_mixing_layers_extraction.recover_ASAP_weights_using_scipy_delaunay(data_hull.points[data_hull.vertices], data_hull.points, option=3)
    
    end=time.time()
    print ("RGBXY method extract mixing weights using time: ", end-start)

    mixing_weights=mixing_weights_2.dot(mixing_weights_1.reshape((-1,M)))
    mixing_weights=mixing_weights.reshape((img.shape[0],img.shape[1],-1)).clip(0,1)

    temp=(mixing_weights.reshape((img.shape[0],img.shape[1],-1,1))*palette_rgb.reshape((1,1,-1,3))).sum(axis=2)
    
    if SAVE:
        recover_name=filepath[:-4]+"-palette_size-"+str(len(palette_rgb))+"-recovered_image-using_5D_hull.png"
        Image.fromarray((temp*255).round().clip(0,255).astype(np.uint8)).save(recover_name)

    img_diff=temp*255-img_copy*255
    diff=square(img_diff.reshape((-1,3))).sum(axis=-1)
    # print 'max diff: ', sqrt(diff).max()
    # print 'median diff', median(sqrt(diff))
    rmse=sqrt(diff.sum()/diff.shape[0])
    print ('Reconstruction RMSE: ', sqrt(diff.sum()/diff.shape[0]))


    if SAVE:
        if mask is None: ### normal image
            mixing_weights_filename=filepath[:-4]+"-palette_size-"+str(len(palette_rgb))+"-linear_mixing-weights-using_5D_hull.js"
            with open(mixing_weights_filename,'wb') as myfile:
                json.dump({'weights': mixing_weights.tolist()}, myfile)
            for i in range(mixing_weights.shape[-1]):
                mixing_weights_map_filename=filepath[:-4]+"-palette_size-"+str(len(palette_rgb))+"-linear_mixing-weights_map-using_5D_hull-%02d.png" % i
                Image.fromarray((mixing_weights[:,:,i]*255).round().clip(0,255).astype(uint8)).save(mixing_weights_map_filename)
        else: ### for foreground and background image

            ### map back to original shape to show weights map as image format.
            weights_map=np.zeros((origin_image.shape[0],origin_image.shape[1], M))
            weights_map[np.where(mask==1)]=mixing_weights.reshape((-1,M))

            mixing_weights_filename=filepath[:-4]+"-"+prefix+"-palette_size-"+str(len(palette_rgb))+"-linear_mixing-weights-using_5D_hull.js"
            with open(mixing_weights_filename,'wb') as myfile:
                json.dump({'weights': weights_map.tolist()}, myfile)

            for i in range(M):
                mixing_weights_map_filename=filepath[:-4]+"-"+prefix+"-palette_size-"+str(len(palette_rgb))+"-linear_mixing-weights_map-using_5D_hull-%02d.png" % i
                Image.fromarray((weights_map[:,:,i]*255).round().clip(0,255).astype(uint8)).save(mixing_weights_map_filename)



    return mixing_weights









