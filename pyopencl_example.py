#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import numpy as np
import pyopencl as cl
import time

def prepare_openCL_multiplication( w_rgb, w_rgbxy_values, w_rgbxy_indices ):
    ## rgbxy indices and values correspond.
    assert w_rgbxy_values.shape == w_rgbxy_indices.shape
    
    ## Our kernel function expects 32-bit values:
    w_rgb = w_rgb.astype(np.float32)
    w_rgbxy_values = w_rgbxy_values.astype(np.float32)
    w_rgbxy_indices = w_rgbxy_indices.astype(np.int32)
    
    npix = w_rgbxy_values.shape[0]
    
    padding = 16
    if padding is not None:
        npix_padded = ((npix-1)//padding+1)*padding
        print( "npix:", npix )
        print( "npix padded to %s:" % padding, npix_padded )
        print( "npix (mod) %s:" % padding, npix % padding )
        print( "npix padded (mod) %s:" % padding, npix_padded % padding )
        if npix != npix_padded:
            ## Add a few extra rows. Make sure to keep the dtype unchanged.
            w_rgbxy_values  = np.append( w_rgbxy_values,  np.zeros((npix_padded-npix,w_rgbxy_values.shape[1]),dtype=w_rgbxy_values.dtype), axis = 0 )
            w_rgbxy_indices = np.append( w_rgbxy_indices, np.zeros((npix_padded-npix,w_rgbxy_indices.shape[1]),dtype=w_rgbxy_indices.dtype), axis = 0 )
            # w_rgbxy_values  = np.append( w_rgbxy_values,  np.tile( w_rgbxy_values[-1:], (npix_padded-npix,1) ), axis = 0 )
            # w_rgbxy_indices = np.append( w_rgbxy_indices, np.tile( w_rgbxy_indices[-1:], (npix_padded-npix,1) ), axis = 0 )

    device = 'gpu'
    if device == 'ask':
        ## Ask the user:
        ctx = cl.create_some_context()
    else:
        ## Choose CPU or GPU automatically.
        platform = cl.get_platforms()
        if device == 'gpu':
            my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        elif device == 'cpu':
            my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.CPU)
        else:
            raise RuntimeError( "Unknown device: %s" % device )
        print( my_gpu_devices )
        ctx = cl.Context(devices=my_gpu_devices)

    queue = cl.CommandQueue(ctx)
    
    mf = cl.mem_flags
    w_rgbxy_values_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w_rgbxy_values)
    w_rgbxy_indices_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w_rgbxy_indices)
    
    NO_COPY = True
    if NO_COPY:
        w_rgb_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=w_rgb)
    else:
        w_rgb_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w_rgb)
    
    output_shape=(w_rgbxy_values.shape[0], w_rgb.shape[1])
    final_matrix = np.empty(output_shape).astype(np.float32)
    
    if NO_COPY:
        dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=final_matrix )
    else:
        dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, final_matrix.nbytes )
    
    prg = cl.Program(ctx, """
        __kernel void multiplymatrices(const unsigned int size, __global const float * w_rgb, __global const float * w_rgbxy_values, __global const int * w_rgbxy_indices, __global float * res) {

        int i = get_global_id(0); 
        int j = get_global_id(1);

        res[i * size + j] = 0;

        for (int k = 0; k < 6; k++)
        {
            res[i * size + j] += w_rgbxy_values[i * 6 + k] * w_rgb[ w_rgbxy_indices[i * 6 + k] * size + j];
        }

        }
        """).build()
    
    reps = 5
    
    all_times = []
    def actually_multiply( new_rgb_data ):
        nonlocal w_rgb_buf
        
        w_rgb[:] = new_rgb_data
        
        t0 = time.time()
        ## If we were really running this interactively, we would update w_rgb
        ## and keep w_rgbxy the same.
        if not NO_COPY:
            w_rgb_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w_rgb)
        ## Run the code.
        ## Automatic
        localsize = None
        print( 'global size:', output_shape )
        if output_shape[0] % 4 == 0: localsize = (4,w_rgb.shape[1])
        # localsize = (2,1)
        print( 'local size:', localsize )
        event = prg.multiplymatrices(queue, output_shape, localsize, np.int32(output_shape[1]), w_rgb_buf, w_rgbxy_values_buf, w_rgbxy_indices_buf, dest_buf )
        ## Copy the result back.
        if NO_COPY:
            event.wait()
        else:
            cl.enqueue_copy(queue, final_matrix, dest_buf)
        t1= time.time()
        delta_t=t1-t0
        all_times.append( delta_t )
        
        # print( final_matrix[:10,:10] )
        
        # print( np.average( np.asarray( all_times ) ) )
        print( "Latest time:", delta_t )
        
        return final_matrix[:npix]
    
    def get_times():
        return np.asarray( all_times )
    
    return actually_multiply, get_times

def openCL_multiplication( w_rgb, w_rgbxy_values, w_rgbxy_indices ):
    mult, get_times = prepare_openCL_multiplication( w_rgb, w_rgbxy_values, w_rgbxy_indices )
    
    for i in range(5):
        final_matrix = mult( w_rgb )
    
    print( final_matrix[:10,:10] )
    
    return final_matrix, get_times()

if __name__=="__main__":
    
    npix = 6*1000*1000
    nmiddle = 3000
    nlayers = 6
    
    np.random.seed(0)
    w_rgbxy_values=np.random.random(npix*6).reshape((npix,6))*1.0
    w_rgb=np.random.random(nmiddle*nlayers).reshape((nmiddle,nlayers))*1.0
    w_rgbxy_indices=(np.random.random((npix,6))*nmiddle).round().astype(np.int32)
    final_matrix, times = openCL_multiplication( w_rgb, w_rgbxy_values, w_rgbxy_indices )
    
    print( 'OpenCL Multiplication times:' )
    print( times )
    print( 'min:', times.min() )
    print( 'max:', times.max() )
    print( 'average:', np.average( times ) )
