#!/usr/bin/env python3

import asyncio
import websockets
import json
import numpy as np
import sys
import os
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _parent_dir)

import Additive_mixing_layers_extraction
Additive_mixing_layers_extraction.DEMO=True
import pyopencl_example
import RGBXY_method

from scipy.spatial import ConvexHull

async def layer_server( websocket, path ):
    the_image = None
    RGBXY_mixing_weights=None
    data_hull=None
    
    async for msg in websocket:
        print (msg)
        if msg == "load-image":
            ## Receive parameters from the websocket.
            width_and_height = await websocket.recv()
            data = await websocket.recv()
 
            ## Parse the parameters.
            # print (width_and_height)
            width = int( width_and_height.split()[0] )
            height = int( width_and_height.split()[1] )
            the_image_new = np.frombuffer( data, dtype = np.uint8 ).reshape(height, width, 4 ).copy()
            
            ## Skip load image if we already have this exact image.
            if the_image is not None and np.all(the_image == the_image_new):
                print( "Skipping duplicate load-image." )
                continue
            else:
                the_image = the_image_new
            
            # the_image = np.frombuffer( data, dtype = np.uint8 ).copy()
            print( "Received an image with", the_image.nbytes, "bytes." )
            
            # Compute RGBXY_mixing_weights.
            print( "Computing RGBXY mixing weights ..." )
            X,Y=np.mgrid[0:the_image.shape[0], 0:the_image.shape[1]]
            XY=np.dstack((X*1.0/the_image.shape[0],Y*1.0/the_image.shape[1]))
            RGBXY_data=np.dstack((the_image[:,:,:3]/255.0, XY))
            print( "\tConvexHull 5D..." )
            data_hull=ConvexHull(RGBXY_data.reshape((-1,5)))
            print( "\t...finished" )
            print( "\tComputing W_RGBXY..." )
            RGBXY_mixing_weights=Additive_mixing_layers_extraction.recover_ASAP_weights_using_scipy_delaunay(data_hull.points[data_hull.vertices], data_hull.points, option=3)
            print( "\t...finished" )
            print( "... finished." )

        
        elif msg == "palette":
            ## Receive parameters from the websocket.
            print (the_image.shape)
            palette = await websocket.recv()
            print (palette)

            ## Parse the parameters.
            palette = json.loads( palette )
            palette = np.asarray(palette)/255.0
            
            print (palette)
            ## Compute something.
            # hull=ConvexHull(palette) 
            # convex_hull_edges = hull.points[hull.simplices]

            num_layers = len( palette )
            print (num_layers)


            ### compute RGB_mixing_weights and use pyopencl version code to dot product with sparse RGBXY_mixing_weights
            img_data=(the_image[:,:,:3].reshape((-1,3))[data_hull.vertices]).reshape((-1,1,3))/255.0
            print (img_data.shape)
#### delaunay triangulation.
            # w_rgb=RGBXY_method.run_one_ASAP(palette, img_data, None)
#### star triangulation using close to black pigment as first color
            w_rgb=Additive_mixing_layers_extraction.Get_ASAP_weights_using_Tan_2016_triangulation_and_then_barycentric_coordinates(img_data, palette, "None", order=0)

            w_rgb=w_rgb.reshape((-1,num_layers))
            w_rgbxy_values=RGBXY_mixing_weights.data
            w_rgbxy_values=w_rgbxy_values.reshape((-1,6))
            w_rgbxy_indices=RGBXY_mixing_weights.indices.reshape((-1,6))
            
            mult, _ = pyopencl_example.prepare_openCL_multiplication( w_rgb, w_rgbxy_values, w_rgbxy_indices )
            final_mixing_weights=mult(w_rgb)
            layers=final_mixing_weights.reshape((the_image.shape[0], the_image.shape[1], num_layers))
            print (layers.shape)
            ## Send data back.
            # print ("send hull edges")
            # await websocket.send( json.dumps( convex_hull_edges.tolist() ) )
            print( "Sending weights..." )
            # await websocket.send( json.dumps( layers.tolist() ) )
            # await websocket.send( np.ascontiguousarray( layers, np.float32 ).tobytes() )
            ## HACK: Send uint8 for speed.
            await websocket.send( np.ascontiguousarray( ( layers*255. ).round().clip(0,255), np.uint8 ).tobytes() )
            print( "... finished." )
        
        elif msg == "automatically-compute-palette":
            ## Receive parameters from the websocket.
            # No additional parameters. Compute an automatic palette for `the_image`.            
            ## Compute palette.
            palette=Additive_mixing_layers_extraction.Hull_Simplification_determined_version(the_image[:,:,:3].reshape((-1,3))/255.0, "./example-", SAVE=False)
            hull=ConvexHull(palette)
            print ("finish compute palette")
            ## Send data back.
            await websocket.send( json.dumps( {'vs': (palette*255).tolist(), 'faces': (hull.points[hull.simplices]*255).tolist() } ) )
        
        elif msg == "user-choose-number-compute-palette":
            palette_size=await websocket.recv()
            palette_size=int(palette_size)
            print ("user choose palette size is: ", palette_size)
            ## Compute palette.
            palette=Additive_mixing_layers_extraction.Hull_Simplification_old(the_image[:,:,:3].reshape((-1,3))/255.0, palette_size, "./example-")
            hull=ConvexHull(palette)
            print ("finish compute palette")
            ## Send data back.
            await websocket.send( json.dumps( {'vs': (palette*255).tolist(), 'faces': (hull.points[hull.simplices]*255).tolist() } ) )
        
        elif msg == "random-add-one-more-color":
            palette = await websocket.recv()
            print (palette)

            ## Parse the parameters.
            palette = json.loads( palette )
            palette = np.asarray(palette)/255.0
            
            print (palette)

            hull=ConvexHull(palette)
            print ("finish compute palette")
            ## Send data back.
            await websocket.send( json.dumps( {'vs': (palette*255).tolist(), 'faces': (hull.points[hull.simplices]*255).tolist() } ) )
        
        else:
            print( "Unknown message:", msg )



port_websocket = 9988
port_http = 8000

## Also start an http server on port 8000
def serve_http( port ):
    import os
    pid = os.fork()
    ## If we are the child, serve files
    if pid == 0:
        import http.server
        import socketserver
        ## Via: https://stackoverflow.com/questions/4465959/python-errno-98-address-already-in-use/25529620#25529620
        socketserver.TCPServer.allow_reuse_address
        Handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("localhost", port), Handler) as httpd:
            print("Serving HTTP on port", port)
            httpd.serve_forever()

## This is too annoying because of address re-use.
# serve_http( port_http )

import argparse
parser = argparse.ArgumentParser( description = "A compute server for interactive layer editing." )
parser.add_argument( "--port", type = int, default = port_websocket, help="The port to listen on." )
args = parser.parse_args()
port_websocket = args.port

print("WebSocket server on port", port_websocket )
start_server = websockets.serve( layer_server, 'localhost', port_websocket, max_size = None )
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
