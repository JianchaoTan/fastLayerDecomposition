#!/usr/bin/env python3

import asyncio
import websockets
# import json
# import numpy as np

async def layer_server( websocket, path ):
    async for msg in websocket:
        print( msg )

port_websocket = 9988
print("WebSocket server on port", port_websocket )
start_server = websockets.serve( layer_server, 'localhost', port_websocket, max_size = None )
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
