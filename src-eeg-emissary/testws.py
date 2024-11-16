import asyncio
import websockets

# 1 GB limit for WebSocket frame size
ONE_GB = 1073741824

async def hello():
    uri = "ws://localhost:8001/"  # Replace with your WebSocket server URL
    async with websockets.connect(uri, max_size=ONE_GB) as websocket:
        print("Collecting...")
        await websocket.send('{"code": "INIT", "ivl": 1000}')
        for i in range(5):
            import json
            resp = json.loads(await websocket.recv())
            print(resp['nchs'], resp['n'])
        await websocket.send('{"code": "TERM"}')
        print("Done!")

asyncio.run(hello())
