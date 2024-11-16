from websockets.asyncio.server import serve, ServerConnection
from openbci import CytonDaisy
import asyncio
import json
from dataclasses import dataclass
from typing import *

@dataclass
class RWs_InitComms:
    code = "INIT"
    ivl: int # interval in ms to receive data

@dataclass
class RWs_TermComms:
    code = "TERM"

# List of Event Type, Onset, Duration
TimestampData = List[Tuple[int, float, float]]

@dataclass
class RWs_Timestamps:
    code = "TIMESTAMPS"
    data: TimestampData

EEGData = List[float]

@dataclass
class EWs_EmitLatest:
    code = "EMISSION"
    chs: int # channels
    n: int # number of samples
    data: EEGData # chs x n array of float

RecvWSMsgs = Union[RWs_InitComms, RWs_Timestamps, RWs_TermComms]
EmitWSMsgs = Union[EWs_EmitLatest]

def interpret_msg(msg) -> RecvWSMsgs:
    msg_dict: RecvWSMsgs = json.loads(msg)
    res: RecvWSMsgs = None

    if msg_dict['code'] == "INIT":
        ivl = int(msg_dict['ivl'])
        res = RWs_InitComms(ivl=ivl)
    elif msg_dict['code'] == "TIMESTAMPS":
        data: TimestampData = map(lambda ts: (int(ts[0]), float(ts[1]), float(ts[2])), msg_dict['data'])
        res = RWs_Timestamps(data=list(data))
    elif msg_dict['code'] == "TERM":
        res = RWs_TermComms()

    return res

async def handler(ws: ServerConnection):
    try:
        while True:
            message = interpret_msg(await ws.recv())
            if isinstance(message, RWs_InitComms):
                # start colecting data
                pass
            elif isinstance(message, RWs_Timestamps):
                # log timestamps, use for emissions
                pass
            elif isinstance(message, RWs_TermComms):
                # stop collecting data
                pass
            print(message)
    except ws.ConnectionClosed:
        pass
    except Exception as e:
        print(e)

async def start_srv():
    async with serve(handler, "", 8001):
        await asyncio.get_running_loop().create_future()

if __name__ == "__main__":
    asyncio.run(start_srv())
    with CytonDaisy("COM17") as board:
        print(board.board.is_prepared())
        print("dssdf", str(board.get_data()))