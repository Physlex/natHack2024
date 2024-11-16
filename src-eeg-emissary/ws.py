from websockets.asyncio.server import serve, ServerConnection
from websockets.exceptions import ConnectionClosedError
from openbci import CytonDaisy
import asyncio
import json
from dataclasses import dataclass
import dataclasses
from typing import *

@dataclass
class RWs_InitComms:
    code = "INIT"
    ivl: int # interval in ms to receive data

@dataclass
class RWs_TermComms:
    code = "TERM"

EEGData = List[List[float]]

@dataclass
class EWs_EmitLatest:
    code = "EMISSION"
    nchs: int # channels
    n: int # number of samples
    hz: float # sampling rate
    data: EEGData # chs x n array of float

RecvWSMsgs = Union[RWs_InitComms, RWs_TermComms]
EmitWSMsgs = Union[EWs_EmitLatest]



class WsEEGAsyncHandler:
    def __init__(self, board: CytonDaisy):
        self.board = board
        self.ivl: int
        self.emitting = False

    @classmethod
    def interpret_msg(_, msg) -> RecvWSMsgs:
        msg_dict: RecvWSMsgs = json.loads(msg)
        res: RecvWSMsgs = None

        if msg_dict['code'] == "INIT":
            ivl = int(msg_dict['ivl'])
            res = RWs_InitComms(ivl=ivl)
        elif msg_dict['code'] == "TERM":
            res = RWs_TermComms()

        return res
    
    async def emit_eeg(self, ws: ServerConnection):
        while self.emitting:
            samp = self.board.get_data()
            print(samp.size, self.board.chs)
            msg = EWs_EmitLatest(nchs=len(self.board.chs), n=samp.size, hz=self.board.hz, data=samp.tolist())
            await ws.send(json.dumps(dataclasses.asdict(msg)))
            await asyncio.sleep(self.ivl/1000)

    async def handler(self, ws: ServerConnection):
        try:
            while True:
                message = WsEEGAsyncHandler.interpret_msg(await ws.recv())
                if isinstance(message, RWs_InitComms):
                    # start colecting data
                    self.ivl = message.ivl
                    self.emitting = True
                    await self.emit_eeg(ws)
                elif isinstance(message, RWs_TermComms):
                    # stop collecting data
                    self.emitting = False
        except ConnectionClosedError:
            pass
        except Exception as e:
            print(e.with_traceback(None))

async def start_srv():
    with CytonDaisy("COM17") as board:
        handler_cls = WsEEGAsyncHandler(board)
        async with serve(handler_cls.handler, "", 8001, max_size=2**30):
            print("Starting EEG Emissary...")
            await asyncio.get_running_loop().create_future()

if __name__ == "__main__":
    asyncio.run(start_srv())
