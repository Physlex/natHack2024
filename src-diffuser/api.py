import datetime
import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import shutil

from DreamDiffusion.code.modelserver import ModelServer

modelserver = ModelServer()
ddapi = FastAPI()
ddapi.mount('/generated', StaticFiles(directory="generated", html=True), name="generated")

@ddapi.get("/", tags=["Root"])
async def read_root():
    """Return a friendly greeting."""
    return {"message": "Hello, World!"}

WEBHOOK_URL = "https://discord.com/api/webhooks/1307663613804281906/V5NO2eXZR0lymqRj2igLks4N-SsPKz0YNblMfVg-udG7RuCJ3A74nD5Oeuji2wGaxLUZ"

class EEGData(BaseModel):
    data: List[List[float]]

@ddapi.post("/gen-img")
async def gen_img(eegdat: EEGData, request: Request):
    # call the image generation function here
    # return file path
    results = modelserver.infer('data/processed_eeg_data_updated.pth', num_samples=5, ddim_steps=250)
    ft = "%Y-%m-%dT%H-%M-%S"
    outfilename = f"{datetime.datetime.now().strftime(ft)}.png"
    outfilepath = os.path.join('generated', outfilename)
    results.save(outfilepath)
    return {"result": f"{str(request.base_url)}generated/{outfilename}"}

# run with:
# uvicorn api:ddapi --port 8888 --reload
