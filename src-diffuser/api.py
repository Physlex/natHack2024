import datetime
import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
import subprocess

import sys

sys.path.append("DreamDiffusion/code")

ddapi = FastAPI()
ddapi.mount(
    "/generated", StaticFiles(directory="generated", html=True), name="generated"
)


@ddapi.get("/", tags=["Root"])
async def read_root():
    """Return a friendly greeting."""
    return {"message": "Hello, World!"}


WEBHOOK_URL = "https://discord.com/api/webhooks/1307663613804281906/V5NO2eXZR0lymqRj2igLks4N-SsPKz0YNblMfVg-udG7RuCJ3A74nD5Oeuji2wGaxLUZ"


@ddapi.post("/gen-img")
async def gen_img(request: Request):
    # Run the gen_eval_eeg.py script

    result = subprocess.run(["python", script_path], capture_output=True, text=True,cwd="DreamDiffusion/code")
    
    if result.returncode != 0:
        return {"error": result.stderr}
    
    # Assuming the script generates an output file in the "generated" directory
    ft = "%Y-%m-%dT%H-%M-%S"
    outfilename = f"{datetime.datetime.now().strftime(ft)}.png"
    outfilepath = os.path.join("generated", outfilename)
    
    return {"result": f"{str(request.base_url)}generated/{outfilename}"}


# run with:
# uvicorn api:ddapi --port 8888 --reload