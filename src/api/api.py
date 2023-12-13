import os
import io
import base64
from pydantic import BaseModel
from typing import Dict, Any, Union
from starlette.responses import StreamingResponse, Response
from fastapi import FastAPI, File, UploadFile, APIRouter, HTTPException

from constants import *
from googlifier import Googlifier


# Initialize fastapi instance
app = FastAPI()
dev_router = APIRouter()
prod_router = APIRouter()

googlifier = Googlifier(CONFIG_FILE_PATH)


class ImageBase64(BaseModel):
    base64_str: str


@dev_router.post("/googlify_upload_file/")
async def googlify_upload_file(file: UploadFile = File(...)) -> Any:
    """ The endpoint that adds googly eyes to your image.

    Args:
        file: A file that should represent an image as an UploadFile object.

    Returns: A dictionary with the service response or an HTTPException.
    """

    # Check that the file is an image with an acceptable format
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    contents = await file.read()

    # preprocess
    success, image_with_googly_eyes = googlifier.detect_eyes_and_googlify(contents)

    if not success:
        return HTTPException(status_code=400, detail="Corrupt input file.")

    return StreamingResponse(io.BytesIO(image_with_googly_eyes), media_type="image/png")


@prod_router.post("/googlify/", response_model=ImageBase64)
async def googlify(image_base64: ImageBase64) -> Any:
    """ The endpoint that adds googly eyes to your image.

    Args:
        image_base64: An image in the format base64.

    Returns: A dictionary with the service response or an HTTPException.
    """

    try:
        contents = base64.b64decode(image_base64.base64_str)
    except:
        raise HTTPException(status_code=400, detail="Unsupported file type.")


    # Googlify main function
    success, image_with_googly_eyes = googlifier.detect_eyes_and_googlify(contents)

    if not success:
        return HTTPException(status_code=400, detail="Corrupt input file.")

    base64_image = base64.b64encode(image_with_googly_eyes).decode('utf-8')

    return {"base64_str": base64_image}

# Only include the "googlify_upload_file" in development mode for easy testing with using the swagger interface.
app.include_router(prod_router)
if os.environ.get('RUNNING_MODE') == 'dev':
    app.include_router(dev_router)
