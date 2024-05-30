import io
import cv2
import imutils
import numpy as np
import uvicorn

from fastapi import FastAPI, UploadFile
from starlette.responses import StreamingResponse

from src import config, image_utils, objects_detector
from src.aruco_marker import ArucoMarker
from src.route_generator import RouteGenerator

app = FastAPI(title="Climbing Crux Route Generator")


@app.post("/boulder/generate")
async def generate_boulder(file: UploadFile):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = imutils.resize(img, width=1216)

    marker = ArucoMarker(config.MARKER_ARUCO_DICT, img, config.MARKER_PERIMETER_IN_CM)
    detected_objects = objects_detector.detect(img)

    route_generator = RouteGenerator(
        img_width=img.shape[1],
        img_height=img.shape[0],
        marker=marker,
        detected_objects=detected_objects
    )

    steps = route_generator.generate_route(
        climber_height_in_cm=config.CLIMBER_HEIGHT_IN_CM,
        starting_steps_max_distance_from_ground_in_cm=40
    )

    for step in steps:
        img = image_utils.draw_climber(
            img=img,
            climber=step,
            draw_labels=False,
            draw_centers=False,
        )

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    _, im_png = cv2.imencode(".png", img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
