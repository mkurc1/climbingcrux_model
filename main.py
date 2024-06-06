import io
import cv2
import imutils
import numpy as np

from fastapi import FastAPI, UploadFile, HTTPException, status
from starlette.responses import StreamingResponse

from src import config, image_utils, objects_detector
from src.aruco_marker import ArucoMarker
from src.route_generator import RouteGenerator

app = FastAPI(title="Climbing Crux Route Generator")


@app.post("/boulder/generate")
async def generate_boulder(file: UploadFile) -> StreamingResponse:
    """
    Generate a boulder route from an image
    """
    contents = await file.read()

    validate_file(file)

    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    img = imutils.resize(img, width=1216)

    try:
        marker = ArucoMarker(config.MARKER_ARUCO_DICT, img, config.MARKER_PERIMETER_IN_CM)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No ArUco marker detected")

    detected_objects = objects_detector.detect(img)

    route_generator = RouteGenerator(
        img_width=img.shape[1],
        img_height=img.shape[0],
        marker=marker,
        detected_objects=detected_objects
    )

    positions = route_generator.generate_route(
        climber_height_in_cm=config.CLIMBER_HEIGHT_IN_CM,
        starting_steps_max_distance_from_ground_in_cm=config.STARTING_STEPS_MAX_DISTANCE_FROM_GROUND_IN_CM
    )

    for climber_position in positions:
        img = image_utils.draw_climber(
            img=img,
            climber=climber_position,
            draw_labels=False,
            draw_centers=False,
        )

    _, im_png = cv2.imencode(".png", img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")


def validate_file(file: UploadFile) -> None:
    if file.content_type not in config.ACCEPTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type",
        )

    real_file_size = 0
    for chunk in file.file:
        real_file_size += len(chunk)
        if real_file_size > config.MAXIMUM_FILE_SIZE:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Too large")
