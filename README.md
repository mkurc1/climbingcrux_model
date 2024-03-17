# Climbing Crux Model

The climbing crux model is a machine-learning project that aims to recognize climbing holds and the distance between them from a photo and suggest routes that fit the user's climbing level.

The model recognizes climbing holds and volumes from a photo using the YOLOv8 object detection algorithm. The model is trained on a custom dataset containing photos of climbing walls and annotations in the YOLO format.

![Climbing holds detection preview](./resources/climbing_holds_detection_preview.jpg)

## Data

To achieve the goal of this project, I will use photos from a private collection. You can download the dataset from the link below. It also contains the annotations in the YOLO format.

* [Download the dataset](https://drive.google.com/file/d/1JBzTWpQVjzBkB_mmd7ztzu2ifw78tLrx/view?usp=sharing)

## Distance detection

The distance between climbing holds is calculated using the distance between the centers of the bounding boxes of the detected climbing holds. The distance is calculated in pixels and then converted to centimeters using the reference to aruco marker.

* [Used marker](./resources/aruco_marker_5x5_200px.png)

![Distance detection preview](./resources/aruco_marker_5x5_200px_preview.png)

## Useful links

* [Train Yolov8 object detection on a custom dataset](https://www.youtube.com/watch?v=m9fH9OWn8YM)
* [Object Detection with YOLO v8 on Mac M1](https://www.youtube.com/watch?v=kEcWUZ8unmc)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
