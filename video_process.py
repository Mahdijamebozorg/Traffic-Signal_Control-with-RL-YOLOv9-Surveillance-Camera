# !pip install ultralytics
# !pip install supervision

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict, deque
import argparse

# Get params from cmd
parser = argparse.ArgumentParser(
                    prog='yolov8',
                    description='This program help to track object and maintain in & out count',
                    epilog='Text at the bottom of help')
parser.add_argument('-i', '--input',required=True)
parser.add_argument('-o', '--output',required=True)
args = parser.parse_args()

if __name__ != "__main__":
    exit

# Contants
SOURCE_VIDEO_PATH = args.input
TARGET_VIDEO_PATH = args.output
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
MODEL_NAME = "yolov9c.pt"
# MODEL_RESOLUTION = 1280
SOURCE = np.array([
    [1252, 787],
    [2298, 803],
    [5039, 2159],
    [-550, 2159]
])
TARGET_WIDTH = 25
TARGET_HEIGHT = 250
TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])

# Transform Perspective
class ViewTransformer:

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

# Model
model = YOLO(MODEL_NAME)

# Frame gens
video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)

# tracer initiation
byte_track = sv.ByteTrack(
    frame_rate=video_info.fps, track_activation_threshold=CONFIDENCE_THRESHOLD
)

thickness = sv.calculate_dynamic_line_thickness(
    resolution_wh=video_info.resolution_wh
)
text_scale = sv.calculate_dynamic_text_scale(
    resolution_wh=video_info.resolution_wh
)

# BoundingBoxAnnotator
bounding_box_annotator = sv.BoundingBoxAnnotator(
    thickness=thickness
)

# LabelAnnotator
label_annotator = sv.LabelAnnotator(
    text_scale=text_scale,
    text_thickness=thickness,
    text_position=sv.Position.BOTTOM_CENTER
)
# TraceAnnotator
trace_annotator = sv.TraceAnnotator(
    thickness=thickness,
    trace_length=video_info.fps * 2,
    position=sv.Position.BOTTOM_CENTER
)

# counter line
start, end = sv.Point(x=0, y=video_info.height/2), sv.Point(x=video_info.width, y=video_info.height/2)
line_zone = sv.LineZone(start=start, end=end)
line_innotator = sv.LineZoneAnnotator(
    thickness=thickness,
    text_scale=text_scale
)

# PolygonZone
polygon_zone = sv.PolygonZone(
    polygon=SOURCE,
    frame_resolution_wh=video_info.resolution_wh
)
polygon_innotator = sv.PolygonAnnotator(
    thickness=thickness,
)

coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

# Open target video
# TODO: This will be removed when connected to SC
with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:

    # loop over source video frames
    for frame in tqdm(frame_generator, total=video_info.total_frames):

        result = model(frame,
                    #    imgsz=MODEL_RESOLUTION,
                       verbose=False
                       )[0]
        detections = sv.Detections.from_ultralytics(result)

        # filter out detections by class and confidence
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        detections = detections[detections.class_id != 0]

        # filter out detections outside the zone
        detections = detections[polygon_zone.trigger(detections)]

        # refine detections using non-max suppression
        detections = detections.with_nms(IOU_THRESHOLD)

        # pass detection through the tracker
        detections = byte_track.update_with_detections(detections=detections)

        # trigger counter line
        crossed_in, crossed_out = line_zone.trigger(detections)

        points = detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        )

        # calculate the detections position inside the target RoI
        points = view_transformer.transform_points(points=points).astype(int)

        # store detections position
        for tracker_id, [_, y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)

        # format labels
        labels = []

        for tracker_id in detections.tracker_id:
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                # calculate speed
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                distance = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / video_info.fps
                speed = distance / time * 3.6
                labels.append(f"#{tracker_id} {int(speed)} km/h")


        annotated_frame = frame.copy()
        # line innotator
        line_innotator.annotate(
            frame=annotated_frame, line_counter=line_zone
        )
        # polygon innotator
        polygon_innotator.annotate(
            scene=annotated_frame, detections=detections
        )
        trace_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        # bounding box annotator
        bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        # label annotator
        label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        # add frame to target video
        # TODO: Remove when connected to cls
        # SC
        sink.write_frame(annotated_frame)