from __future__ import annotations
import math
import PIL.Image
from fastapi import FastAPI, UploadFile, File
import os
from datetime import datetime, timezone
from PIL import Image
import torch
from dataclasses import dataclass, field
import json
from shutil import copyfile
import copy


SCALED_IMAGE_FIXED_WIDTH: int = 640
YOLOV5_MODEL = 'yolov5l'
STORAGE_FOLDER = 'storage'
DETECTED_FOLDER = 'detected'
IMAGES_FOLDER = 'images'
FALSE_DETECTED_FOLDER = 'false-detected'

images_directories = [IMAGES_FOLDER, DETECTED_FOLDER, STORAGE_FOLDER, FALSE_DETECTED_FOLDER]
for directory in images_directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

print(torch.__version__)
global_cached_yolo_torch_model = torch.hub.load('ultralytics/yolov5', YOLOV5_MODEL, force_reload=True)


def detect_objects(yolo_model, filename: str) -> dict:
    results = yolo_model(filename)
    yolo_objects = results.pandas().xyxy[0].to_dict(orient="records")
    information = f'{datetime.now():%d/%m/%Y, %H:%M:%S} Found objects: '
    for yolo in yolo_objects:
        information += f'{yolo["name"]} [{yolo["confidence"]*100:.2f}%]; '
    print(information)

    return yolo_objects


@dataclass
class YoloObject:
    name: str
    confidence: float


def yolo_to_dataclass(yolo_object: dict) -> YoloObject:
    return YoloObject(yolo_object['name'],
                      yolo_object['confidence'])


def detect_feeding_cat_object(yolo_objects: dict) -> bool:
    cat = None
    for yolo_object in yolo_objects:
        if yolo_object['name'] == 'cat' or yolo_object['name'] == 'dog':
            if not cat or (cat and cat.confidence < yolo_object['confidence']):
                cat = yolo_to_dataclass(yolo_object)

    return cat and cat.confidence > 0.4


def create_history_record(timestamp: int, filename: str, yolo_objects: list) -> dict:
    return {
        'timestamp': float(timestamp) / 1000,
        'filename': filename,
        'objects': yolo_objects
    }


@dataclass(slots=True)
class DetectionsHistory:
    values: list = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.values = self.load()

        if len(self.values) == 0:
            self.values = self.rebuild_history()
            self.commit()

    @staticmethod
    def generate_file_path() -> str:
        return os.path.join(STORAGE_FOLDER, 'history.json')

    @staticmethod
    def rebuild_history() -> list:
        output = list()

        local_yolo_torch_model = torch.hub.load('ultralytics/yolov5', YOLOV5_MODEL, force_reload=True)

        print(f'{datetime.now():%d/%m/%Y, %H:%M:%S} Starting history rebuilding from images archive...')
        for filename in os.listdir(DETECTED_FOLDER):
            if os.path.isfile(os.path.join(DETECTED_FOLDER, filename)):
                print(filename)
                name, extension = os.path.splitext(filename)
                timestamp = int(name)
                detected_objects = detect_objects(local_yolo_torch_model, os.path.join(DETECTED_FOLDER, filename))
                if detect_feeding_cat_object(detected_objects):
                    output.append(create_history_record(timestamp, filename, detected_objects))
                else:
                    os.rename(os.path.join(DETECTED_FOLDER, filename), os.path.join(FALSE_DETECTED_FOLDER, filename))

        print(f'{datetime.now():%d/%m/%Y, %H:%M:%S} ... history rebuilding completed')
        return output

    def load(self) -> list:
        output = list()
        filename = self.generate_file_path()
        if os.path.isfile(filename):
            with open(filename, 'r', encoding='utf-8') as outfile:
                output = json.load(outfile)
        return output

    def commit(self) -> None:
        filename = self.generate_file_path()
        print(f'Saving changes to {filename}')

        if os.path.isfile(filename):
            copyfile(filename, f'{filename}.backup')
        with open(filename, 'w', encoding='utf-8') as outfile:
            json.dump(self.values, outfile, indent=2, ensure_ascii=False)


history_data = DetectionsHistory()
app = FastAPI()


@app.get("/")
async def root():
    return 'Welcome to cat feeding detector!'


def create_eating_period(start: datetime, end: datetime, periods: list) -> None:
    eating_duration = (end - start).total_seconds()

    if eating_duration > 10:
        periods.append({
            'start': f'{start:%d/%m/%Y, %H:%M:%S}',
            'duration_seconds': eating_duration
        })


@app.get("/history/detect-eating")
async def get_cat_eating_periods():
    global history_data

    periods = list()
    period_start: datetime = datetime.fromtimestamp(history_data.values[0]['timestamp'], tz=timezone.utc)
    last_timestamp: datetime = period_start

    for item in history_data.values:
        current_timestamp = datetime.fromtimestamp(item['timestamp'], tz=timezone.utc)

        if (current_timestamp - last_timestamp).total_seconds() > 60:
            create_eating_period(period_start, last_timestamp, periods)
            period_start = current_timestamp

        last_timestamp = current_timestamp

    create_eating_period(period_start, last_timestamp, periods)

    return copy.deepcopy(periods)


def convert_current_datetime_to_int() -> int:
    return int(datetime.now().replace(tzinfo=timezone.utc).timestamp() * 1000)


def save_history_record(filename: str, yolo_objects: list) -> None:
    global history_data
    history_data.values.append(create_history_record(convert_current_datetime_to_int(), filename, yolo_objects))
    history_data.commit()


def generate_unique_name(filename: str) -> str:
    name, extension = os.path.splitext(filename)
    name = convert_current_datetime_to_int()
    return f"{name}{extension}"


def file_saver(filename: str) -> None:
    os.rename(os.path.join(IMAGES_FOLDER, filename), os.path.join(DETECTED_FOLDER, filename))


def resize_image(filename: str) -> str:
    image = Image.open(filename)
    fixed_width = SCALED_IMAGE_FIXED_WIDTH
    width_ratio = (fixed_width / image.size[0])
    scaled_height = int(float(image.size[1]) * float(width_ratio))
    image = image.resize((fixed_width, scaled_height), PIL.Image.ANTIALIAS)
    name, extension = os.path.splitext(filename)
    new_filename = f"{name}_scaled{extension}"
    image.save(new_filename)
    return new_filename


@app.post("/upload")
async def file_uploader(file: UploadFile = File(...)):
    file_content = await file.read()
    unique_filename = generate_unique_name(file.filename)
    filename = os.path.join(IMAGES_FOLDER, unique_filename)
    with open(filename, 'wb') as input_file:
        input_file.write(file_content)
    scaled_image = resize_image(filename)

    global global_cached_yolo_torch_model
    detected_objects = detect_objects(global_cached_yolo_torch_model, scaled_image)

    if detect_feeding_cat_object(detected_objects):
        # double verify detection with fresh torch model and full image - sometimes system produces false positives
        global_cached_yolo_torch_model = torch.hub.load('ultralytics/yolov5', YOLOV5_MODEL, verbose=False)
        detected_objects = detect_objects(global_cached_yolo_torch_model, filename)

        if detect_feeding_cat_object(detected_objects):
            file_saver(unique_filename)
            save_history_record(unique_filename, detected_objects)
        else:
            os.remove(filename)
    else:
        os.remove(filename)

    os.remove(scaled_image)

    return f'Image {file.filename} processed and saved locally as {unique_filename}'
