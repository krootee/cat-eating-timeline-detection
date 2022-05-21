import cv2
import depthai as dai
import time
import requests
import os
import sys

IMAGE_FOLDER = 'data'

if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# Create pipeline
pipeline = dai.Pipeline()

camera_rgb = pipeline.create(dai.node.ColorCamera)
camera_rgb.setInterleaved(False)
camera_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
# Downscale 4k -> 720P video frames
camera_rgb.setIspScale(1, 3)

output_frames = pipeline.create(dai.node.XLinkOut)
output_frames.setStreamName("raw-image")
camera_rgb.video.link(output_frames.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    camera_frames = device.getOutputQueue(name="raw-image", maxSize=1, blocking=False)

    while True:
        frame = camera_frames.get().getCvFrame()

        try:
            filename = f'{IMAGE_FOLDER}\\frame-{time.time()}.jpg'
            cv2.imwrite(filename, frame)
            print(f'saved to {filename}')
            cv2.imshow("Raw image", frame)

            url = f'http://{sys.argv[1]}:{sys.argv[2]}/upload'
            file = {
                'file': open(filename, 'rb')
            }
            resp = requests.post(url=url, files=file)
            print(resp.json())
        except BaseException as err:
            print(f'Unexpected {err=}, {type(err)=}')
        finally:
            time.sleep(1.0)
            if file['file']:
                file['file'].close()
            os.remove(filename)

        if cv2.waitKey(1) == ord('q'):
            break
