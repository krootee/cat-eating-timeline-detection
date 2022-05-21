import torch
import sys

model = torch.hub.load('ultralytics/yolov5', 'yolov5l')

results = model(sys.argv[1])

results.render()
results.show()
