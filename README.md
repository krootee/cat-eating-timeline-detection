**Introduction** <br>
Main server detects eating cat on received images and persists events to disk. <br>
API call is used to analyze time periods and remove false positives.

**Limitations** <br>
Would not work at night time by default - requires infrared camera.  <br>
If you could place camera at right spot, then you could reduce false detections to zero by computing distance between cat and bowls, but in our camera setup it was not possible.

**Project installation** <br>
1. Create Python 3.10 environment using conda / <your favorite tool>
2. Install dependencies with "pip install -r requirements.txt"

**Server** <br>
Run main application with "uvicorn main:app --host 0.0.0.0 --port 8082". <br>
Configured to use YOLO v5 large model by default.

**Client application** <br>
Run "python camera-app.py <server ip> 8082". <br>
Requires DepthAI camera, but could be easily adapted to use any web camera that allows capturing of RGB image 1 time per second.

**Helper application** <br>
Run "python show-detections.py <image path>" to analyze given image you show detected bounding boxes, classes and confidence levels. 
