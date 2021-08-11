#https://towardsdatascience.com/object-detection-using-yolov3-and-opencv-19ee0792a420



import cv2
import numpy as np 
import argparse
import time

def load_yolo():
  #net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
  #net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")
  net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
  net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
  net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
  classes = []
  with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
  layers_names = net.getLayerNames()
  output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
  colors = np.random.uniform(0, 255, size=(len(classes), 3))
  return net, classes, colors, output_layers


def load_image(img_path):
  # image loading
  img = cv2.imread(img_path)
  img = cv2.resize(img, None, fx=0.4, fy=0.4)
  height, width, channels = img.shape
  return img, height, width, channels

def detect_objects(img, net, outputLayers):			
  blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
  net.setInput(blob)
  outputs = net.forward(outputLayers)
  return blob, outputs

def get_box_dimensions(outputs, height, width):
  boxes = []
  confs = []
  class_ids = []
  for output in outputs:
    for detect in output:
      scores = detect[5:]
      ##print(scores)
      class_id = np.argmax(scores)
      conf = scores[class_id]
      if conf > 0.3:
        center_x = int(detect[0] * width)
        center_y = int(detect[1] * height)
        w = int(detect[2] * width)
        h = int(detect[3] * height)
        x = int(center_x - w/2)
        y = int(center_y - h / 2)
        boxes.append([x, y, w, h])
        confs.append(float(conf))
        class_ids.append(class_id)
  return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img, out): 
  indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
  font = cv2.FONT_HERSHEY_PLAIN
  for i in range(len(boxes)):
    if i in indexes:
      x, y, w, h = boxes[i]
      label = str(classes[class_ids[i]])
      if label=='person':
        color = colors[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
        if ((x>=0) and (x<=180) and (y>=0) and (y<=200)):
          cv2.rectangle(img, (0,0), (200, 200),(0, 0, 255), 2)
          cv2.putText(img, 'DANGER', (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)        
  cv2.imshow("Image", img)
  out.write(img.astype('uint8'))




initime=time.time()

def webcam_detect():
  model, classes, colors, output_layers = load_yolo()
  timee=time.time() - initime
  while timee<5:
    pass
  initime=time.time()
  cap = cv2.VideoCapture(0)
  initime=time.time()
  out = cv2.VideoWriter(
    'output.mp4',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))
  while True:
    _, frame = cap.read()
    timee=time.time() - initime
    if timee>=0:
      height, width, channels = frame.shape
      blob, outputs = detect_objects(frame, model, output_layers)
      boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
      draw_labels(boxes, confs, colors, class_ids, classes, frame, out)
      initime=time.time()
      key = cv2.waitKey(1)
      if key == 27:
        break
      pass
    cv2.imshow("Image", frame)
    out.write(frame.astype('uint8'))
    key = cv2.waitKey(1)
    if key == 27:
      break
  cap.release()
  out.release()

webcam_detect()
cap.release()
out.release()

cv2.destroyAllWindows()