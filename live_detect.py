# Live Object Detector (student starter)
# Run it from inside this folder (labels.json and detector.onnx must be next to it):
#     python live_detect.py
# Lines marked (given) are done for you. Your job is TODO 1 to TODO 5. Do them in order.
# To stop: click the video window and press q. If nothing responds, press Ctrl+C in the terminal.

import json
import cv2
import numpy as np

# (given) settings. You change MIN_SCORE in step 5.
MIN_SCORE = 0.5
SIZE = 640
STRIDES = [8, 16, 32]
FONT = cv2.FONT_HERSHEY_SIMPLEX

with open("labels.json") as file:
    coco = json.load(file)["coco"]
NAMES = [coco[key] for key in sorted(coco, key=int)]

net = cv2.dnn.readNetFromONNX("detector.onnx")

cells = [SIZE // stride for stride in STRIDES]
GRID = np.concatenate([np.indices((n, n))[::-1].reshape(2, -1).T for n in cells])
STRIDE = np.concatenate([np.full((n * n, 1), s) for n, s in zip(cells, STRIDES)])


# (given) detect(frame) looks at one picture and returns a list of (name, box).
def detect(frame):
    # (given) A: fit the picture into a 640 by 640 square, padded with grey, so nothing is stretched.
    height, width = frame.shape[:2]
    scale = min(SIZE / height, SIZE / width)
    small = cv2.resize(frame, (round(width * scale), round(height * scale)))
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    canvas[:small.shape[0], :small.shape[1]] = small

    # (given) B: run the model. output has 8400 rows (guesses). Each row has 85 numbers:
    #          4 for a box, 1 for "is there an object here?", and 80 name scores.
    net.setInput(cv2.dnn.blobFromImage(canvas))
    output = net.forward()[0]

    # (given) C: turn each guess's 4 box numbers into a box in pixels.
    centres = (output[:, :2] + GRID) * STRIDE
    sizes = np.exp(output[:, 2:4]) * STRIDE
    # TODO 1: score every guess.
    scores = output[:, 4:5]*output[:, 5:]
    class_ids = scores.argmax(axis = 1)
    confidences = scores.max(axis = 1)

    # (given) D: change each box into (x, y, w, h) on your ORIGINAL picture.
    boxes = (np.hstack([centres - sizes / 2, sizes]) / scale).astype(int)
    keep = cv2.dnn.NMSBoxesBatched(boxes.tolist(),confidences.tolist(),class_ids.tolist(),MIN_SCORE,0.5)
    return [(NAMES[class_ids[i]], boxes[i])for i in keep]

video = cv2.VideoCapture(0)
while True:
    ok, frame = video.read()
    if not ok:
        break
    for name, (x, y, w, h) in detect(frame):
        # (given) a different colour for each name
        hsv = np.uint8([[[NAMES.index(name) * 29 % 180, 255, 255]]])
        colour = tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0])
        cv2.rectangle(frame, (x, y), (x+w, y+h), colour, 2)

        cv2.putText(frame, name, (x, y - 6), FONT, 0.6, colour, 2)

    cv2.imshow("Live detect", frame)
    if cv2.waitKey(1) & 255 == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
