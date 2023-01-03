import cv2
import os
import numpy as np
from gtts import gTTS
from playsound import playsound

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

classes = []
inc = 0
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    for i in range(len(class_ids)):
        conf = confidences[i]
        label = classes[class_ids[i]]  # 'label' variable holds name of object detected
        print(label, conf * 100)  # prints object detected with how confident it is in its predition

        voice = str(label) + "in front of you"  # string being passed to convert to voice with gtts

    file_path = 'voice{}.mp3'.format(inc)  # u can specify path to temporarily store text to voice conversion
    inc += 1
    sound = gTTS(text=voice, lang='en')  # text to voice conversion with gtts
    sound.save(file_path)  # voice file saving in specified path
    if class_ids:  # if any object is detected it says the name else says no 'no object detected'
        playsound(file_path)
    else:
        playsound('no_obj.mp3')  # create an mp3 file saying 'no object detected' refer README
    os.remove(file_path)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255, 255, 255), 2)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()