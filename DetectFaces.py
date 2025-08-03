import cv2
import sys
import numpy as np
import Recognition
import os

PROTOTXT_PATH = "face_detector_model/deploy.prototxt.txt"
CAFFEMODEL_PATH = "face_detector_model/res10_300x300_ssd_iter_140000.caffemodel"

try:
    detection_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
except cv2.error as e:
    print(f"Error loading DNN model: {e}")
    sys.exit(1)

def get_all_image_paths():
    image_paths = []
    for root, _, files in os.walk("Images"):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

Image_Paths = get_all_image_paths()

def detectFaceOnImage(IMAGE_PATH, IMAGE, CONFIDENCE_THRESHOLD):
    if IMAGE is None: return

    original_h, original_w = IMAGE.shape[:2]
    working_image = IMAGE.copy()
    scale = 800 / max(original_w, original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)
    resized_image = cv2.resize(working_image, (new_w, new_h))

    (h, w) = IMAGE.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(resized_image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    detection_net.setInput(blob)
    detections = detection_net.forward()
    
    for det_idx in range(0, detections.shape[2]):
        confidence = detections[0, 0, det_idx, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            box = detections[0, 0, det_idx, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            name, perc = Recognition.recognize_face(IMAGE_PATH, 'database')
            name = str(name).replace("_", " ")
            if name == "None": name = "Unknown"
            text = f"{name}({perc:.2f}%)"
            y = endY + 40 if endY + 20 > 20 else endY - 20
            cv2.rectangle(IMAGE, (startX, startY), (endX, endY), (238, 66, 66), 4)
            putText_with_outline(IMAGE, text, (startX-30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (170, 255, 57), thickness=2, outline_thickness=6)

def putText_with_outline(image, text, org, fontFace, fontScale, color, thickness, outline_thickness=None, outline_color=(0, 0, 0)):
    if outline_thickness is None:
        outline_thickness = thickness + 2
    cv2.putText(image, text, org, fontFace, fontScale, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.putText(image, text, org, fontFace, fontScale, color, thickness, cv2.LINE_AA)