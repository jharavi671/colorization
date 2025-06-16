import numpy as np
import cv2
from PIL import Image, ImageOps
import os

# Model paths
MODEL_DIR = "model"
PROTOTXT = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
MODEL = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
POINTS = os.path.join(MODEL_DIR, "pts_in_hull.npy")

# Load model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorize_image(image_path):
    image = cv2.imread(image_path)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    return Image.fromarray(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB))

def apply_edits(image_path, brightness, contrast, border_color=None):
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    edited = cv2.convertScaleAbs(img_array, alpha=contrast, beta=brightness)
    edited_image = Image.fromarray(edited)

    if border_color:
        edited_image = ImageOps.expand(edited_image, border=20, fill=border_color)

    return edited_image
