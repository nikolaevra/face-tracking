from keras.models import model_from_json
from skimage import color, exposure, transform
import numpy as np

IMG_SIZE = 48
CLASS_MAP = {
    2: "Other",
    1: "Open",
    0: "Closed"
}


def preprocess_img(img):
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2,
              :]

    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.rollaxis(img, -1)

    return np.expand_dims(img, axis=0)


class EyeOpener:
    def __init__(self):
        json_file = open('./eye_classifier/model/eyes-model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights("./eye_classifier/model/eyes-model.h5")

    def predict(self, image):
        processed = preprocess_img(image)
        return CLASS_MAP[self.model.predict_classes(processed)[0]]
