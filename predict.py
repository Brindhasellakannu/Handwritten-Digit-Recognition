
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("model/digit_model.h5")


def predict_digit(image_path):
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Invert colors: black drawing becomes white on a black background
    img = 255 - img

    # Remove faint background noise
    _, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)

    # Find the boundaries of the drawn digit
    coords = cv2.findNonZero(img)

    if coords is None:
        raise ValueError("Please draw a digit before clicking Predict.")

    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]

    # Resize the digit while preserving its proportions
    scale = 20.0 / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )

    # Center the digit on a 28x28 black background
    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img

    # Normalize pixel values
    img = canvas.astype("float32") / 255.0

    # Add the channel dimension if the model expects it
    if len(model.input_shape) == 4:
        img = img.reshape(1, 28, 28, 1)
    else:
        img = img.reshape(1, 28, 28)

    # Predict
    prediction = model.predict(img, verbose=0)
    digit = np.argmax(prediction)

    return int(digit)
