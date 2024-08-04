import numpy as np
import cv2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = load_model('mnist_number_recognizer.keras')

random_index = np.random.randint(0, len(x_test))

image = x_test[random_index]
true_label = y_test[random_index]

image_for_prediction = image.reshape(1, 28, 28, 1).astype('float32') / 255.0

predictions = model.predict(image_for_prediction)
predicted_label = np.argmax(predictions)

image_display = (image * 255).astype(np.uint8)  

cv2.imshow("Random Digit", image_display)
print(f"True Label: {np.argmax(true_label)}")
print(f"Predicted Label: {predicted_label}")
cv2.waitKey(0) 

cv2.destroyAllWindows()
