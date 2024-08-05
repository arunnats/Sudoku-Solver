import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("TMNIST_Data.csv", header=0)
X = df.drop(columns=['names', 'labels'], axis=1)
y = df['labels']

# Reshape and normalize the input data
X = X.values.reshape(-1, 28, 28, 1).astype('float32') / 255

# One-hot encode the labels (for neural network training)
y_nn = pd.get_dummies(y).values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_nn, test_size=0.2, random_state=42)

# K-Nearest Neighbors training
samples = X.reshape(X.shape[0], -1).astype(np.float32)
responses = y.values.astype(np.float32).reshape(-1, 1)

knn = cv2.ml.KNearest_create()
knn.train(samples, cv2.ml.ROW_SAMPLE, responses)

# Testing part with a sample image
def knn_predict_image(img_path):
    im = cv2.imread(img_path)
    out = np.zeros(im.shape, np.uint8)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if h > 28:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = thresh[y:y + h, x:x + w]
                roismall = cv2.resize(roi, (28, 28))
                roismall = roismall.reshape((1, 784))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = knn.findNearest(roismall, k=1)
                string = str(int((results[0][0])))
                cv2.putText(out, string, (x, y + h), 0, 1, (0, 255, 0))

    # Display images using matplotlib
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Input Image')
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.title('Processed Image')
    plt.imshow(out, cmap='gray')
    plt.show()

knn_predict_image('./images/sudoku1.png')
