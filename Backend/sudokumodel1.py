import numpy as np
import cv2
import operator
import matplotlib.pyplot as plt
from keras.models import model_from_json

def convert_when_colour(colour, img):
    """Dynamically converts an image to colour if the input colour is a tuple and the image is grayscale."""
    if len(colour) == 3:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def pre_process_image(img, skip_dilate=False):
    """Uses a blurring function, adaptive thresholding and dilation to expose the main features of an image."""
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    proc = cv2.bitwise_not(proc, proc)
    if not skip_dilate:
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        proc = cv2.dilate(proc, kernel)
    return proc

def find_corners_of_largest_polygon(img):
    """Finds the 4 extreme corners of the largest contour in the image."""
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

def distance_between(p1, p2):
    """Returns the scalar distance between two points"""
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))

def crop_and_warp(img, crop_rect):
    """Crops and warps a rectangular section from an image into a square of similar size."""
    top_left, top_right, bottom_right, bottom_left = crop_rect
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    side = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
    ])
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (int(side), int(side)))

def infer_grid(img):
    """Infers 81 cell grid from a square image."""
    squares = []
    side = img.shape[0] / 9
    for j in range(9):
        for i in range(9):
            p1 = (i * side, j * side)
            p2 = ((i + 1) * side, (j + 1) * side)
            squares.append((p1, p2))
    return squares

def cut_from_rect(img, rect):
    """Cuts a rectangle from an image using the top left and bottom right points."""
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]

def scale_and_centre(img, size, margin=0, background=0):
    """Scales and centres an image onto a new background square."""
    h, w = img.shape[:2]
    def centre_pad(length):
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))

def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
    """Finds the biggest connected pixel structure in the image."""
    img = inp_img.copy()
    height, width = img.shape[:2]
    max_area = 0
    seed_point = (None, None)

    if scan_tl is None:
        scan_tl = [0, 0]

    if scan_br is None:
        scan_br = [width, height]

    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            if img.item(y, x) == 255 and x < width and y < height:
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:
                    max_area = area[0]
                    seed_point = (x, y)

    return seed_point

def extract_digits(img):
    """Extracts digits from an image using contour detection."""
    img = pre_process_digit_image(img)
    
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    digits = []

    for c in contours:
        if cv2.contourArea(c) > 100: 
            x, y, w, h = cv2.boundingRect(c)
            digit = img[y:y+h, x:x+w]
            digit = cv2.resize(digit, (28, 28))
            digits.append(digit)
    return digits

def show_image(img, title="Image"):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def pre_process_digit_image(img):
    """Preprocesses the digit image to remove the box around digits."""
    scale_factor = 2
    upscaled = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    # proc = cv2.blur(upscaled, (9, 9))
    proc = cv2.GaussianBlur(upscaled.copy(), (9, 9), 0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    proc = cv2.bitwise_not(proc)

    margin_w = int(proc.shape[1] * 0.15)
    margin_h = int(proc.shape[0] * 0.15)

    proc[0:margin_h, :] = 0
    proc[-margin_h:, :] = 0
    proc[:, 0:margin_w] = 0
    proc[:, -margin_w:] = 0

    # show_image(proc, "Preprocessed Image")
    return proc

def recognize_digit(img):
    """Recognizes a single digit using EasyOCR."""
    predictions = model.predict(img)
    predicted_label = np.argmax(predictions)
    return predicted_label if predicted_label else None
  
def recognize_digits(digits):
    """Recognizes all digits in the grid using EasyOCR."""
    recognized_digits = []
    for digit_img in digits:
        if np.mean(digit_img) < 50:
            recognized_digits.append(None)
        else:
            recognized_digit = recognize_digit(digit_img)
            recognized_digits.append(recognized_digit if recognized_digit is not None else None)
    return recognized_digits

def print_predictions_grid(predictions):
    """Prints the predictions as a 9x9 grid."""
    for i in range(9):
        row = predictions[i * 9:(i + 1) * 9]
        print(" ".join(str(num) for num in row))

def load_model(model_json_path, model_weights_path):
    """Loads a pre-trained Keras model from JSON and weights files."""
    with open(model_json_path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(model_weights_path)
    return model

def main(image_path):
    """Main function to process the image, extract digits, and make predictions."""
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Image not found at path: {image_path}")
        return
    
    # Load model
    model = load_model(model_json_path, model_weights_path)
    
    # Preprocess image
    preprocessed_img = pre_process_image(img)
    
    # Find corners and warp perspective
    corners = find_corners_of_largest_polygon(preprocessed_img)
    warped_img = crop_and_warp(img, corners)
    
    # Infer grid
    grid_squares = infer_grid(warped_img)
    
    # Extract digits from each grid square
    digits = []
    for idx, sq in enumerate(grid_squares):
        rect_img = cut_from_rect(warped_img, sq)
        extracted_digits = extract_digits(rect_img)
        if len(extracted_digits) > 0:
            digits.append(extracted_digits[0]) 
        else:
            digits.append(None)
    
    # Predict digits
    predictions = []
    for digit in digits:
        if digit is None:
            predictions.append(".") 
        else:
            recognized_digits = recognize_digits([digit])
            predictions.append(str(recognized_digits[0]) if recognized_digits[0] is not None else ".")
    
    # Show results
    print("Predictions:")
    print_predictions_grid(predictions)
    
    return predictions

if __name__ == "__main__":
    image_path = './images/sudoku4.png'
    model_json_path = './model.json'
    model_weights_path = './model.weights.h5'
    
    main(image_path, model_json_path, model_weights_path)

