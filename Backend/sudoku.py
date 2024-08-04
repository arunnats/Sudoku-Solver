import numpy as np
import cv2 as cv
import operator
from tensorflow.keras.models import load_model

def distance_between(p1, p2):
    """Calculate the Euclidean distance between two points"""
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))

def preprocess_image(image_path):
    """Read and preprocess the image"""
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (9, 9), 0)

    thresholded = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    inverted = cv.bitwise_not(thresholded)
    return image, inverted

def find_largest_contour(processed_image):
    """Find the largest contour in the processed image"""
    contours, _ = cv.findContours(processed_image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    return contours[0]

def get_corner_points(contour):
    """Get the corner points from the largest contour."""
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in contour]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in contour]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in contour]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in contour]), key=operator.itemgetter(1))
    return [contour[top_left][0], contour[top_right][0], contour[bottom_right][0], contour[bottom_left][0]]

def draw_corners(image, points):
    """Draw circles on the detected corner points"""
    for point in points:
        cv.circle(image, tuple(point), 5, (0, 0, 255), -1)

def warp_perspective(image, points):
    """Apply perspective transform"""
    top_left, top_right, bottom_right, bottom_left = points
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    side = max([distance_between(bottom_right, top_right), 
                distance_between(top_left, bottom_left),
                distance_between(bottom_right, bottom_left),   
                distance_between(top_left, top_right)])
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    m = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(image, m, (int(side), int(side)))
    return warped

def divide_into_cells(image):
    """Divide the image into cells"""
    grid_size=9
    height, width = image.shape[:2]
    cell_size = height // grid_size
    cells = []
    for i in range(grid_size):
        row_cells = []
        for j in range(grid_size):
            start_x = j * cell_size
            start_y = i * cell_size
            end_x = start_x + cell_size
            end_y = start_y + cell_size
            cell = image[start_y:end_y, start_x:end_x]
            row_cells.append(cell)
        cells.append(row_cells)
    return cells

def preprocess_cell(cell):
    """Preprocess the cell for prediction"""
    gray = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
    resized = cv.resize(gray, (28, 28))  
    normalized = resized.astype('float32') / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1) 
    return reshaped

def predict_digit(cell, model):
    """Predict the digit in a cell"""
    preprocessed_cell = preprocess_cell(cell)
    prediction = model.predict(preprocessed_cell)
    predicted_digit = np.argmax(prediction)
    return predicted_digit

def create_sudoku_matrix(cells, model):
    """Create a 2D matrix with predicted digits and blank spaces"""
    for row in cells:
        for cell in row:
            digit = predict_digit(cell, model)
            print(digit)
        print('\n')

def main():
    model = load_model('mnist_number_recognizer.keras')

    original, processed = preprocess_image('./images/sudoku4.png')
    
    largest_contour = find_largest_contour(processed)
    
    corner_points = get_corner_points(largest_contour)
    
    draw_corners(original, corner_points)
    
    warped_image = warp_perspective(original, corner_points)
    
    cells = divide_into_cells(warped_image)

    sudoku_matrix = create_sudoku_matrix(cells, model)
    
    cv.imshow("Processed", processed)
    cv.imshow("Corners", original)
    cv.imshow("Warped", warped_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
