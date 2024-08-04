import numpy as np
import cv2 as cv
import operator

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

def infer_grid(warped_image):
    """Infer the 81-cell grid from the square image"""
    side = warped_image.shape[0] / 9
    squares = []
    for j in range(9):
        for i in range(9):
            p1 = (i * side, j * side)  # Top left corner of a box
            p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner
            squares.append((p1, p2))
    return squares

def extract_digit(img, rect, size):
    """Extract the digit from a cell if present"""
    x1, y1 = map(int, rect[0])
    x2, y2 = map(int, rect[1])
    digit = img[y1:y2, x1:x2]
    h, w = digit.shape[:2]
    margin = int(np.mean([h, w]) / 2.5)
    _, bbox, _ = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
    digit = digit[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]
    if w > 0 and h > 0 and (w * h) > 100:
        return scale_and_center(digit, size, 4)
    else:
        return np.zeros((size, size), np.uint8)

def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
    """Find the largest feature in an image"""
    img = inp_img.copy()
    h, w = img.shape[:2]

    if scan_tl is None: scan_tl = [0, 0]
    if scan_br is None: scan_br = [w, h]
    max_area = 0
    seed_point = (None, None)

    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            if img[y, x] == 255:
                area = cv.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:
                    max_area = area[0]
                    seed_point = (x, y)
    
    for x in range(w):
        for y in range(h):
            if img[y, x] == 64:
                cv.floodFill(img, None, (x, y), 255)
    
    mask = np.zeros((h + 2, w + 2), np.uint8)
    area = cv.floodFill(img, mask, seed_point, 255)
    bbox = cv.boundingRect(mask[1:-1, 1:-1])
    bbox = [(bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])]
    return area[0], bbox, seed_point


def scale_and_center(img, size, margin=0):
    """Scale and center the digit within the square"""
    h, w = img.shape[:2]

    def center_pad(length):
        if length % 2 == 0:
            return length // 2
        else:
            return length // 2, length // 2 + 1

    def scale_to_size(size, max_size):
        return size * (max_size / max(size))

    if h > w:
        t_pad, b_pad = center_pad(h - w)
        l_pad, r_pad = 0, 0
    else:
        l_pad, r_pad = center_pad(w - h)
        t_pad, b_pad = 0, 0
    img = cv.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv.BORDER_CONSTANT, None, (0, 0, 0))

    img = cv.resize(img, (size - 2 * margin, size - 2 * margin))
    img = cv.copyMakeBorder(img, margin, margin, margin, margin, cv.BORDER_CONSTANT, None, (0, 0, 0))
    return img

def main():
    # Preprocess the image and find the largest contour
    original, processed = preprocess_image('./images/sudoku4.png')
    largest_contour = find_largest_contour(processed)
    
    # Get the corner points and draw them
    corner_points = get_corner_points(largest_contour)
    draw_corners(original, corner_points)
    
    # Warp perspective to get a top-down view of the Sudoku
    warped_image = warp_perspective(original, corner_points)
    
    # Infer the grid and extract digits
    squares = infer_grid(warped_image)
    digits = []
    for square in squares:
        digit = extract_digit(warped_image, square, 28)
        digits.append(digit)
    
    # Display the results
    cv.imshow("Processed", processed)
    cv.imshow("Corners", original)
    cv.imshow("Warped", warped_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
