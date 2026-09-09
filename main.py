import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    # Find sum of x + y for each vertex
    s = pts.sum(axis = 1)

    # The smallest x + y is the top left
    rect[0] = pts[np.argmin(s)]

    # The largest x + y is the bottom right
    rect[2] = pts[np.argmax(s)]

    # Find the difference between y and x (y - x) for each vertex
    diff = np.diff(pts, axis = 1)

    # The smallest y - x (x > y) is the top right
    rect[1] = pts[np.argmin(diff)]

    # The largest y - x (y > x) is the bottom left
    rect[3] = pts[np.argmax(diff)]

    return rect;

def get_perspective_matrix(image):
    # Turn the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur image for reduce noises
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny algorithm to find edges
    edged = cv2.Canny(blurred, 50, 150)

    # Thicken the lines to close the gaps
    kernel = np.ones((3, 3), np.uint8);
    edged = cv2.dilate(edged, kernel, iterations=1)

    # Find all contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort to find the largest area (field)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find field contour
    field_contour = None
    # min_field_area = None
    # h_img, w_img = image.shape[:2]

    # # The contour area must more than 20% of original image
    # min_field_area = (h_img * w_img) * 0.2

    for c in contours:
        # if cv2.contourArea(c) < min_field_area:
        #     continue

        # Find each contour's perimeter and make sure each conture in closed shape
        peri = cv2.arcLength(c, True)

        # Approximating vertices of shape
        vertices = cv2.approxPolyDP(c, 0.02 * peri, True)

        # If this contour has 4 vertices it is the field
        if len(vertices) == 4:
            field_contour = vertices
            break

    if field_contour is None:
        print("Not found a rectangle please adjust your image properties again")
        return image

    # Resize field contour that might contains over 2 dimension value
    pts = field_contour.reshape(4, 2)

    # Order the point from top left -> top right -> bottom right -> bottom left
    rect = order_points(pts)

    # Destructure the array
    (tl, tr, br, bl) = rect;

    width_bottom = np.linalg.norm(br - bl)
    width_top = np.linalg.norm(tr - tl)

    # Find the max width that will be the real width of rectangle
    max_width = max(int(width_bottom), int(width_top))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    # Find the max height that will be the real height of rectangle
    max_height = max(int(height_left), int(height_right))

    # Declare destination image size
    dst = np.float32([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ])

    # Calculate the matrix that will change image to rectangle field
    matrix = cv2.getPerspectiveTransform(rect, dst)

    return [max_width, max_height, matrix]

def dynamic_warp_perspective(image):
    (max_width, max_height, matrix) = get_perspective_matrix(image)

    # Stretching four corners of image to fit with dest coordinate
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped;

img_set = ["data/field.png", "data/test1.jpg"]

for i in range(len(img_set)):
    img = cv2.imread(img_set[i])

    if img is not None: 
        rectangle_field = dynamic_warp_perspective(img)

        cv2.imshow(f"Original image {i + 1}", img)
        cv2.imshow(f"Dynamic rectangle field {i + 1}", rectangle_field)

    else:
        print("Can't find the image")
        
cv2.waitKey(0);
cv2.destroyAllWindows()

# # Use camera
# cap = cv2.VideoCapture(0)

# matrix = None
# target_size = None

# while cap.isOpened():
#     ret, frame = cap.read()

#     # If can't read frame from camera
#     if not ret:
#         break;

#     # Find matrix just once for first frame
#     if matrix is None:
#         (max_width, max_height, matrix) = get_perspective_matrix(frame)
#         target_size = (max_width, max_height)

#     # Get an image every frame
#     if matrix is not None:
#         warped_frame = cv2.warpPerspective(frame, matrix, target_size)
#         cv2.imshow("Realtime robot vision", warped_frame)

#     # End program when pressed q
#     if cv2.waitKey(1) & 0xFF == ord('0'):
#         break 

# # Completely close a program
# cap.release()
# cv2.destroyAllWindows()