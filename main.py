import cv2
import numpy as np
import math
from enum import Enum

class RockStatus(Enum):
    WAITING = 1,
    ON_ROBOT = 2,
    DELIVERED = 3

class RunMode(Enum):
    PRODUCTION = 1
    SIMULATION = 2

mode = RunMode.SIMULATION

simulation_config = {
    "robot_speed" : 5.0,
    "robot_turn_speed" : 5.0
}

class Rock:
    def __init__(self, rock_id, x, y, color_name, bgr_color):
        self.id = rock_id
        self.x = float(x)
        self.y = float(y)
        self.color_name = color_name
        self.bgr = bgr_color
        self.status = RockStatus.WAITING

    def draw(self, canvas):
        # Draw only the rokcs that not on the robot
        if self.status != RockStatus.ON_ROBOT:
            cv2.circle(canvas, (int(self.x), int(self.y)), 8, self.bgr, -1)
            # Draw border
            cv2.circle(canvas, (int(self.x), int(self.y)), 8, (40, 40, 40), 1)
        

# Create a simulation robot class
class Robot2D:
    def __init__(self, start_x, start_y, theta, width, length):
        self.x = float(start_x)
        self.y = float(start_y)
        self.theta = float(theta)
        self.width = width
        self.length = length
        self.picked = None

    def get_corners(self):
        rad = math.radians(self.theta)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        w2 = self.width / 2
        l2 = self.length / 2

        # Define 4 corners of robot with 0 deg rotating (east)
        # front right, front back, left back, front left
        offsets = [(l2, -w2), (-l2, -w2), (-l2, w2), (l2, w2)]

        # Calculate 4 corners after rotating theta radian
        corners = []
        for dx, dy in offsets:
            px = self.x + (dx * cos_a - dy * sin_a)
            py = self.y + (dx * sin_a + dy * cos_a)
            corners.append([px, py])

        return np.array(corners, dtype=np.int32)

    def draw(self, canvas):
        corners = self.get_corners()

        # Draw robot body (color : red)
        cv2.fillPoly(canvas, [corners], (255, 0, 0))

        # Draw robot border (color : white)
        cv2.polylines(canvas, [corners], isClosed=True, color=(255, 255, 255), thickness=2)

        # Draw an arrow on the robot's body
        rad = math.radians(self.theta)
        head_len = self.length * 0.75
        fx = int(self.x + head_len * math.cos(rad))
        fy = int(self.y + head_len * math.sin(rad))
        cv2.arrowedLine(canvas, (int(self.x), int(self.y)), (fx, fy), (255, 0, 0), 2, tipLength=0.3)

        # If robot is contain a stone just show it on the robot
        if self.picked is not None:
            cv2.circle(canvas, (int(self.x), int(self.y)), 8, self.picked.bgr, -1)
            cv2.circle(canvas, (int(self.x), int(self.y)), 8, (255, 255, 255), 1)

    def move_towards(self, target_x, target_y, speed=4.0, turn_speed=5.0, threshold=15.0):
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.hypot(dx, dy)

        if dist <= threshold:
            return True

        target_theta = math.degrees(math.atan2(dy, dx))
        delta_theta = (target_theta - self.theta + 180.0) % 360.0 - 180.0

        if abs(delta_theta) > turn_speed:
            self.theta += math.copysign(turn_speed, delta_theta)
        else:
            self.theta = target_theta
            rad = math.radians(self.theta)
            self.x += speed * math.cos(rad)
            self.y += speed * math.sin(rad)

        return False

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


    for c in contours:
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

field_img = cv2.imread("data/field.png")
# If image is not found, draw a temporary field
if field_img is None:
    field_img = np.full((600, 800, 3), (220, 220, 220), dtype=np.uint8)

if field_img is not None: 
    rectangle_field = dynamic_warp_perspective(field_img)

    fh, fw = rectangle_field.shape[:2]

    # Simulated drop zone of each color
    drop_zones = {
        "red":   {"pos": (80, 80),             "color": (0, 0, 255)},
        "green": {"pos": (fw - 80, 80),        "color": (0, 220, 0)},
        "blue":  {"pos": (fw - 80, fh - 80),   "color": (255, 120, 0)}
    }

    # Simulated stones positions
    rocks = [
        Rock(1, fw // 2 - 20, fh // 2 - 15, "red",   (0, 0, 255)),
        Rock(2, fw // 2 + 15, fh // 2 + 10, "blue",  (255, 120, 0)),
        Rock(3, fw // 2 - 10, fh // 2 + 25, "green", (0, 220, 0)),
        Rock(4, fw // 2 + 25, fh // 2 - 20, "red",   (0, 0, 255)),
        Rock(5, fw // 2,      fh // 2 + 5,  "blue",  (255, 120, 0)),
    ]

    if mode == RunMode.SIMULATION:
        # If field not founded, create a gray field 600 x 800 px instead
        if rectangle_field is None:
            rectangle_field = np.full((600, 800), (220, 220, 220), 3, np.unit8)

        # Simulation robot setup
        h, w = rectangle_field.shape[:2]
        robot = Robot2D(start_x=w // 2, start_y=h // 2, theta=0, width=20, length=50)

        speed = simulation_config["robot_speed"]
        turn_speed = simulation_config["robot_turn_speed"]

        # Initialize machine state
        state = "SEARCH"
        target_rock = None
        placed_count = {"red": 0, "green": 0, "blue": 0}

        # Simulation loop
        while True:
            canvas = rectangle_field.copy()

            # Draw all drop zones
            for name, data in drop_zones.items():
                pos = data["pos"]
                cv2.circle(canvas, pos, 35, data["color"], 2)
                cv2.putText(canvas, f"ZONE: {name.upper()}", (pos[0] - 35, pos[1] - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, data["color"], 1)

            if state == "SEARCH":
                # Find for available rocks
                waiting_rocks = [r for r in rocks if r.status == RockStatus.WAITING]
                if waiting_rocks:
                    # Find the nearest rock use euclidian distance
                    target_rock = min(waiting_rocks, key=lambda r: math.hypot(r.x - robot.x, r.y - robot.y))
                    state = "GO_TO_ROCK"
                else:
                    cv2.putText(canvas, "ALL ROCKS DELIVERED!", (fw // 2 - 130, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 0), 2)

            elif state == "GO_TO_ROCK":
                reached = robot.move_towards(target_rock.x, target_rock.y)
                if reached:
                    # Collect stone
                    target_rock.status = RockStatus.ON_ROBOT
                    robot.picked = target_rock
                    state = "DELIVER"

            elif state == "DELIVER":
                # Go to target zone (same color with the stone on the robot)
                zone = drop_zones[robot.picked.color_name]
                zx, zy = zone["pos"]
                reached = robot.move_towards(zx, zy)
                if reached:
                    # Place stone down, place it in distributive position
                    count = placed_count[robot.picked.color_name]
                    robot.picked.x = zx + ((count % 3) - 1) * 15
                    robot.picked.y = zy + ((count // 3) - 1) * 15
                    robot.picked.status = RockStatus.DELIVERED
                    placed_count[robot.picked.color_name] += 1
                    
                    robot.picked = None
                    # Back to search next stone
                    state = "SEARCH" 

            # Draw every stone
            for r in rocks:
                r.draw(canvas)

            robot.draw(canvas)

            cv2.imshow("Robot simulation", canvas)

            key = cv2.waitKey(30) & 0xFF
            # Exit simulation
            if key == ord('q') or key == 27:
                break
            # Turn left
            elif key == ord('a'):
                robot.theta -= turn_speed
            # Turn right
            elif key == ord('d'):
                robot.theta += turn_speed
            # Move forward according to robot direction
            elif key == ord('w'):
                rad = math.radians(robot.theta)
                robot.x += speed * math.cos(rad)
                robot.y += speed * math.sin(rad)
            # Move backward
            elif key == ord('s'):  # ถอยหลัง
                rad = math.radians(robot.theta)
                robot.x -= speed * math.cos(rad)
                robot.y -= speed * math.sin(rad)
    
        cv2.destroyAllWindows()

        

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