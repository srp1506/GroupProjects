import cv2
import cv2.aruco as aruco
import numpy as np
import csv
from statistics import mode  
# from qgis.core import QgsVectorLayer, QgsProject

marker_locations = {}
readfile = "GG_3895_task_4b.csv"
writefile = "live_location.csv"
closest_marker = None

robot_coords = [0,0]

def read_file():
    
    with open(readfile, 'r', newline = '') as file:
        
        reader = csv.reader(file)
        next(reader, None)
        
        for row in reader:
            
            if len(row) == 3:
                id, lat, lon = row
                # print(row)
                marker_locations[int(id)] = [float(lat), float(lon)]

def calculate_marker_center(corners):
    corners = np.array(corners)
    avg_x = np.mean(corners[:, 0])
    avg_y = np.mean(corners[:, 1])
    return int(avg_x), int(avg_y)

def calculate_orientation(corners):
    corners = np.array(corners)
    
    top_left = corners[0]
    top_right = corners[1]
    
    
    vector_top = top_right - top_left
    
    
    angle_rad = np.arctan2(vector_top[1], vector_top[0])
    angle_deg = round(np.degrees(angle_rad))
    
    
    angle_deg = (angle_deg + 360) % 360
    
    return angle_deg

def calculate_nearest_aruco_marker(ArUco_details_dict):
    
    closest_aruco_id = None
    min_distance = float('inf')

    for marker_id, details in ArUco_details_dict.items():
        center_coords = details[0]
        # print(robot_coords)
        distance = (center_coords[0] - robot_coords[0])**2 + (center_coords[1] - robot_coords[1])**2
        # distance = np.sqrt(distance_squared)

        if distance < min_distance:
            min_distance = distance
            closest_aruco_id = marker_id

    return closest_aruco_id

    
    
def detect_ArUco_details(image):
    
    ArUco_details_dict = {}
    ArUco_corners = {}
    # robot_coords = []
    alpha = -1.5
    beta= 50
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    
    # Create DetectorParameters directly
    parameters = cv2.aruco.DetectorParameters()

    _, thresholded = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
    corners_tuple, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)
    corners = [element.reshape(-1, 2).tolist() for element in corners_tuple]

    if ids is not None:
        for i in range(len(ids)):
            center_x, center_y = calculate_marker_center(corners[i])
            orientation = calculate_orientation(corners[i])
            marker_id = int(ids[i][0])
            # print(marker_id)
            if marker_id == 6:
                # print(marker_id)
                robot_coords[0] = center_x
                robot_coords[1] = center_y
                break
            ArUco_details_dict[marker_id] = [[center_x, center_y], orientation]
            ArUco_corners[marker_id] = np.array(corners[i], dtype=np.float32)

    return ArUco_details_dict, ArUco_corners, ids

def mark_ArUco_image(image, closest_marker, ids):
    
    # # cv2.polylines(image, [corner.astype(int)], True, (0, 255, 0), 2)
    for marker_id, details in ArUco_details_dict.items():
        if marker_id != 6:
            corners = ArUco_corners[marker_id]
            cv2.polylines(image, [corners.astype(int)], True, (0, 0, 255), 2) 
               
    if closest_marker is not None:
        corner = ArUco_corners[closest_marker]
        cv2.polylines(image, [corner.astype(int)], True, (0, 255, 0), 2)
        cv2.circle(image, robot_coords, 5, (255, 0, 0), -1)
    return image



def write_file(closest_marker):
    
    if closest_marker is not None:
        with open(writefile, 'w', newline='') as file:
            writer = csv.writer(file)
            header = ["lat", "lon"]
            writer.writerow(header)
            writer.writerow((marker_locations[closest_marker][0], marker_locations[closest_marker][1]))
            # reload_delimited_text_layer()
    else:
        print("No ArUco markers detected. Cannot write to file.")

if __name__ == "__main__":

    read_file()
    # with open(writefile, file.'w', newline = '') as file:
        
    #     writer = csv.writer(file)
        
    #     header = ["lat", "lon"]
        
    #     writer.writerow(header)
    #     # print(marker_locations['26'])
    #     # print(marker_locations[closest_marker])
    #     writer.writerow((-74.362853,39.613201))
    # write_file([-74.362853,39.613201])
    cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # cv2.waitKey(4000)
    # cv2.waitKey(3000)
    # closest = []
    # count = 0
    while True:
        ret, frame = cap.read()
        # count = count + 1

        # Detect ArUco markers and get details
        ArUco_details_dict, ArUco_corners, ids = detect_ArUco_details(frame)
        # print("Detected details of ArUco: ", ArUco_details_dict)

        # Display the marked image
        # marked_frame = mark_ArUco_image(frame, ArUco_details_dict, ArUco_corners)
        closest_marker  = calculate_nearest_aruco_marker(ArUco_details_dict)        
        write_file(closest_marker)
        marked_frame = mark_ArUco_image(frame, closest_marker, ids)
        # cv2.waitKey(3000)
        desired_width = 1000
        desired_height = 600
        resized_frame = cv2.resize(marked_frame, (desired_width, desired_height))
        cv2.imshow("Marked Image", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()