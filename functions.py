import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


orientations = ['N', 'E', 'S', 'W']

def rotate_orientation(current_orientation, steps):
    index = orientations.index(current_orientation)
    new_index = (index + steps) % len(orientations)
    return orientations[new_index]

def euclidian_distance(p1, p2):
    """
        Compute euclidean distance

        :param p1: first coordinate tuple
        :param p2: second coordinate tuple
        :return: distance Float
    """

    return math.sqrt((p2[0] - p1[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def edge_distance(e_1, e_2):
    """Compute the size difference between two edges

    Args:
        e_1 (tuple): Matrix of coordinates defining the first edge
        e_2 (tuple): Matrix of coordinates defining the second edge
        
    Returns:
        comparison (float): Difference in length between the two edges
        average (float): The average length of the two edges
    """
    e_1_distance = euclidian_distance(e_1[0], e_1[-1])
    e_2_distance = euclidian_distance(e_2[0], e_2[-1])
    comparison = math.fabs(e_1_distance - e_2_distance)
    average = (e_1_distance + e_2_distance)/2
    
    return comparison, average

def similar_edge_length(e_1, e_2, percent):
    """Checking if the two edges have a similar length

    Args:
        e_1 (tuple): Matrix of coordinates defining the first edge
        e_2 (tuple): Matrix of coordinates defining the second edge
        percent (float): Threshold

    Returns:
        Boolean: True of the difference is less than the threshold
    """
    res, val = edge_distance(e_1, e_2)    
    return res < (val * percent)

def edge_line(p_1, p_2):
    """computes the straight line from one corner to the other of an edge

    Args:
        p_1 (tuple): _description_
        p_2 (tuple): _description_

    Returns:
        float: slope, x(y=0)
    """
    x1,y1 = p_1
    x2,y2 = p_2
    
    a = (y2 - y1)/(x2 - x1)
    b = y1 - a*x1
    
    return a, b

def distance_to_centerline(border, centerline, resolution):
    
    border_length = len(border)
    centerline_length = len(centerline)
    # Compute evenly spaced indexes
    border_idx = np.linspace(0, border_length - 1, resolution).astype(int)
    centerline_idx = np.linspace(0, centerline_length - 1, resolution).astype(int)
    distance_vector = []
    
    for idx, i in enumerate(centerline_idx):
        x = centerline[idx][0]
        relevant_pixels = []
        for idy, j in enumerate(border):
            if j[0] == x:
                relevant_pixels.append(j)
        if len(relevant_pixels) == 1:
            #distance_vector.append(euclidian_distance(centerline[idx], relevant_pixels))
            distance_vector.append(relevant_pixels[1] - centerline[1])
        elif len(relevant_pixels) > 1:
            dist_tuple = []
            for idz, k in enumerate(relevant_pixels):
                #dist_tuple.append(euclidian_distance(centerline[idx], relevant_pixels[k]))
                dist_tuple.append(relevant_pixels[idz][1] - centerline[1])
            distance_vector.append(dist_tuple)
        elif len(relevant_pixels) == 0:
            distance_vector.append(0)
            
    return distance_vector
      
  
def calculate_angle(p1, p2, p3):
    """
    Calculate the angle formed by three points.
    
    :param p1: First point (tuple of x, y coordinates)
    :param p2: Second point (tuple of x, y coordinates)
    :param p3: Third point (tuple of x, y coordinates)
    :return: Angle in degrees
    """
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)
    return angle_deg  
      
def filter_corners(corners):
    '''
    filtered_corners = []
    for i in range(len(corners)):
        p1 = corners[i]
        p2 = corners[(i + 1) % len(corners)]
        p3 = corners[(i + 2) % len(corners)]
        angle = calculate_angle(p1, p2, p3)
        if angle < 170:  # Adjust this threshold as needed
            filtered_corners.append(p2)
    #return filtered_corners  
    '''
    
def CornerDetection(imagename, drawCircle=False, crop_strategy=False, gopro = False, test = False):
    input = imagename.copy()
    #blur = cv2.GaussianBlur(input, (7, 7), cv2.BORDER_DEFAULT)
    #gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    h, w = input.shape
    orig_y = h / 2
    orig_x = w / 2
    if crop_strategy == False:
        if gopro or test:
            corners = cv2.goodFeaturesToTrack(input, 1000, 0.001, 1, useHarrisDetector=False, k=0.04)
        else:
            corners = cv2.goodFeaturesToTrack(input, 100, 0.005, 4, useHarrisDetector=False, k=0.04) # orginalt: cv2.goodFeaturesToTrack(input, 4, 0.01, 300)
        #print(f'Cornerdetection corners: {corners}')
    else:
        corners = cv2.goodFeaturesToTrack(input, 100, 0.005, 300, useHarrisDetector=False, k=0.04)
    
    corners = np.int0(corners)
    #corners = cv2.cornerHarris(input, 2, 5, 0.07) 

    cnr = []
    # res = []
    for i in corners:
        x, y = i.ravel()
        if drawCircle == True:
            cv2.circle(input, (x, y), 50, (0, 255, 0), 5)
        cnr.append([x, y])

    n_cnr = [[coord[0] - orig_x, orig_y - coord[1]] for coord in cnr]

    quad1, quad2, quad3, quad4 = [], [], [], []
    for coord in n_cnr:
        if coord[0] >= 0 and coord[1] >= 0:
            quad1.append(coord)
        elif coord[0] < 0 and coord[1] >= 0:
            quad2.append(coord)
        elif coord[0] < 0 and coord[1] < 0:
            quad3.append(coord)
        else:
            quad4.append(coord)

    quad1 = sorted(quad1, key=lambda c: c[0])
    quad2 = sorted(quad2, key=lambda c: c[0], reverse=True)
    quad3 = sorted(quad3, key=lambda c: c[0], reverse=True)
    quad4 = sorted(quad4, key=lambda c: c[0])

    res = quad2 + quad1 + quad3 + quad4

    res = [[int(coord[0] + orig_x), int(- coord[1] + orig_y)] for coord in res]

    return input, res


def filter_4_corner_points(points, gopro=False, test = False):
    # Convert points to numpy array for easier manipulation
    points_array = np.array(points)
    
    # Find the bounding box of all points
    min_x = np.min(points_array[:, 0])
    max_x = np.max(points_array[:, 0])
    min_y = np.min(points_array[:, 1])
    max_y = np.max(points_array[:, 1])
    if gopro:
        points_array = np.array([x for x in points_array if 200 < x[0] < 1250 and 150 < x[1] < 1200])
        #print(points_array)
    if test:
        points_array = np.array([x for x in points_array if 250 < x[0] < 2000 and 300 < x[1] < 2200])
        #print(points_array)
    # Calculate distances from each corner
    # May have to swap right and left for these:
    distances = {
        'bottom_right': np.sqrt((points_array[:, 0] - min_x) ** 2 + (points_array[:, 1] - min_y) ** 2),
        'bottom_left': np.sqrt((points_array[:, 0] - max_x) ** 2 + (points_array[:, 1] - min_y) ** 2),
        'top_right': np.sqrt((points_array[:, 0] - min_x) ** 2 + (points_array[:, 1] - max_y) ** 2),
        'top_left': np.sqrt((points_array[:, 0] - max_x) ** 2 + (points_array[:, 1] - max_y) ** 2)
    }
    
    # Find the indices of the maximum distance for each corner
    corner_indices = {
        corner: distances[corner].argmax() for corner in distances
    }
    
    # Extract the corner points
    corner_points = {
        corner: points_array[index].tolist() for corner, index in corner_indices.items()
    }
    print(f'filter_corners corners: {corner_points}')
    return corner_points



def extract_edges_from_contour(contours, corner_points):
    edge_regions = []

    # Sort the corner points to define the edges
    sorted_corners = [corner_points['bottom_left'], corner_points['bottom_right'], corner_points['top_right'], corner_points['top_left']]

    # Iterate over the edges defined by corner points
    for i in range(len(sorted_corners)):
        # Get the current and next corner points
        start_point = sorted_corners[i]
        end_point = sorted_corners[(i + 1) % len(sorted_corners)]

        # Create a black mask image
        mask = np.zeros((len(contours[0]), len(contours[0][0]), 1), dtype=np.uint8)

        # Find the points on the contour between the current and next corner points
        points_on_contour = []
        for contour in contours:
            for point in contour:
                point_x, point_y = point[0]
                if start_point[0] <= point_x <= end_point[0] and start_point[1] <= point_y <= end_point[1]:
                    points_on_contour.append((point_x, point_y))

        # Draw the contour region on the mask if points were found
        if points_on_contour:
            cv2.drawContours(mask, [np.array(points_on_contour)], -1, (255), cv2.FILLED)

            # Extract the edge region using the mask
            edge_region = cv2.bitwise_and(mask, mask, mask=mask)

            edge_regions.append(edge_region)

    return edge_regions

def edgeFeatureExtraction(img):
    """
    Function that finds the edge contours in an image and
    returns a cropped image with only the dominant edge
    
    :param img: Binary image of cropped side piece
    
    :return: image containing only the longest edge. 
    
    """
    
    edge = cv2.Canny(img,threshold1=200,threshold2=700)
    contours,_ = cv2.findContours(edge,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if (len(contours)!=0):
        top_contour = [len(i) for i in contours]
        index = np.argmax(np.array(top_contour))
        for i in range(len(contours)):
            if i == index:
                pass
            else:
                cv2.boundingRect(contours[i])       
                x,y,w,h = cv2.boundingRect(contours[i])
                img[y:y+h,x:x+w] = 255

        return img
    else:
        return img
 

def rotationAboutFixedAxis(img,pointOfRotation,secondPoint):
    """
    Calculates the angle of rotation and rotates the image so that the two points lie on the same y-axis.
    
    :param img: Image
    :param pointOfRotation: Specify point of rotation in tuple(int(x),int(y)) coordinates.
    :param secondPoint: Point to be rotated (x,y)
    
    :return: Rotated image
    
    """
    angle = math.atan((secondPoint[1]-pointOfRotation[1])/(secondPoint[0]-pointOfRotation[0]))*180/math.pi
    M = cv2.getRotationMatrix2D(pointOfRotation,angle,1)
    dst = cv2.warpAffine(img,M,(0,0))
    
    return dst

 
def BinaryMask(imgname):
    input = imgname.copy()
    thresh = 100
    maxValue = 255
    th, dst = cv2.threshold(input, thresh, maxValue, cv2.THRESH_OTSU)

    return dst 
    
    

def rotate_image_to_same_y(image, point1, point2, rotate_points=False, show_process = False):
    # Calculate the angle needed to rotate the image
    dy = point2[1] - point1[1]
    dx = point2[0] - point1[0]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    #rotation_angle = -angle_deg  # Negative angle to rotate in the opposite direction
    rotation_angle = angle_deg
    if show_process:
        print(f'angle_rad: {angle_rad}')
        print(f'Rotation angle: {rotation_angle}')
    # Ensure rotation angle is within the range of -90 to 90 degrees
    if rotation_angle < -90:
        rotation_angle += 180
    elif rotation_angle > 90:
        rotation_angle -= 180
    
    # Get image dimensions
    height, width = image.shape[:2]
    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_angle, 1)
    # Apply the rotation to the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    if show_process:
        show_resized_image(image, 'Image before rotation to level', (500, 500))
        show_resized_image(rotated_image, 'Image after leveling', (500, 500))
    if rotate_points:
        if len(point1) == 2:
            corner_1 = np.dot(rotation_matrix, np.append(point1, 1)).astype(int)[:2]
            corner_2 = np.dot(rotation_matrix, np.append(point2, 1)).astype(int)[:2]  
        else:
            corner_1 = np.dot(rotation_matrix, point1).astype(int)[:2]
            corner_2 = np.dot(rotation_matrix, point2).astype(int)[:2]             
        return rotated_image, corner_1, corner_2      
    else:
        return rotated_image

def find_corners_on_edge(edge_image):
    # Get the height of the image
    height = edge_image.shape[0]
    # Get the width of the image
    width = edge_image.shape[1]
    # Initialize variables to store the transition points
    left_point = None
    right_point = None

    # Iterate through the first column
    for row in range(height):
        # Check for the transition from black (0) to white (255)
        if edge_image[row, 0] == 0 and edge_image[row+1, 0] == 255:
            left_point = (0, row)
            break  # Stop checking once the transition point is found
        # Check for the transition from white (255) to black (0)
        elif edge_image[row, 0] == 255 and edge_image[row+1, 0] == 0:
            left_point = (0, row)
            break  # Stop checking once the transition point is found

    # Iterate through the last column
    for row in range(height):
        # Check for the transition from black (0) to white (255)
        if edge_image[row, width - 1] == 0 and edge_image[row+1, width - 1] == 255:
            right_point = (width - 1, row)
            break  # Stop checking once the transition point is found
        # Check for the transition from white (255) to black (0)
        elif edge_image[row, width - 1] == 255 and edge_image[row+1, width - 1] == 0:
            right_point = (width - 1, row)
            break  # Stop checking once the transition point is found

    return left_point, right_point


def calculate_line_parameters(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    
    # Calculate the slope (m) of the line
    if x2 - x1 != 0:  # Avoid division by zero
        slope = (y2 - y1) / (x2 - x1)
    else:
        slope = float('inf')  # Vertical line
    
    # Calculate the y-intercept (b) of the line
    intercept = y1 - slope * x1
    
    return slope, intercept

def draw_line_from_parameters(image, slope, intercept, color=(0, 0, 255), thickness=2):
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate endpoints of the line based on image boundaries
    x1 = 0
    y1 = int(intercept)
    x2 = width - 1
    y2 = int(slope * x2 + intercept)
    
    # Draw the line on the image
    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    
    
def calculate_vertical_distances_2(edge_image, centerline_slope, centerline_intercept, resolution, crop_strat=False, corner_1=0, corner_2=0):
    # Get image dimensions
    height, width = edge_image.shape[:2]
    
    # Calculate the centerline points
    if crop_strat == True:
        x_coordinates = np.linspace(0, width - 1, resolution)
    else:
        x_coordinates = np.linspace(corner_1[0], corner_2[0], resolution)
    
    # Initialize list to store the values
    vertical_distances = []
    
    for x in x_coordinates:
        # Calculate corresponding y coordinate on the centerline
        centerline_y = int(centerline_slope * x + centerline_intercept)
        if 0 <= centerline_y < height:
            edge_y = centerline_y
            edge_y_2 = centerline_y
            second_edge = 0
            if edge_image[centerline_y, int(x)] == 0:
                while edge_image[edge_y, int(x)] == 0 and edge_y > 0:
                    edge_y -= 1
                # check if the point is also under an ear
                while edge_image[edge_y_2, int(x)] == 0 and edge_y_2 < centerline_y + 100: # kan bruke height istedenfor centerline
                    edge_y_2 +=1
                if edge_y_2 < centerline_y + 99:
                    second_edge = edge_y_2
            else:
                while edge_image[edge_y, int(x)] == 255 and edge_y < height -1:
                    edge_y += 1
                # check if the point is also under an ear
                while edge_image[edge_y_2, int(x)] == 255 and edge_y_2 > 0:
                    edge_y_2 -= 1
                if edge_y_2 > 2:
                    second_edge = edge_y_2
            if second_edge == 0:
                vertical_distance = edge_y - centerline_y
                vertical_distances.append(vertical_distance)
            else:
                #print('second edge')
                vertical_distance = edge_y - centerline_y
                vertical_distance_2 = second_edge - centerline_y
                vertical_distances.append((vertical_distance, vertical_distance_2))
        else:
            vertical_distances.append(None)
    return vertical_distances
   
   
def calculate_vertical_distances(edge_image, centerline_slope, centerline_intercept, resolution, crop_strat=False, corner_1=0, corner_2=0):
    # Get image dimensions
    height, width = edge_image.shape[:2]
    
    # Calculate the centerline points
    if crop_strat == True:
        x_coordinates = np.linspace(0, width - 1, resolution)
    else:
        x_coordinates = np.linspace(corner_1[0], corner_2[0], resolution)
    
    # Initialize list to store the values
    vertical_distances = []
    
    for x in x_coordinates:
        # Calculate corresponding y coordinate on the centerline
        centerline_y = int(centerline_slope * x + centerline_intercept)
        if 0 <= centerline_y < height:
            edge_y = centerline_y
            edge_y_2 = centerline_y
            edge_y_3 = centerline_y
            second_edge = 0
            third_edge = 0
            if edge_image[centerline_y, int(x)] == 0:
                while edge_image[edge_y, int(x)] == 0 and edge_y > 0:
                    edge_y -= 1
                # check if the point is also under an ear
                while edge_image[edge_y_2, int(x)] == 0 and edge_y_2 < centerline_y + 100: # kan bruke height istedenfor centerline
                    edge_y_2 +=1
                if edge_y_2 < centerline_y + 99:
                    second_edge = edge_y_2
                    edge_y_3 = edge_y_2
                    while edge_image[edge_y_3, int(x)] == 255 and edge_y_3 < edge_y_2 +150:
                        edge_y_3 += 1
                    if edge_y_3 < edge_y_2 + 148:
                        third_edge = edge_y_3
                    
            else:
                while edge_image[edge_y, int(x)] == 255 and edge_y < height -1:
                    edge_y += 1
                # check if the point is also under an ear
                while edge_image[edge_y_2, int(x)] == 255 and edge_y_2 > 0:
                    edge_y_2 -= 1
                if edge_y_2 > 2:
                    second_edge = edge_y_2
                    edge_y_3 = edge_y_2
                    while edge_image[edge_y_3, int(x)] == 0 and edge_y_3 > 2:
                        edge_y_3 -= 1
                    if edge_y_3 > 3:
                        third_edge = edge_y_3
                    
            if second_edge == 0:
                vertical_distance = edge_y - centerline_y
                vertical_distances.append(vertical_distance)
            else:
                if third_edge == 0:
                    #print('second edge')
                    vertical_distance = edge_y - centerline_y
                    vertical_distance_2 = second_edge - centerline_y
                    vertical_distances.append((vertical_distance, vertical_distance_2))
                else:
                    vertical_distance = edge_y - centerline_y
                    vertical_distance_2 = second_edge - centerline_y
                    vertical_distance_3 = third_edge - centerline_y
                    vertical_distances.append((vertical_distance, vertical_distance_2, vertical_distance_3))
                                        
        else:
            vertical_distances.append(None)
    return vertical_distances   
   
    
def root_mean_square_difference(array1, array2):
    diff_sum = 0
    
    for i, j in zip(array1, array2):
        i = np.array(i) if not isinstance(i, np.ndarray) else i
        j = np.array(j) if not isinstance(j, np.ndarray) else j
        
        
        dim_1 = i.ndim
        dim_2 = j.ndim
        
        if dim_1 == 1 and dim_2 == 1:
            diff_sum += (i[0] - j[0]) ** 2
        
        elif dim_1 == 2 and dim_2 == 2:
            # kan ver det e i[0] + j[1] og i[1] + j[0]
            diff_sum += ((i[0] - j[0]) ** 2 + (i[1] - j[1]) ** 2) / 2
        
        elif dim_1 == 2 and dim_2 == 1:
            # kanskje i[1]
            diff_sum += (i[0] - j) ** 2
            
        elif dim_1 == 1 and dim_2 == 2:
            # kanskje j[1]
            diff_sum += (i - j[0]) ** 2
            
    return np.sqrt(diff_sum)



def root_mean_square_difference_3(array1, array2):
    diff_sum = 0
    
    for i, j in zip(array1, array2):
        i = np.array(i) if not isinstance(i, np.ndarray) else i
        j = np.array(j) if not isinstance(j, np.ndarray) else j
        
        # flatten
        i = i.flatten()
        j = j.flatten()
        
        # Calculate the squared difference between elements
        #if i.ndim == 1 and j.ndim == 1:
        if i.shape == (1,) and j.shape == (1,):
            squared_diff = (i + j) ** 2
            diff_sum += squared_diff
        
        #elif i.ndim == 2 and j.ndim == 2:
        elif i.shape == (2,) and j.shape == (2,):
            diff_sum += (i[0] + j[0]) ** 2
            diff_sum += (i[1] + j[1]) ** 2
        
        #elif i.ndim == 2 and j.ndim == 1:
        elif i.shape == (2,) and j.shape == (1,):
            diff_sum += (i[0] + j) ** 2
        
        #elif i.ndim == 1 and j.ndim == 2:
        elif i.shape == (1,) and j.shape == (2,):
            diff_sum += (i + j[0]) ** 2
        
        # Accumulate the squared differences
        #diff_sum += np.sum(squared_diff)
            
    # Compute the mean squared difference
    mean_squared_diff = diff_sum / len(array1)
    
    # Compute the root mean square difference
    root_mean_square_diff = np.sqrt(mean_squared_diff)
    #print(f'rmse in function: {root_mean_square_diff}')
    return root_mean_square_diff

def root_mean_square_difference_2(array1, array2):
    diff_sum = 0
    
    for i, j in zip(array1, array2):
        i = np.array(i) if not isinstance(i, np.ndarray) else i
        j = np.array(j) if not isinstance(j, np.ndarray) else j
        
        i = i.flatten()
        j = j.flatten()
        
        if i.shape == (1,):
            if j.shape == (1,):
                diff_sum += (i + j) ** 2 
            else:
                diff_sum += (i + j[0]) ** 2 
                
        if i.shape == (2,):
            if j.shape == (1,):
                diff_sum += (i[0] + j) ** 2
            else:
                diff_sum += (i[0] + j[0]) ** 2
                diff_sum += (i[1] + j[1]) ** 2  
                
        if i.shape == (3,):
            if j.shape == (1,):
                diff_sum += (i[0] + j) ** 2
            elif j.shape == (2,):
                diff_sum += (i[0] + j[0]) ** 2
                diff_sum += (i[1] + j[1]) ** 2   
            else:
                diff_sum += (i[0] + j[0]) ** 2
                diff_sum += (i[1] + j[1]) ** 2                   
                diff_sum += (i[2] + j[2]) ** 2                
            
    # Compute the mean squared difference
    mean_squared_diff = diff_sum / len(array1)
    
    # Compute the root mean square difference
    root_mean_square_diff = np.sqrt(mean_squared_diff)
    #print(f'rmse in function: {root_mean_square_diff}')
    return root_mean_square_diff
        
def show_resized_image(img, title='RESIZED IMAGE', resolution=(800, 800)):
    img_r = cv2.resize(img, resolution)
    cv2.imshow(title, img_r)  
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
       