import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from functions import *

class Puzzle():
    
    def __init__(self, gopro = False, test= False):
        self.pieces = []
        self.corner_pieces = []
        self.perimeter_pieces = []
        self.inner_pieces = []
        self.unmatched_edges = []
        self.grid = None
        self.gopro = gopro
        self.test=test
        
    def load_piece(self, img, show_corners = False, show_edges = False, test = False):
        piece = Puzzle_piece(img, show_corners=show_corners, show_edges=show_edges, test = test)
        piece.determine_piece_type()
        if piece.type == 'corner':
            self.corner_pieces.append(piece)
        elif piece.type == 'perimeter':
            self.perimeter_pieces.append(piece)
        elif piece.type == 'inner':
            self.inner_pieces.append(piece)
        print(piece.type)
        self.pieces.append(piece)
        for key, i in piece.edges.items():
            if i.type == 'inner':
                self.unmatched_edges.append(i)
                
    def construct_grid(self):
        self.grid_size = int(math.sqrt(len(self.pieces)))
        self.grid = [[None] * self.grid_size for _ in range(self.grid_size)]
        self.grid_obj = [[None] * self.grid_size for _ in range(self.grid_size)]
        self.perimeter_pieces_per_side = self.grid_size - 2
    
    def add_piece_to_grid(self, row, col, piece):
        self.grid[row][col] = piece.id
        self.grid_obj[row][col] = piece
        
    def fill_grid(self):
        self.grid_fill_perimeter(self)
        self.grid_fill_inner(self)
        
    def grid_fill_perimeter(self):
        # start with upper left corner
        corner_1 = self.corner_pieces[0]
        self.add_piece_to_grid(0, 0, self.corner_pieces[0])
        self.corner_pieces.remove(self.corner_pieces[0])
        while corner_1.edges['N'].type != 'perimeter' or corner_1.edges['W'].type != 'perimeter':
            corner_1.rotate_90()
        self.display_grid()
        
        # fill out the row
        cur_edge = corner_1.edges['E']
        for i in range(self.perimeter_pieces_per_side):
            rmse_v = np.array([])
            #possible = []
            for j in self.perimeter_pieces:
                for key, k in j.edges.items():
                    if k.type == 'perimeter':
                        dir = rotate_orientation(k.direction, -1)
                rmse_v = np.append(rmse_v, root_mean_square_difference_2(cur_edge.distances, j.edges[dir].distances_flipped)) 
                #possible.append(j.edges[dir]) 
            matching = self.perimeter_pieces[np.argmin(rmse_v)] 
            self.perimeter_pieces.remove(matching)
            while matching.edges['N'].type != 'perimeter':
                print(f'befor rot: {matching.edges["N"].direction}')
                matching.rotate_90()
                print(f'after rot: {matching.edges["N"].direction}')
            self.add_piece_to_grid(0, i+1, matching)
            self.display_grid() 
            cur_edge = matching.edges['E']
        
        # filled out top, find top right corner
        rmse_v = np.array([])
        for idx, i in enumerate(self.corner_pieces):
            
            #for key, edge in i.edges.items():
            #    edge.show_edge(title = f'id: {i.id} dir: {edge.direction}, type: {edge.type}')
            
            while i.edges['N'].type != 'perimeter' or i.edges['E'].type != 'perimeter':   
                i.rotate_90()
            #    i.show_piece(title = 'rotation skal være top right corner')
            #i.show_piece(title = 'Skal nå ha blitt gjort om til top right corner')
            
            #for key, edge in i.edges.items():
            #    edge.show_edge(title = f'NAA ETTER ROTATION: id: {i.id} dir: {edge.direction}, type: {edge.type}')
            
            rmse_v = np.append(rmse_v, root_mean_square_difference_2(cur_edge.distances, i.edges['W'].distances_flipped))
        matching = self.corner_pieces[np.argmin(rmse_v)]    
        self.corner_pieces.remove(matching)
        self.add_piece_to_grid(0, self.grid_size-1, matching)
        self.display_grid() 
        
        # now fill the right perimeter
        cur_edge = matching.edges['S']
        for i in range(self.perimeter_pieces_per_side):
            rmse_v = np.array([])        
            for j in self.perimeter_pieces:
                for key, k in j.edges.items():
                    if k.type == 'perimeter':
                        dir = rotate_orientation(k.direction, -1)
                rmse_v = np.append(rmse_v, root_mean_square_difference_2(cur_edge.distances, j.edges[dir].distances_flipped)) 
            matching = self.perimeter_pieces[np.argmin(rmse_v)] 
            self.perimeter_pieces.remove(matching)
            while matching.edges['E'].type != 'perimeter':
                matching.rotate_90()
            self.add_piece_to_grid(i+1, self.grid_size-1, matching)
            self.display_grid() 
            cur_edge = matching.edges['S']
        
        # now find the bottom right corner
        rmse_v = np.array([])
        for i in self.corner_pieces:   
            while i.edges['E'].type != 'perimeter' or i.edges['S'].type != 'perimeter':   
                i.rotate_90()
            rmse_v = np.append(rmse_v, root_mean_square_difference_2(cur_edge.distances, i.edges['N'].distances_flipped))
        matching = self.corner_pieces[np.argmin(rmse_v)]    
        self.corner_pieces.remove(matching)
        self.add_piece_to_grid(self.grid_size-1, self.grid_size-1, matching)
        self.display_grid()   
        
        # now fill out the left perimeter
        cur_edge = self.grid_obj[0][0].edges['S']      
        for i in range(self.perimeter_pieces_per_side):
            rmse_v = np.array([])        
            for j in self.perimeter_pieces:
                for key, k in j.edges.items():
                    if k.type == 'perimeter':
                        dir = rotate_orientation(k.direction, 1)
                rmse_v = np.append(rmse_v, root_mean_square_difference_2(cur_edge.distances, j.edges[dir].distances_flipped)) 
            matching = self.perimeter_pieces[np.argmin(rmse_v)] 
            self.perimeter_pieces.remove(matching)
            while matching.edges['W'].type != 'perimeter':
                matching.rotate_90()
            self.add_piece_to_grid(1+i, 0, matching)
            self.display_grid() 
            cur_edge = matching.edges['S']
            
        # we now know the bottom left corner
        matching = self.corner_pieces[0]    
        self.corner_pieces.remove(matching)
        while matching.edges['W'].type != 'perimeter' or matching.edges['S'].type != 'perimeter':   
            matching.rotate_90()
        self.add_piece_to_grid(self.grid_size-1, 0, matching)
        cur_edge = matching.edges['E']
        self.display_grid()   
        
        # now fill out the bottom perimeter          
        for i in range(self.perimeter_pieces_per_side):
            rmse_v = np.array([])        
            for j in self.perimeter_pieces:
                for key, k in j.edges.items():
                    if k.type == 'perimeter':
                        dir = rotate_orientation(k.direction, 1)
                rmse_v = np.append(rmse_v, root_mean_square_difference_2(cur_edge.distances, j.edges[dir].distances_flipped)) 
            matching = self.perimeter_pieces[np.argmin(rmse_v)] 
            self.perimeter_pieces.remove(matching)
            while matching.edges['S'].type != 'perimeter':
                matching.rotate_90()
            self.add_piece_to_grid(self.grid_size-1, 1+i, matching)
            self.display_grid() 


    def grid_fill_inner(self):
        cur_edge = self.grid_obj[1][0].edges['E']
        cur_edge_above = self.grid_obj[0][1].edges['S']
        for i in range(self.perimeter_pieces_per_side):
        # for every row
            for j in range(self.perimeter_pieces_per_side):
            # for every column
                rmse_v = np.array([])
                keys = []
                for k in self.inner_pieces:
                    for key, l in k.edges.items():
                        #rmse_v = np.append(rmse_v, root_mean_square_difference_2(cur_edge.distances, l.distances_flipped))
                        rmse_v_east_and_north = 0
                        rmse_v_east_and_north += root_mean_square_difference_2(cur_edge.distances, l.distances_flipped)
                        print(f'direction before: {key}')
                        above_dir = rotate_orientation(key, 1)
                        print(f'direction after: {above_dir}')
                        rmse_v_east_and_north += root_mean_square_difference_2(cur_edge_above.distances, k.edges[above_dir].distances_flipped)
                        rmse_v = np.append(rmse_v, rmse_v_east_and_north)
                        keys.append(key)
                matching = self.inner_pieces[math.floor(np.argmin(rmse_v)/4)] 
                #print(f'matching inner id: {matching.id}')
                dir = keys[np.argmin(rmse_v)]
                self.inner_pieces.remove(matching)
                if dir == 'N':
                    matching.rotate_90()
                elif dir == 'E':
                    matching.rotate_90()
                    matching.rotate_90()
                elif dir == 'S':
                    matching.rotate_90()
                    matching.rotate_90()
                    matching.rotate_90()
                self.add_piece_to_grid(i+1, j+1, matching)
                self.display_grid()
                cur_edge = matching.edges['E']   
                cur_edge_above = self.grid_obj[i][j+1].edges['S']
                

        
    def display_grid(self):
        for row in self.grid:
            print(row)
        print('')
        
    def find_matching_edge_rough(self, edge):
        '''
        Compares an edge to every unmatched edge left
        
        :param edge: edge to match
        
        :return match: matching edge
        '''
        self.unmatched_edges.remove(edge)
        rmse_v = np.array([])
        for idx, i in enumerate(self.unmatched_edges):
            rmse_v = np.append(rmse_v, root_mean_square_difference_2(edge.distances, i.distances_flipped))
        matching = self.unmatched_edges[np.argmin(rmse_v)]
        self.unmatched_edges.remove(matching)
        matching.connected = True
        matching.match = edge
        edge.connected = True
        edge.match = matching
        print(f'rmse_v = {rmse_v}')
        edge.parent_piece.show_piece(title = f'Candidate Piece')
        edge.show_edge(title = f'Candidate edge: {edge.direction}')
        matching.parent_piece.show_piece(title = 'Matching piece')
        matching.show_edge(title = f'Matching edge: {matching.direction}')
        
        
        
    def find_cornerpiece_match(self, piece):
        possible_matches = []
        if piece.edges['N'].type == 'perimeter' and piece.edges['E'].type == 'inner':
            candidate = piece.edges['E']
            for i in self.perimeter_pieces:
                dir = rotate_orientation(i.perimeter_direction, -1)
                if i.edges[dir].connected == False:
                    #print(f'perimeter direction = {i.perimeter_direction}')
                    #print(f'rotated direction = {dir}')
                    possible_matches.append(i.edges[dir])
                    #i.edges[dir].show_edge()
            
                    
                
        rmse_v = np.array([])
        for idx, i in enumerate(possible_matches):
            rmse_v = np.append(rmse_v, root_mean_square_difference_2(candidate.distances, i.distances_flipped))
            #print(root_mean_square_difference_2(candidate.distances, i.distances_flipped))
            #print(f'edge {idx} distances = {i.distances_flipped}')
            #i.show_edge(title = f'edge {idx}')
        #print(f'rmse_v = {rmse_v}')
        matching = possible_matches[np.argmin(rmse_v)]
        matching.connected = True
        matching.match = candidate
        candidate.connected = True
        candidate.match = matching
        matching.parent_piece.show_piece(title = 'Matching piece')
        candidate.show_edge(title = 'Candidate edge')
        matching.show_edge(title= 'Matching edge')
        #print(f'canditade: {candidate.distances}')
        #print(f'matching: {matching.distances}')
           
    def display_solved_puzzle(self):
        grid = []
        for i in self.grid_obj:
            grid_inner = []
            for j in i:
                img = j.opened_img
                #img = cv2.resize(img, dsize = (0,0), fx = 0.5, fy = 0.5)
                img = cv2.resize(img, (300, 300))
                grid_inner.append(img)
            grid.append(grid_inner)      
        self.solved_image = cv2.vconcat([cv2.hconcat(list_h) for list_h in grid])  
        self.solved_image = cv2.resize(self.solved_image, (600, 600))   
        cv2.imshow('Puzzle layout', self.solved_image) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

class Puzzle_piece():
    count = 0
    
    def __init__(self, img, show_corners=False, gopro = False, show_edges = False, test=False):
        self.id = Puzzle_piece.count
        Puzzle_piece.count += 1
        self.edges = {}
        self.gopro = gopro
        self.test = test
        self.show_edges = show_edges
        self.img = img
        self.type = None
        self.perimeter_direction = None
        self.in_grid = False
        img_blur = cv2.GaussianBlur(img, (7, 7), cv2.BORDER_DEFAULT)
        gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] 
        kernel = np.ones((3, 3), np.uint8) 
        closing = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel, iterations=1) 
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=1) 
        self.opened_img = opening
        if gopro:
            # gopro cropping
            #print(f'dims: {self.opened_img.shape}')
            self.opened_img = self.opened_img[500:1900,1200:2700] #600:1700,1300:2500
            show_resized_image(self.opened_img, 'cropped', (400,400))
            #cv2.imshow('cropped', self.opened_img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
        edged = cv2.Canny(self.opened_img, 30, 200) 
        contours, hierarchy = cv2.findContours(edged,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        if self.gopro:
            cornerdetection, corners = CornerDetection(self.opened_img, drawCircle=True, gopro=True)
            self.corners = filter_4_corner_points(corners, gopro=True)
        elif self.test:
            cornerdetection, corners = CornerDetection(self.opened_img, drawCircle=True, test = True)
            self.corners = filter_4_corner_points(corners, test=True)             
        else:
            cornerdetection, corners = CornerDetection(self.opened_img, drawCircle=True)
            self.corners = filter_4_corner_points(corners)
        if show_corners:
            show_resized_image(cornerdetection, 'All corners', (500, 500))
        #self.corners = filter_4_corner_points(corners)
        
        if show_corners == True:
            if self.gopro:
                img_copy = self.opened_img.copy()
            else:
                img_copy = img.copy()
            for corner, point in self.corners.items():
                cv2.circle(img_copy, tuple(point), 25, (0, 0, 255), 5)
            corner_image_r = cv2.resize(img_copy, (400, 400))
            cv2.imshow('corners', corner_image_r)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            
        # get the north edge
        north_rot, north_corner_1, north_corner_2 = rotate_image_to_same_y(self.opened_img, self.corners['top_left'], self.corners['top_right'], rotate_points=True)
        if self.gopro:
            north_crop = north_rot[0:north_corner_1[1]+300, north_corner_1[0]:north_corner_2[0]]
        else:
            north_crop = north_rot[0:north_corner_1[1]+300, north_corner_1[0]:north_corner_2[0]]
        #self.edges['N'] = Edge(north_crop, 'N', corners['top_left'], corners['top_right'])
        
        if self.show_edges:
            show_resized_image(north_crop, 'North edge')
        
        self.edges['N'] = Edge(north_crop, 'N', north_corner_1, north_corner_2, self, gopro=self.gopro, test = self.test) 
            
            
            
        # get the east edge
        angle = 90
        if self.gopro:
           rotation_center = (self.opened_img.shape[1] // 2, self.opened_img.shape[0] // 2) 
        else:
            rotation_center = (img.shape[1] // 2, img.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center, angle, 1.0)
        img_east_edge = cv2.warpAffine(self.opened_img, rotation_matrix, (self.opened_img.shape[1], self.opened_img.shape[0]))
        east_corner_1 = np.dot(rotation_matrix, np.append(self.corners['top_right'], 1)).astype(int)[:2]
        east_corner_2 = np.dot(rotation_matrix, np.append(self.corners['bottom_right'], 1)).astype(int)[:2]
        east_rot, east_corner_1, east_corner_2 = rotate_image_to_same_y(img_east_edge, east_corner_1, east_corner_2, rotate_points=True)
        east_crop = east_rot[0:east_corner_1[1]+300, east_corner_1[0]:east_corner_2[0]] # img_east_edge
        if self.show_edges:
            show_resized_image(east_crop, 'East edge')
        
        # MÅ NOK ROTERA PUNKTENE OG !!!!!!!!!!!!!!!!!!!!!!
        self.edges['E'] = Edge(east_crop, 'E', east_corner_1, east_corner_2, self, gopro=self.gopro, test = self.test)
        
        # get the south side
        angle = 180
        if self.gopro:
            rotation_center = (self.opened_img.shape[1] // 2, self.opened_img.shape[0] // 2)
        else:
            rotation_center = (img.shape[1] // 2, img.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center, angle, 1.0)
        img_south_edge = cv2.warpAffine(self.opened_img, rotation_matrix, (self.opened_img.shape[1], self.opened_img.shape[0]))
        south_corner_1 = np.dot(rotation_matrix, np.append(self.corners['bottom_right'], 1)).astype(int)[:2]
        south_corner_2 = np.dot(rotation_matrix, np.append(self.corners['bottom_left'], 1)).astype(int)[:2]
        south_rot, south_corner_1, south_corner_2 = rotate_image_to_same_y(img_south_edge, south_corner_1, south_corner_2, rotate_points=True)
        south_crop = south_rot[0:south_corner_1[1]+300, south_corner_1[0]:south_corner_2[0]]
        if self.show_edges:
            show_resized_image(south_crop, 'South edge')
        
        # ROTER PUNKTENE RETT FØR CROP
        self.edges['S'] = Edge(south_crop, 'S', south_corner_1, south_corner_2, self, gopro=self.gopro, test = self.test)
        
        # get the west edge
        angle = 270
        rotation_center = (self.opened_img.shape[1] // 2, self.opened_img.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center, angle, 1.0)
        img_west_edge = cv2.warpAffine(self.opened_img, rotation_matrix, (self.opened_img.shape[1], self.opened_img.shape[0]))
        west_corner_1 = np.dot(rotation_matrix, np.append(self.corners['bottom_left'], 1)).astype(int)[:2]
        west_corner_2 = np.dot(rotation_matrix, np.append(self.corners['top_left'], 1)).astype(int)[:2]
        west_rot, west_corner_1, west_corner_2 = rotate_image_to_same_y(img_west_edge, west_corner_1, west_corner_2, rotate_points=True)
        west_crop = west_rot[0:west_corner_1[1]+300, west_corner_1[0]:west_corner_2[0]]
        if self.show_edges:
            show_resized_image(west_crop, 'West edge')
        # ROTER PUNKTENE RETT FØR CROP
        self.edges['W'] = Edge(west_crop, 'W', west_corner_1, west_corner_2, self, gopro=self.gopro, test = self.test)
        

    def show_piece(self, title='Puzzle piece'):
        show_resized_image(self.opened_img, title=title)
    
    
    def determine_piece_type(self):
        perimeters = 0
        for dir, edge in self.edges.items():
            if edge.type == 'perimeter':
                perimeters += 1
                self.perimeter_direction = edge.direction
        if perimeters == 2:
            self.type = 'corner'
        elif perimeters == 1:
            self.type = 'perimeter'
        elif perimeters == 0:
            self.type = 'inner'
        else:
            print('Determine piece type gave to many perimeter edges')
            
    def rotate_90(self):
        angle = 90
        rotation_center = (self.opened_img.shape[1] // 2, self.opened_img.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center, angle, 1.0)
        self.opened_img = cv2.warpAffine(self.opened_img, rotation_matrix, (self.opened_img.shape[1], self.opened_img.shape[0]))
        #print(f'perimeter_direction before: {self.perimeter_direction}')
        try:
            self.perimeter_direction = rotate_orientation(self.perimeter_direction, -1)
        except:
            print(f"could not rotate perimeter, makes sense if this says inner: {self.type}")
        #print(f'perimeter_direction after: {self.perimeter_direction}')
        prev = self.edges.copy()
        self.edges = {}
        #print(f'north direction before: {prev["N"].direction}')
        self.edges["N"] = prev["E"]
        self.edges["E"] = prev["S"]
        self.edges["S"] = prev["W"]
        self.edges["W"] = prev["N"]
        #self.edges['N'].direction = 'N'
        #self.edges['E'].direction = 'E'
        #self.edges['S'].direction = 'S'
        #self.edges['W'].direction = 'W'
        #print(f'north direction after: {self.edges["W"].direction}')
        
        #self.edges['N'] = prev_E
        #self.edges['N'].direction = 'N'
        #self.edges['E'] = prev_S
        #self.edges['E'].direction = 'E'
        #self.edges['S'] = prev_W
        #self.edges['S'].direction = 'S'
        #self.edges['W'] = prev_N
        #self.edges['W'].direction = 'W'
        
        '''
        prev_N = self.edges['N']
        prev_E = self.edges['E']
        prev_S = self.edges['S']
        prev_W = self.edges['W']
        self.edges['N'] = Edge(prev_E.img, 'N', prev_E.corner_1, prev_E.corner_2, self)
        self.edges['E'] = Edge(prev_S.img, 'E', prev_S.corner_1, prev_S.corner_2, self)
        self.edges['S'] = Edge(prev_W.img, 'S', prev_W.corner_1, prev_W.corner_2, self)
        self.edges['W'] = Edge(prev_N.img, 'W', prev_N.corner_1, prev_N.corner_2, self)
        '''
        #print(f'W etter: {self.edges["W"].direction}')
        #print(f'N etter: {self.edges["N"].direction}')
    
class Edge():
    
    def __init__(self, img, direction, corner_1, corner_2 , parent_piece, connected=None, match = None, res=50, gopro=False, test = False):
        self.img = img
        self.type = None
        self.direction = direction
        self.gopro = gopro
        self.test = test
        self.parent_piece = parent_piece
        self.connected = connected
        self.match = match
        self.corner_1 = corner_1
        self.corner_2 = corner_2
        self.res = res
        
        self.corner_2[0] = self.img.shape[1]
        self.corner_1[0] = 0
        self.slope, self.intercept = calculate_line_parameters(self.corner_1, self.corner_2)
        self.distances = calculate_vertical_distances(self.img, self.slope, self.intercept, self.res, crop_strat=True, corner_1=self.corner_1, corner_2=self.corner_2)
        self.distances_flipped = self.distances[::-1]
        if self.gopro:
            self.determine_edge_type_2(threshold=30, percentage=95)
        elif self.test:
            self.determine_edge_type_2(threshold=40, percentage=95)
        else:
            self.determine_edge_type_2()
    
    def show_edge(self, title='edge'):
        show_resized_image(self.img, title=title)
    
    def is_border(self, threshold):
        """
            Fast check to determine of the edge is a border.

            :param threshold: distance threshold
            :return: Boolean
        """

        def dist_to_line(p1, p2, p3):
            return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
        
        total_dist = 0
        for p in self.shape:
            total_dist += dist_to_line(self.shape[0], self.shape[-1], p)
        return total_dist < threshold
    
    
    def determine_edge_type(self, threshold = 2):
        total = 0
        for item in self.distances:
            if isinstance(item, tuple):
                total += sum(item)  # If it's a tuple, sum its elements
            else:
                total += item  # If it's not a tuple, just add the value
        avg = total/len(self.distances)
        #print(f'avg: {avg}')
        if abs(avg) < threshold:
            self.type = 'perimeter'
        else:
            self.type = 'inner'
     
    def determine_edge_type_2(self, threshold = 10, percentage = 93):
        #flattened_distances = [x if not isinstance(x, tuple) else x for sublist in self.distances for x in (sublist if isinstance(sublist, tuple) else [sublist])] 
        flattened_distances = []
        for i in self.distances:
            try:
                flattened_distances.append(i[0]) 
            except:
                flattened_distances.append(i)
        count_abs_below_threshold = np.sum(np.abs(flattened_distances) < threshold)
        percentage_abs_below_threshold = (count_abs_below_threshold / len(flattened_distances)) * 100
        print(percentage_abs_below_threshold)
        if percentage_abs_below_threshold >= percentage:
            self.type = 'perimeter'
        else:
            self.type = 'inner'
        #print(f'Edge type: {self.type}')
    
    
    def compare_edges(self, edge):
        
        if self.direction == 'N':
            return root_mean_square_difference_2(self.distances, edge.distances_flipped)
        if self.direction == 'E':
            pass
           
                