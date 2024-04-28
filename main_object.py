import cv2
import numpy as np
from pathlib import Path
from functions import *
from objects import *
import matplotlib.pyplot as plt
import time

t0 = time.time()
cwd = Path.cwd()
puzzle = Puzzle(test=True)

#for i in range(9):
#    puzzle.load_piece(cv2.imread(f'{cwd}/3/puslespill_{i+1}_alt.png'))



'''
for i in range(9):
    img = cv2.imread(f'{cwd}/1/puzzle_piece_{i+1}.jpg')
    img = img[700:3300, 300:2700]
    print(img.shape)
    show_resized_image(img, f'piece{i+1}')
    puzzle.load_piece(img, test = True)
'''


img = cv2.imread(f'{cwd}/1/puzzle_piece_{6}.jpg')
img = img[700:3300, 300:2700]
puzzle.load_piece(img, test = True)
img = cv2.imread(f'{cwd}/1/puzzle_piece_{2}.jpg')
img = img[700:3300, 300:2700]
puzzle.load_piece(img, test = True)
img = cv2.imread(f'{cwd}/1/puzzle_piece_{5}.jpg')
img = img[700:3300, 300:2700]
puzzle.load_piece(img, test = True)
img = cv2.imread(f'{cwd}/1/puzzle_piece_{7}.jpg')
img = img[700:3300, 300:2700]
puzzle.load_piece(img, test = True)
img = cv2.imread(f'{cwd}/1/puzzle_piece_{1}.jpg')
img = img[700:3300, 300:2700]
puzzle.load_piece(img, test = True)
img = cv2.imread(f'{cwd}/1/puzzle_piece_{9}.jpg')
img = img[700:3300, 300:2700]
puzzle.load_piece(img, test = True)
img = cv2.imread(f'{cwd}/1/puzzle_piece_{4}.jpg')
img = img[700:3300, 300:2700]
puzzle.load_piece(img, test = True)
img = cv2.imread(f'{cwd}/1/puzzle_piece_{8}.jpg')
img = img[700:3300, 300:2700]
puzzle.load_piece(img, test = True)
img = cv2.imread(f'{cwd}/1/puzzle_piece_{3}.jpg')
img = img[700:3300, 300:2700]
puzzle.load_piece(img, test = True)


'''
puzzle.load_piece(cv2.imread(f'{cwd}/3/puslespill_{9}_alt.png'))
puzzle.load_piece(cv2.imread(f'{cwd}/3/puslespill_{8}_alt.png'))
puzzle.load_piece(cv2.imread(f'{cwd}/3/puslespill_{3}_alt.png'))
puzzle.load_piece(cv2.imread(f'{cwd}/3/puslespill_{5}_alt.png'))
puzzle.load_piece(cv2.imread(f'{cwd}/3/puslespill_{4}_alt.png'))
puzzle.load_piece(cv2.imread(f'{cwd}/3/puslespill_{6}_alt.png'))
puzzle.load_piece(cv2.imread(f'{cwd}/3/puslespill_{1}_alt.png'))
puzzle.load_piece(cv2.imread(f'{cwd}/3/puslespill_{7}_alt.png'))
puzzle.load_piece(cv2.imread(f'{cwd}/3/puslespill_{2}_alt.png'))
'''

puzzle.construct_grid()
puzzle.grid_fill_perimeter()
puzzle.grid_fill_inner()
t1 = time.time()
puzzle.display_solved_puzzle()

print(f'total time: {t1-t0}')
#puzzle.find_cornerpiece_match(puzzle.corner_pieces[0])

# Compare every edge to every edge
#while len(puzzle.unmatched_edges) > 0:
#    puzzle.find_matching_edge_rough(puzzle.unmatched_edges[0])
