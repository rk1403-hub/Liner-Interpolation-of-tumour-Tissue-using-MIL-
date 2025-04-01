import os
import openslide
from openslide import open_slide
import openslide
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
from openslide.deepzoom import DeepZoomGenerator

# Open the SVS image using Openslide
slide = openslide.OpenSlide('/Users/prajwalrk/Downloads/CRC-dataset/TU58492_97 L.svs')
width, height = slide.dimensions
print(width, height)

thumbnail = slide.get_thumbnail((640, 480))  # Adjust size as needed

# Convert thumbnail to NumPy array
image = np.array(thumbnail)

# Convert color space
if image.shape[2] == 4:
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

# Display the image
plt.imshow(image)
plt.title('Medical Image')
plt.axis('off')  # Hide x and y axes
plt.show()

#Generate object for tiles using the DeepZoomGenerator
tiles = DeepZoomGenerator(slide, tile_size=2048, overlap=0, limit_bounds=False)
#Here, we have divided our svs into tiles of size 2048 with no overlap. 

#The tiles object also contains data at many levels. 
#To check the number of levels
print("The number of levels in the tiles object are: ", tiles.level_count)

#How many tiles at a specific level?
level_num = 14
print("Tiles shape at level ", level_num, " is: ", tiles.level_tiles[level_num])

###### Saving each tile to local directory
cols, rows = tiles.level_tiles[14]
print(cols, rows)

import os
tile_dir = "/Users/prajwalrk/Desktop/code_thesis/tiles/TU58492_97 L"

# Check if the folder exists
if not os.path.exists(tile_dir):
    # Create the folder if it doesn't exist
    os.makedirs(tile_dir)
    print(f"Folder '{tile_dir}' created!")
else:
    print(f"Folder '{tile_dir}' already exists.")
    
for row in range(rows):
    for col in range(cols):
        tile_name = os.path.join(tile_dir, '%d_%d' % (col, row))
        print("Now saving tile with title: ", tile_name)
        temp_tile = tiles.get_tile(level_num, (col, row))
        temp_tile_RGB = temp_tile.convert('RGB')
        temp_tile_np = np.array(temp_tile_RGB)
        plt.imsave(tile_name + ".png", temp_tile_np)

print('Done!!!')

def find_rows_cols(folder_path):
  """
  This function iterates over image files in a folder named "original_tiles" and
  returns the number of rows and columns based on the filename format "row_col.png".

  Args:
      folder_path (str): Path to the folder containing image tiles.

  Returns:
      tuple: A tuple containing the number of rows (int) and columns (int).
  """
  rows = 0
  cols = 0

  for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        # Extract row and column numbers from the filename
        try:
            col_str, row_str = filename.split("_")
            row = int(row_str.split(".")[0])
            col = int(col_str)
        except ValueError:
          #   # Handle files with invalid naming format
            print(f"Skipping file: {filename} (invalid format)")
            continue

    # Update max row and column values
    rows = max(rows, row + 1)  # +1 to account for 0-based indexing
    cols = max(cols, col + 1)  # +1 to account for 0-based indexing

  return rows, cols

def verify_tiles(folder_path):
    """
    This function reads image tiles from a folder, concatenates them based on row and column positions,
    and plots the resulting original image.
  
    Args:
        folder_path (str): Path to the folder containing image tiles.
    """
    
    rows, cols = find_rows_cols(folder_path)

    # Initialize an empty image with appropriate dimensions
    empty_image = np.zeros((rows * 2048, cols * 2048, 3), dtype=np.uint8)  # Assuming tiles are 2048*2048 and RGB
    tile_size = 2048  # Assuming tile size (adjust if needed)

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            col_str, row_str = filename.split("_")
            row = int(row_str.split(".")[0])
            col = int(col_str)

            # Read the image tile
            tile = cv2.imread(os.path.join(folder_path, filename))
            tile = cv2.resize(tile, (2048, 2048), interpolation=cv2.INTER_LINEAR)
            
            # Check if tile has the expected dimensions (optional)
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                print(f"Warning: Tile {filename} has unexpected dimensions.")
                continue

            # Place the tile in its corresponding position in the empty image
            y_start = row * tile_size
            y_end = y_start + tile_size
            x_start = col * tile_size
            x_end = x_start + tile_size
            
            empty_image[y_start:y_end, x_start:x_end] = tile


    plt.imshow(cv2.cvtColor(empty_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    plt.show()

verify_tiles(tile_dir)



