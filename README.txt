
# README - Part_1 Folder

## Overview
This project focuses on colorectal tumor detection using whole-slide images. It involves generating tiles from the original images, classifying these tiles, and visualizing the results through heat maps. The `Part_1` directory contains scripts for tile generation, model loading, heat map generation, and the corresponding dataset.

### Folder Structure:
- `Model/`
  - Contains the pre-trained weights and classifier used to classify the image tiles.
  - **Files**:
    - `PretrainedModel#1.pth`: Pre-trained model for tile classification.
    - `test_bag#1.pt`: Test dataset sample (bag) for model evaluation.
    - `load_model.py`: Script to load the pre-trained model.
    - `demo.py`: Script for running demo classifications on test data.
    - `__pycache__/`: Python cache folder (automatically generated).

- `CRC-dataset/`
  - Contains the whole slide images (WSIs) of colorectal tumor tissue. These images are processed to generate tiles using the `tiles_gen.py` script.

- `tiles copy/`
  - Folder that will store the generated tiles for each image after running `tiles_gen.py`.

## Files and Scripts:
- `heat_map_gen.py`:
  - This script generates heat maps based on the tile classification results.
  - **Input**: The tiles generated using the `tiles_gen.py` script.
  - **Output**: Heat maps showing the tumor/tissue distribution, stored in the `heat_map` folder.

- `tiles_gen.py`:
  - This script is responsible for generating image tiles from the WSIs stored in the `CRC-dataset` folder.
  - **Input**: Whole-slide images from `CRC-dataset`.
  - **Output**: Tiles stored in the `tiles copy` folder, separated by image.

- `seperating_tiles.py`:
  - Script to process and organize the tiles generated from WSIs. It separates the tiles based on specific criteria.

## Workflow:

1. **Tile Generation**:
   - Run the `tiles_gen.py` script to generate tiles from the whole-slide images located in the `CRC-dataset` folder.
   - The tiles will be stored in the `tiles copy` folder, where each image's tiles are organized into subfolders.

2. **Heat Map Generation**:
   - After generating the tiles, run the `heat_map_gen.py` script to construct heat maps based on tile classification.
   - The heat maps will visualize the tumor distribution across the slide images and are stored in a designated folder (e.g., `heat_mapp`).

3. **Model Classification**:
   - The `Model/` folder contains a pre-trained model for tile classification.
   - You can load the model using `load_model.py` and classify the generated tiles. The classification results will feed into the heat map generation process.

---

This file provides an overview of the project structure, functionality of each script, and a step-by-step guide to generating tiles and heat maps for colorectal tumor detection.
