#import dataset
!kaggle datasets download -d tapakah68/yandextoloka-water-meters-dataset
!unzip yandextoloka-water-meters-dataset.zip -d /content/dataset
-----------------------------------------------------------------------------
import os
os.listdir('/content/dataset')
------------------------------------------------------------------------------

#Import Relevant Libraries
import re, cv2, os, json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import decimal
import shutil
-----------------------------------------------------------------------------
# Load the dataset
data = pd.read_csv('/content/dataset/WaterMeters/data.csv')

# Set the correct paths for images, masks, and collages
images_folder = "/content/dataset/WaterMeters/images"
masks_folder = "/content/dataset/WaterMeters/masks"
coll_folder = "/content/dataset/WaterMeters/collage"
-------------------------------------------------------------------------------
#Obtain a count of images, masks, and observations.
print(f'Total number of images: {len(os.listdir(images_folder))}')
print(f'Total number of image masks: {len(os.listdir(masks_folder))}')
print(f'Length of dataset: {len(data)}')
-------------------------------------------------------------------------------

import os
import cv2
import matplotlib.pyplot as plt

# Define the path to the images folder
images_folder = '/content/dataset/WaterMeters/images'

# Create figure and empty list for axes
axes = []
fig = plt.figure(figsize=(15, 15))

# Show first 4 images in the dataset with the corresponding shape
for a in range(4):
    # Obtain file name and create path
    file = os.listdir(images_folder)[a]  # Using images_folder path here
    image_path = os.path.join(images_folder, file)

    # Read the file image and resize it for display
    img = cv2.imread(image_path)
    resized_image = cv2.resize(img, (1300, 1500), interpolation=cv2.INTER_AREA)

    # Print the resized image and display the shape
    axes.append(fig.add_subplot(1, 4, a+1))
    subplot_title = f"Original Size: {img.shape}"
    axes[-1].set_title(subplot_title)
    plt.imshow(resized_image)

# Remove ticks from each image
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

# Plot the images
fig.tight_layout()
plt.show()

-------------------------------------------------------------------------------
#analysing the area of interest
import os
import cv2
import matplotlib.pyplot as plt

# Define the correct path for masks
masks_folder = '/content/dataset/WaterMeters/masks'

# Create figure and empty list for axes
axes = []
fig = plt.figure(figsize=(15, 15))

# Show first 4 mask images in dataset with corresponding shape
for a in range(4):
    # Obtain file name and create full path
    file = os.listdir(masks_folder)[a]
    image_path = os.path.join(masks_folder, file)

    # Read the image and resize it for display
    img = cv2.imread(image_path)
    resized_image = cv2.resize(img, (1300, 1500), interpolation=cv2.INTER_AREA)

    # Add subplot and display image with shape as title
    axes.append(fig.add_subplot(1, 4, a + 1))
    subplot_title = f"Original Size: {img.shape}"
    axes[-1].set_title(subplot_title)
    plt.imshow(resized_image)

# Remove ticks
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

# Show plot
fig.tight_layout()
plt.show()
