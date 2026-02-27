# Imports

import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import cv2
from tqdm import tqdm_notebook, tnrange
from glob import glob
from itertools import chain

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers.legacy import Adam # Using Legacy 'Adam' Optimizer for 'decay'.

from sklearn.model_selection import train_test_split

from utils import *
from unet import *

import pprint # Printing objects


# Seed
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)


# Setting Image Size Parameters
im_width = 256
im_height = 256

# Number of Rows and Columns for Printing the image
row = 3
column = 3
n = 3

# HyperParameters
EPOCHS = 100
BATCH_SIZE = 32 
LEARNING_RATE = 1e-4
SMOOTH = 100

# Loading Image Path and Mask Path
image_filenames_train, mask_files = load_image_filename_train()

# Plot Image and Masks
plot_from_img_path(row, column, image_filenames_train, mask_files) # Prints Image and Mask while overlapping each other.
show_img_mask_rows(n, image_filenames_train, mask_files) # Prints Image and Mask Image at side by side.

# DataFrame
df = pd.DataFrame(data= {
    'image_filenames_train':image_filenames_train,
    'mask':mask_files
})


# Train Test Split
df_train, df_test = train_test_split(df, test_size=0.1)
df_train, df_val = train_test_split(df_train, test_size=0.2)
print(df_train.shape, df_val.shape, df_test.shape)

# Generator 
train_generator_param = dict(
    rotation_range=0.2,          # Randomly rotate images up to ±20% degrees
    width_shift_range=0.05,      # Random horizontal shift up to 5% of image width
    height_shift_range=0.05,     # Random vertical shift up to 5% of image height
    shear_range=0.05,            # Apply small shear transformations
    zoom_range=0.05,             # Random zoom in/out up to 5%
    horizontal_flip=True,        # Randomly flip images horizontally
    fill_mode='nearest'          # Fill missing pixels after transform using nearest values
)

train_gen = train_generator(
    df_train,                    # DataFrame containing training image paths and labels
    BATCH_SIZE,                 # Number of samples per training batch
    train_generator_param,      # Augmentation configuration dictionary
    target_size=(im_height, im_width)  # Resize images to model input size
)

# Not applying Augmentation on Test Set.
val_gen = train_generator(
    df_val,
    BATCH_SIZE,
    dict(),  # No augmentation for validation
    target_size=(im_height, im_width)
)

# Optimizers
# Decay rate gradually reduces the learning rate during training so the model "Learns fast at the beginning. Fine-tunes carefully later".
# Learning rate schedule (modern replacement for decay)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=LEARNING_RATE,   # Starting learning rate
    decay_steps=1000,                      # Steps before each decay update
    decay_rate=LEARNING_RATE / EPOCHS,    # How fast LR decreases over time
    staircase=True                        # Apply decay in discrete steps
)

optimizer = Adam(
    learning_rate=lr_schedule,  # Uses LR schedule instead of fixed value
    beta_1=0.9,                 # Momentum for first moment (mean of gradients)
    beta_2=0.999,               # Momentum for second moment (variance of gradients)
    epsilon=1e-7,               # Small value to avoid division by zero
    amsgrad=False               # AMSGrad variant (improves convergence)
)

# Model
model = unet(input_size=(im_height, im_width, 3)) # Input shape (Height, Width, RGB channels)

model.compile(
    optimizer=optimizer,        # Optimization algorithm
    loss=dice_coefficients_loss,# Loss function (Dice loss for segmentation)
    metrics=[
        'binary_accuracy',      # Pixel-wise classification accuracy
        iou,                    # Intersection over Union metric
        dice_coefficients       # Dice similarity score
    ]
)

callbacks = [
    ModelCheckpoint(
        'unet.keras',        # File path to save the best model
        save_best_only=True  # Save model only when validation performance improves
    ),
    EarlyStopping(
        patience=10,                 # Stop training if validation metric doesn't improve for 10 epochs
        restore_best_weights=True   # Reload weights from the best validation epoch
    )
]



history = model.fit(
    train_gen,                                 # Training data generator
    steps_per_epoch=len(df_train) // BATCH_SIZE,# Batches per epoch
    epochs=EPOCHS,                             # Total number of training epochs
    callbacks=callbacks,                       # List of callbacks
    validation_data=val_gen, # Validation generator
    validation_steps=len(df_val) // BATCH_SIZE # Validation batches per epoch
)

# Preety Print Objects
pprint.pprint(history.history)

# Plot Accuracy and Loss
plot_accuracy_loss(history)



# Load Previouly Trained Model
model = load_model("unet.keras", custom_objects = {"dice_coefficients_loss": dice_coefficients_loss, 'iou': iou,
                                                 "dice_coefficients" : dice_coefficients})

test_gen = train_generator(df_test, BATCH_SIZE, dict(), target_size = (im_height, im_width))
results = model.evaluate(test_gen, steps = len(df_test)//BATCH_SIZE)

print("TEST LOSS", results[0])
print("TEST IOU", results[1])
print("TEST DICE COEFFICIENT", results[2])


# Plot Prediction
for i in range(20):
    index = np.random.randint(1, len(df_test.index))
    img = cv2.imread(df_test['image_filenames_train'].iloc[index]) # Original Image NOt Mask
    img = cv2.resize(img, (im_height, im_width))
    img = img / 255
    #print(img.shape) (256, 256, 3)
    img = img[np.newaxis, :, :, :] # 3d array will become 4d array
    #print(img.shape) (1, 256, 256, 3)
    predicted_img = model.predict(img)

    plt.figure(figsize=(12,12))
    # 3 columns original image, mask, predicted image
    plt.subplot(1,3,1)
    plt.imshow(np.squeeze(img))
    plt.title('Original IMage')

    plt.subplot(1,3,2)
    plt.imshow(np.squeeze(cv2.imread(df_test['mask'].iloc[index])))
    plt.title("Mask Image")

    plt.subplot(1,3,3)
    plt.imshow(np.squeeze(predicted_img) > 0.5) # Checking Probabilities
    plt.title('Predicted Image')

    plt.show()