import cv2
import numpy as np

# Function to extract (H, S) values from an image
def extract_hs_values(img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0].flatten()
    s = hsv[:,:,1].flatten()
    hs_values = np.column_stack((h, s))
    return hs_values

# Function to build a 2D color histogram Histo(H,S)
def build_histogram(samples):
    histo = np.zeros((180, 256))
    np.add.at(histo, (samples[:,0], samples[:,1]), 1)
    return histo / np.sum(histo)

# Function to apply skin color detection using histogram and threshold
def detect_skin_color(testing_img, histo, threshold=0.0001):
    testing_hsv = cv2.cvtColor(testing_img, cv2.COLOR_BGR2HSV)
    h_test = testing_hsv[:,:,0]
    s_test = testing_hsv[:,:,1]
    
    skin_mask = histo[h_test, s_test] > threshold
    skin_color_regions = np.zeros_like(testing_img)
    skin_color_regions[skin_mask] = testing_img[skin_mask]
    
    return skin_color_regions

# List to store samples from multiple images
all_samples = []

# Paths to multiple images
image_paths = ['/home/jihai/MSAI495/MP4/train1.bmp', 
               '/home/jihai/MSAI495/MP4/train2.bmp', 
               '/home/jihai/MSAI495/MP4/train3.bmp',
               '/home/jihai/MSAI495/MP4/train4.bmp',
               '/home/jihai/MSAI495/MP4/train5.bmp',
               '/home/jihai/MSAI495/MP4/train6.bmp',
               '/home/jihai/MSAI495/MP4/train7.bmp',
               '/home/jihai/MSAI495/MP4/train8.bmp',
               '/home/jihai/MSAI495/MP4/train9.bmp',
               ]

# Extract (H, S) values from each image
for img_path in image_paths:
    hs_values = extract_hs_values(img_path)
    all_samples.extend(hs_values)

# Build histogram from samples
histo = build_histogram(np.array(all_samples))

# Load the testing image
testing_img = cv2.imread('/home/jihai/MSAI495/MP4/pointer1.bmp')

# Apply skin color detection
skin_color_regions = detect_skin_color(testing_img, histo)

# Display the result
cv2.imshow('Skin Color Regions', skin_color_regions)
cv2.waitKey(0)
cv2.destroyAllWindows()