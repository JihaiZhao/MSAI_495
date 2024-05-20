import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def draw_box(img):
    x1 = 52 
    y1 = 19
    x2 = 91 
    y2 = 67

    # Draw the box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)  
    return img

def Exhaustive_Search(template_img, current_img, method):
    min_num = float('inf')
    min_loc = None
    for i in range(current_img.shape[0]-47):
        for j in range(current_img.shape[1]-38):
            if method == 'SSD':
                SSD = np.sum((current_img[i:i+48, j:j+39] - template_img)**2)
                if SSD < min_num:
                    min_num = SSD
                    min_loc = (j, i)
            elif method == 'CC':
                CC = np.sum((current_img[i:i+48, j:j+39] * template_img))
                if CC < min_num:
                    min_num = CC
                    min_loc = (j, i)
            elif method == 'NCC':
                avg_current = np.mean(current_img[i:i+48, j:j+39])
                avg_template = np.mean(template_img)
                I_hat = current_img[i:i+48, j:j+39] - avg_current
                T_hat = template_img - avg_template
                NCC = np.sum((I_hat * T_hat))/np.sqrt(np.sum(I_hat**2) * np.sum(T_hat**2))
                if NCC < min_num:
                    min_num = NCC
                    min_loc = (j, i)

    return min_num, min_loc


# Folder path containing the images
folder_path = '/home/jihai/MSAI495/MP7/image_girl/'

# List to store the images
images = []

file_names = os.listdir(folder_path)
file_names.sort(key=lambda x: int(x.split('.')[0]))

# Iterate over all files in the folder
for filename in file_names:
    # Construct the full file path
    file_path = os.path.join(folder_path, filename)
    
    # Read the image
    image = cv2.imread(file_path)
    
    # Check if the image is successfully read
    if image is not None:
        images.append(image)
    else:
        print(f"Failed to read image: {file_path}")

gray_image_temp = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
template = gray_image_temp[19:67, 52:91]
pre_loc = [52,19]
pre_ssd = float('inf')
pre_temp = template

# Iterate over all images except the first one
for image in images[1:]:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    # Determine the size of the detection window
    window_size = (template.shape[0] + 20, template.shape[1] + 60)

    # Calculate the top-left corner coordinates of the detection window
    window_top_left = (max(0, pre_loc[1] - 10), max(0, pre_loc[0] - 30))

    # Calculate the bottom-right corner coordinates of the detection window
    window_bottom_right = (min(gray_image.shape[0], window_top_left[0] + window_size[0]),
                        min(gray_image.shape[1], window_top_left[1] + window_size[1]))
    
    # Extract the detection window from the grayscale image
    detection_window = gray_image[window_top_left[0]:window_bottom_right[0]
                                  ,window_top_left[1]:window_bottom_right[1]]
    
    min_ssd, min_loc = Exhaustive_Search(template, detection_window, 'SSD') 
    min_loc = [min_loc[0]+window_top_left[1], min_loc[1]+window_top_left[0]]
    # pre_ssd, min_loc_pre = Exhaustive_Search(pre_temp, detection_window, 'SSD')   
    
    # if min_ssd < pre_ssd:
    #     min_loc = [min_loc[0]+window_top_left[1], min_loc[1]+window_top_left[0]]
    # else:
    #     min_loc = [min_loc_pre[0]+window_top_left[1], min_loc_pre[1]+window_top_left[0]]

    image2_with_box = image.copy()
    cv2.rectangle(image2_with_box, min_loc, (min_loc[0] + template.shape[1], min_loc[1] + template.shape[0]), (0, 255, 0), 2)   

    pre_temp = gray_image_temp[min_loc[1]: min_loc[1]+48, min_loc[0]:min_loc[0]+39]

    # Display images
    plt.subplot(121), plt.imshow(cv2.cvtColor(draw_box(images[0]), cv2.COLOR_BGR2RGB))
    plt.title('Template'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(cv2.cvtColor(image2_with_box, cv2.COLOR_BGR2RGB))
    plt.title('Result'), plt.xticks([]), plt.yticks([])
    plt.show(block=False)
    plt.pause(0.1)  # Pause for a short duration to allow the window to refresh

    plt.clf()