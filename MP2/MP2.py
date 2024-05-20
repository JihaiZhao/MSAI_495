from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math


def dilation(img_array, img_array_dilation, SE):
    for i in range(SE[0], img_array.shape[1]-SE[1]):
        for j in range(SE[2], img_array.shape[0]-SE[3]):
            if img_array[j][i] == 1:
                # check the surrounding pixel
                for k1 in range(-SE[0], SE[1]+1):
                    for k2 in range(-SE[2], SE[3]+1):
                        img_array_dilation[j][i+k1] = 1
                        img_array_dilation[j+k2][i] = 1
                        img_array_dilation[j+k2][i+k1] = 1

    return img_array_dilation

def erosion(img_array, img_array_dilation, SE):
    for i in range(SE[0], img_array.shape[1]-SE[1]):
        for j in range(SE[2], img_array.shape[0]-SE[3]):
            if img_array[j][i] == 1:
                img_array_dilation[j][i] = 1
                # check the surrounding pixel
                for k1 in range(-SE[0], SE[1]+1):
                    for k2 in range(-SE[2], SE[3]+1):
                        if img_array[j][i+k1] != 1:
                            break
                        if img_array[j+k2][i] != 1:
                            break 
                        if img_array[j+k2][i+k1] != 1:
                            break                        
                    else:
                        continue
                    img_array_dilation[j][i] = 0
                    break

    return img_array_dilation

def opening(img_array, img_array_opening, SE1, SE2):
    img_erosion = erosion(img_array, img_array_opening, SE1)

    img_array_dilation = np.zeros(img_array.shape)
    
    img = dilation(img_erosion, img_array_dilation, SE2)
    return img

def closing(img_array, img_array_closing, SE1, SE2):
    img_dilation = dilation(img_array, img_array_closing, SE1)

    img_array_erosion = np.zeros(img_array.shape)

    img = erosion(img_dilation, img_array_erosion, SE2)
    return img

# Boundary(img)
def boundary(img_array, img_array_boundary):
    # for palm SE1 = [4,4,4,4], SE2 = [2,2,2,2]
    img_close = closing(img_array, img_array_boundary, [3,3,3,3], [4,4,4,4])

    img_array_new = np.zeros(img_array.shape)
    # for palm SE1 = [1,1,1,1]
    img_dilate = erosion(img_close, img_array_new, [1,1,1,1])

    img_array_result = img_close

    for i in range(img_array.shape[1]):
        for j in range(img_array.shape[0]):
            if img_dilate[j][i] == 1:
                img_array_result[j][i] = img_close[j][i] - img_dilate[j][i]

    return img_array_result

def main():
    # Open the image file
    img = Image.open("/home/jihai/MSAI495/MP2/gun.bmp")
    img_array = np.array(img)
    img_array_new = np.zeros(img_array.shape)

    # for palm SE1 = [4,4,4,4]
    # img = dilation(img_array, img_array_new, [3,3,3,3])

    # for palm SE1 = [0,1,0,1]
    # img = erosion(img_array, img_array_new, [0,1,0,1])

    # for palm SE1 = [0,1,0,1], SE2 = [3,3,3,3]
    # img = opening(img_array, img_array_new, [0,1,0,1], [3,3,3,3])

    # for palm SE1 = [4,4,4,4], SE2 = [2,2,2,2]
    # img = closing(img_array, img_array_new, [3,3,3,3], [4,4,4,4])

    img = boundary(img_array, img_array_new)

    plt.imshow(img)
    plt.show()


if __name__=='__main__':
    main()