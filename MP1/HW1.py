from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def ccl(img_array, labeling, E_table, L):
    for i in range(1, img_array.shape[1]):
        for j in range(1, img_array.shape[0]):
            if img_array[j][i] == 1:
                # upper label
                L_u = labeling[j-1][i]
                # left label
                L_l = labeling[j][i-1]
                if (L_u == L_l) and (L_l != 0 and L_u !=0):
                    labeling[j][i] = L_u
                elif L_u!=L_l and not (L_u and L_l):
                    labeling[j][i] = max(L_u , L_l)
                elif L_u!=L_l and (L_u >0 and L_l >0):
                    labeling[j][i]=min(L_u , L_l)
                    if min(L_u, L_l) in E_table: 
                        E_table[max(L_u , L_l)] = E_table[min(L_u, L_l)]
                    else:
                        E_table[max(L_u , L_l)] = min(L_u, L_l)
                else: 
                    labeling[j][i] = L+1
                    L+=1
    print(E_table)
    for i in range(img_array.shape[1]):
        for j in range(img_array.shape[0]):
            if img_array[j][i] == 1:
                if labeling[j][i] in E_table:
                    labeling[j][i] = E_table[labeling[j][i]]

def noise_avoid(img_array, labeling):
    # add threshold to ignore noise
    num_count = {}
    for i in range(img_array.shape[1]):
        for j in range(img_array.shape[0]):
            if labeling[i][j] in num_count:
                num_count[labeling[j][i]] += 1
            else:
                num_count[labeling[j][i]] = 1

    for i in range(img_array.shape[1]):
        for j in range(img_array.shape[0]):
            if img_array[j][i] == True:
                if num_count[labeling[j][i]] < 500:
                    labeling[j][i] = 0


def main():
    # Open the image file
    img = Image.open("/home/jihai/MSAI495/MP1/face.bmp")
    img_array = np.array(img)
    print(img_array.shape[0])
    labeling = np.zeros(img_array.shape)
    L = 0
    E_table = {}
    ccl(img_array, labeling, E_table, L)

    # use noise_avoid for gun.bmp
    # noise_avoid(img_array, labeling)

    thresholded_img = Image.fromarray((labeling).astype(np.uint8))

    # Start with white (1, 1, 1), then blue (0, 0, 1), green (0, 1, 0), and red (1, 0, 0)
    colors = [(1, 1, 1), (0, 0, 1), (0, 1, 0), (1, 0, 0)]  

    # Define the positions for each color (ranging from 0 to 1)
    positions = [0.0, 0.3, 0.6, 1.0]

    # Create the custom colormap
    cmap_name = 'custom_colormap'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, list(zip(positions, colors)))

    # Display the image
    plt.imshow(thresholded_img, cmap=custom_cmap)
    plt.colorbar()
    plt.show()


if __name__=='__main__':
    main()
