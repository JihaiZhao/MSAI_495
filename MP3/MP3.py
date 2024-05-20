import cv2
import matplotlib.pyplot as plt
import numpy as np

def HistoEqualization(image):
    histogram = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            histogram[image[i][j]]+=1
    for i in range(len(histogram)):
        histogram[i] /= 65536

    cdf_m = histogram.cumsum()

    cdf = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = cdf.astype('uint8')
    img2 = cdf[image]

    print(img2)

    histogram2 = np.zeros(256)
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            histogram2[img2[i][j]]+=1
    for i in range(len(histogram2)):
        histogram2[i] /= 65536

    return img2

def lighting_correction_linear(image):
    img = np.zeros([image.shape[0], image.shape[1]])
    A = np.ones([image.shape[0],3])
    for j in range(image.shape[0]):
        A[j][0] = j
        A[j][1] = j

    intensity = np.zeros(image.shape[0])
    mean_intensity = np.mean(image)

    for i in range(image.shape[0]):
        intensity[i] = image[i][i]/256

    res = np.dot(np.linalg.pinv(A), intensity)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img[i][j] = image[i][j] - (res[0] * i + res[1] * j + res[2]) * 255 + mean_intensity

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def lighting_correction_quadratic(image):
    img = np.zeros([image.shape[0], image.shape[1]])
    A = np.ones([image.shape[1],6])
    for j in range(image.shape[1]):
        A[j][0] = j * j
        A[j][1] = j * j
        A[j][2] = j * j
        A[j][3] = j
        A[j][4] = j

    intensity = np.zeros(image.shape[0])
    mean_intensity = np.mean(image)

    for i in range(image.shape[0]):
        intensity[i] = image[i][i]/256

    res = np.dot(np.linalg.pinv(A), intensity)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img[i][j] = image[i][j] - (res[0] * i*i + res[1] * j*j + res[2] * j*i + res[3]*i + res[4]*j + res[5]) * 255 + mean_intensity

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def main():

    # Load the image
    image = cv2.imread('/home/jihai/MSAI495/MP3/moon.bmp')

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    img_euqalized = HistoEqualization(gray_image)
    # print(img_euqalized)

    linear_fit = lighting_correction_linear(img_euqalized)

    quadratic_fit = lighting_correction_quadratic(img_euqalized)

    # Plot the histogram
    plt.subplot(2, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original Image')
    plt.subplot(2, 2, 2)
    plt.imshow(img_euqalized, cmap='gray')
    plt.title('Equalized Image')
    plt.subplot(2, 2, 3)
    plt.imshow(linear_fit, cmap='gray')
    plt.title('Lighting correction - linear')
    plt.subplot(2, 2, 4)
    plt.imshow(quadratic_fit, cmap='gray')
    plt.title('Lighting correction - quadratic')

    plt.show()

if __name__=='__main__':
    main()