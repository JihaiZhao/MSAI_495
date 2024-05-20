import cv2
import matplotlib.pyplot as plt
import numpy as np

def build_histo(img):
    histogram = np.zeros((180, 256))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            h = img[row, col, 0].flatten()  # Hue
            s = img[row, col, 1].flatten()  # Saturation
            histogram[int(h), int(s)] += 1
    return histogram / np.sum(histogram)  # Normalize the histogram

def main():
    # Load training images
    training1 = cv2.imread('/home/jihai/MSAI495/MP4/train1.bmp')
    training1_hsv = cv2.cvtColor(training1, cv2.COLOR_BGR2HSV)
    training2 = cv2.imread('/home/jihai/MSAI495/MP4/train2.bmp')
    training2_hsv = cv2.cvtColor(training2, cv2.COLOR_BGR2HSV)
    training3 = cv2.imread('/home/jihai/MSAI495/MP4/train3.bmp')
    training3_hsv = cv2.cvtColor(training3, cv2.COLOR_BGR2HSV)
    training4 = cv2.imread('/home/jihai/MSAI495/MP4/train4.bmp')
    training4_hsv = cv2.cvtColor(training4, cv2.COLOR_BGR2HSV)
    training5 = cv2.imread('/home/jihai/MSAI495/MP4/train5.bmp')
    training5_hsv = cv2.cvtColor(training5, cv2.COLOR_BGR2HSV)
    training6 = cv2.imread('/home/jihai/MSAI495/MP4/train6.bmp')
    training6_hsv = cv2.cvtColor(training6, cv2.COLOR_BGR2HSV)
    training7 = cv2.imread('/home/jihai/MSAI495/MP4/train7.bmp')
    training7_hsv = cv2.cvtColor(training7, cv2.COLOR_BGR2HSV)
    training8 = cv2.imread('/home/jihai/MSAI495/MP4/train8.bmp')
    training8_hsv = cv2.cvtColor(training8, cv2.COLOR_BGR2HSV)
    training9 = cv2.imread('/home/jihai/MSAI495/MP4/train9.bmp')
    training9_hsv = cv2.cvtColor(training9, cv2.COLOR_BGR2HSV)

    # Load testing image
    test = cv2.imread('/home/jihai/MSAI495/MP4/pointer1.bmp')
    test_hsv = cv2.cvtColor(test, cv2.COLOR_BGR2HSV)

    # Build histograms for training images
    histo1 = build_histo(training1_hsv)
    histo2 = build_histo(training2_hsv)
    histo3 = build_histo(training3_hsv)
    histo4 = build_histo(training4_hsv)
    histo5 = build_histo(training5_hsv)
    histo6 = build_histo(training6_hsv)
    histo7 = build_histo(training7_hsv)
    histo8 = build_histo(training8_hsv)
    histo9 = build_histo(training9_hsv)
    
    combined_histo = (histo1+histo2+histo3+histo4+histo5+histo6+histo7+histo8+histo9)

    # Classify pixels based on combined histogram
    img2 = np.zeros((test.shape[0], test.shape[1]), dtype=np.uint8)
    threshold = 0.0001
    for i in range(test.shape[0]):
        for j in range(test.shape[1]):
            if combined_histo[test_hsv[i, j, 0], test_hsv[i, j, 1]] > threshold:
                print(test_hsv[i, j, 0])
                img2[i, j] = 255  # Mark as skin color pixel
            else:
                img2[i, j] = 0

    colored_img = test.copy()  # Create a copy of the original testing image

    # Overlay skin color pixels onto the colored image
    for i in range(test.shape[0]):
        for j in range(test.shape[1]):
            if img2[i, j] == 255:  # Skin color pixel
                colored_img[i, j] = test[i, j]
            else:
                colored_img[i, j] = (0, 0, 0)
    
    colored_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB)

    # Display the colored image
    plt.imshow(colored_img)
    plt.title('Skin Color Pixels (Colored)')
    plt.show()

    # # Generate grid coordinates for the 2D histogram
    # x = np.arange(256)
    # y = np.arange(180)
    # X, Y = np.meshgrid(x, y)

    # fig, axes = plt.subplots(1, subplot_kw={'projection': '3d'})

    # # Plot the first histogram
    # axes.plot_surface(X, Y, combined_histo, cmap='jet')
    # axes.set_xlabel('Saturation')
    # axes.set_ylabel('Value')
    # axes.set_zlabel('Frequency')
    # axes.set_title('Histogram 1')

    # plt.show()


if __name__ == '__main__':
    main()