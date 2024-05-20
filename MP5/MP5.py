import cv2
import matplotlib.pyplot as plt
import numpy as np

def GaussSmoothing(I, N, Sigma):
    G = np.zeros((N,N))
    S = np.zeros((I.shape[0],I.shape[1]))
    for i in range(0, N):
        for j in range(0, N):
            G[i,j] = (1/(2*np.pi*Sigma**2)) * np.exp(-((i - (N-1)/2)**2 + (j - (N-1)/2)**2)/(2*Sigma**2))
            
    G = G/sum(G)

    for i in range(N//2, I.shape[0] - N//2):
        for j in range(N//2, I.shape[1] - N//2):
            S[i, j] = np.sum(I[i - N // 2:i + N // 2 + 1, j - N // 2:j + N // 2 + 1] * G)

    return S

def ImageGradient(S, method):
    mag = np.zeros((S.shape[0],S.shape[1]))
    theta = np.zeros((S.shape[0],S.shape[1]))
    if method == 'sobel':
        for i in range(2, S.shape[0] - 2):
            for j in range(2, S.shape[1] - 2):
                i_x = (S[i,j] - S[i,j-2]) + 2 * (S[i-1,j] - S[i-1,j-2]) + (S[i-2,j] - S[i-2,j-2])
                i_y = (S[i-2,j] - S[i,j]) + 2 * (S[i-2,j-1] - S[i,j-1]) + (S[i-2,j-2] - S[i,j-2])
                mag[i,j] = np.sqrt(i_x**2 + i_y**2)
                theta[i,j] = np.arctan2(i_y,i_x)

        mag = cv2.normalize(mag, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    elif method == 'roberts':
        # Roberts edge detection
        roberts_x = np.array([[1, 0], [0, -1]])
        roberts_y = np.array([[0, 1], [-1, 0]])
        roberts_x_edges = cv2.filter2D(S, -1, roberts_x)
        roberts_y_edges = cv2.filter2D(S, -1, roberts_y)
        
        # Compute edge magnitude and orientation
        mag = np.sqrt(roberts_x_edges**2 + roberts_y_edges**2)
        theta = np.arctan2(roberts_y_edges, roberts_x_edges)
        mag = cv2.normalize(mag, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
    elif method == 'zerocross':
        # Zero-crossing edge detection
        laplacian = cv2.Laplacian(S, cv2.CV_64F)
        mag = np.array(np.abs(laplacian))
        theta = np.zeros_like(laplacian)  # No orientation for zero-crossing method
        mag = cv2.normalize(mag, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return mag, theta

def NonmaximaSupress(Mag, Theta):
    mag = Mag.copy()
    for i in range(1, Mag.shape[0] - 1):
        for j in range(1, Mag.shape[1] - 1):
            # use LUT, there are total 8 different combination
            if(Theta[i,j] >= -np.pi/8 and Theta[i,j] < np.pi/8):
                if (Mag[i,j] < Mag[i+1,j] or Mag[i,j] < Mag[i-1,j]):
                    mag[i,j] = 0
            elif(Theta[i,j] >= np.pi/8 and Theta[i,j] < np.pi*3/8):
                if (Mag[i,j] < Mag[i-1,j+1] or Mag[i,j] < Mag[i+1,j-1]):
                    mag[i,j] = 0
            elif(Theta[i,j] >= np.pi*3/8 and Theta[i,j] < np.pi*5/8):
                if (Mag[i,j] < Mag[i,j-1] or Mag[i,j] < Mag[i,j+1]):
                    mag[i,j] = 0
            elif(Theta[i,j] >= np.pi*5/8 and Theta[i,j] < np.pi*7/8):
                if (Mag[i,j] < Mag[i-1,j-1] or Mag[i,j] < Mag[i+1,j+1]):
                    mag[i,j] = 0
            elif(Theta[i,j] >= np.pi*7/8 or Theta[i,j] < -np.pi*7/8):
                if (Mag[i,j] < Mag[i+1,j] or Mag[i,j] < Mag[i-1,j]):
                    mag[i,j] = 0
            elif(Theta[i,j] >= -np.pi*7/8 and Theta[i,j] < -np.pi*5/8):
                if (Mag[i,j] < Mag[i+1,j-1] or Mag[i,j] < Mag[i-1,j+1]):
                    mag[i,j] = 0
            elif(Theta[i,j] >= -np.pi*5/8 and Theta[i,j] < -np.pi*3/8):
                if (Mag[i,j] < Mag[i,j+1] or Mag[i,j] < Mag[i,j-1]):
                    mag[i,j] = 0
            elif(Theta[i,j] >= -np.pi*3/8 and Theta[i,j] < -np.pi/8):
                if (Mag[i,j] < Mag[i+1,j+1] or Mag[i,j] < Mag[i-1,j-1]):
                    mag[i,j] = 0  
    return mag

def FindThreshold(Mag, percentageOfNonEdge):
    histogram = np.zeros(256)
    for i in range(Mag.shape[0]):
        for j in range(Mag.shape[1]):
            histogram[Mag[i][j]]+=1

    hist_normalized = histogram / np.sum(histogram)
    # print(np.sum(histogram))
    cdf_m = hist_normalized.cumsum()
    cdf = (cdf_m - cdf_m.min())/(cdf_m.max()-cdf_m.min())

    # Find low and high thresholds based on percentageOfNonEdge

    high_threshold = np.argmax(cdf > (percentageOfNonEdge / 100))
    low_threshold = 0.5*high_threshold
    
    return low_threshold, high_threshold

def double_thresholding(grad_mag, low_threshold, high_threshold):
    # Create arrays to store edge information
    strong_edges = np.zeros_like(grad_mag)
    weak_edges = np.zeros_like(grad_mag)
    
    # Find strong edges
    strong_edges[grad_mag >= high_threshold] = 255
    
    # Find weak edges
    weak_edges[(grad_mag >= low_threshold) & (grad_mag <= high_threshold)] = 100
    
    return strong_edges, weak_edges

def EdgeLinking(Mag_low, Mag_high):
    # Create a copy of Mag_high to store the result
    mag_res = Mag_high.copy()
    
    # 8-neighbor in strong edges
    offsets = [(i, j) for i in range(-1, 2) for j in range(-1, 2) if not (i == 0 and j == 0)]
    
    # Iterate over each pixel in Mag_high
    for i in range(2, Mag_high.shape[0] - 2):
        for j in range(2, Mag_high.shape[1] - 2):
            # If the pixel is a weak edge
            if Mag_low[i, j] > 0 and Mag_low[i, j] < 255:
                # Check its 8-connected neighbors for strong edges
                for offset in offsets:
                    ni, nj = i + offset[0], j + offset[1]
                    if Mag_high[ni, nj] == 255:
                        # If a strong edge is found, mark the weak edge as strong
                        mag_res[i, j] = 255
                        break  # Exit the loop after finding one strong edge
    
    return mag_res

def main():
    # Load the image
    image = cv2.imread('/home/jihai/MSAI495/MP5/pointer1.bmp')
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    plt.show()

    image_gaussSmothing = GaussSmoothing(gray_image,3,3)
    plt.imshow(image_gaussSmothing, cmap='gray')
    plt.title('Gaussian smoothing')
    plt.axis('off')
    plt.show()

    mag, theta = ImageGradient(image_gaussSmothing, 'roberts')
    plt.imshow(mag, cmap='gray')
    plt.title('Sobel operators')
    plt.axis('off')
    plt.show()

    mag_2 = NonmaximaSupress(mag, theta)
    plt.imshow(mag_2, cmap='gray')
    plt.title('Non-maxima suppressing')
    plt.axis('off')
    plt.show()

    T_low, T_high = FindThreshold(mag_2,65)
    print(T_low)
    print(T_high)
    high, low = double_thresholding(mag_2,T_low,T_high)
    plt.imshow(high, cmap='gray')
    plt.title('T_high')
    plt.axis('off')
    plt.show()

    plt.imshow(low, cmap='gray')
    plt.title('T_low')
    plt.axis('off')
    plt.show()

    E = EdgeLinking(low, high)
    plt.imshow(E, cmap='gray')
    plt.title('E')
    plt.axis('off')
    plt.show()


if __name__=='__main__':
    main()