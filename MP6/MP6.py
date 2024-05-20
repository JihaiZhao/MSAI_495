import numpy as np
import cv2
import matplotlib.pyplot as plt
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
                if theta[i,j] < -np.pi/2:
                    theta[i,j] += np.pi
                elif theta[i,j] > np.pi/2:
                    theta[i,j] -= np.pi
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
        
    return mag, theta

def hough_transform(image, theta_resolution=1):
    height, width = image.shape
    max_rho = int(np.sqrt(height**2 + width**2))  # Maximum possible rho value

    # Define the range of theta from 0 to 180 degrees
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))

    # Pre-compute sine and cosine values of the angles
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)

    # Initialize accumulator matrix
    accumulator = np.zeros((2*max_rho, len(thetas)), dtype=np.uint64)

    # Iterate through each pixel in the image
    for y in range(height):
        for x in range(width):
            # Check if the pixel is part of an edge (e.g., if its value is non-zero)
            if image[y, x] > 50:
                # Iterate through each theta value
                for t_idx in range(len(thetas)):
                    # Calculate rho
                    rho = int(x * cos_thetas[t_idx] + y * sin_thetas[t_idx])
                    # Increment the corresponding accumulator cell
                    accumulator[rho+max_rho, t_idx] += 1

    return accumulator

def detect_peaks(accumulator, threshold, min_distance):
    peaks = []

    height, width = accumulator.shape

    # Iterate over the accumulator array
    for i in range(height):
        for j in range(width):
            # Check if the current cell value is greater than the threshold
            if accumulator[i, j] > threshold:
                # Check if the current cell is a local maximum
                is_max = True
                for dx in range(-min_distance, min_distance + 1):
                    for dy in range(-min_distance, min_distance + 1):
                        # Skip the central pixel
                        if dx == 0 and dy == 0:
                            continue
                        x = i + dx
                        y = j + dy
                        if x >= 0 and x < height and y >= 0 and y < width:
                            if accumulator[i, j] < accumulator[x, y]:
                                is_max = False
                                break
                    if not is_max:
                        break
                if (is_max and i != height/2) and abs(j-90) != 45:
                    peaks.append((i-height/2, j-90))
    return peaks

def draw_lines(image, rhos, thetas):
    lines_image = np.copy(image)  # Create a copy of the image to draw lines on

    for rho, theta in zip(rhos, thetas):
        # Convert polar coordinates to cartesian coordinates
        x0 = int(rho * np.cos(theta))
        y0 = int(rho * np.sin(theta))
        x1 = int(x0 + 1000 * (-np.sin(theta))) 
        y1 = int(y0 + 1000 * (np.cos(theta)))  
        x2 = int(x0 - 1000 * (-np.sin(theta)))  
        y2 = int(y0 - 1000 * (np.cos(theta)))  

        # Draw the line
        cv2.line(lines_image, (x1, y1), (x2, y2), (255, 255, 255), 1)

    return lines_image

# Read the input image
image = cv2.imread('/home/jihai/MSAI495/MP6/input.bmp')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image_gaussSmothing = GaussSmoothing(gray_image, 1, 1)

mag, theta = ImageGradient(image_gaussSmothing, 'sobel')

# Perform Hough Transform
accumulator = hough_transform(mag)

plt.imshow(accumulator, cmap='gray')
plt.show()

intersections = detect_peaks(accumulator, 68, 4)
print(intersections)
rhos = [peak[0] for peak in intersections]
thetas = [peak[1] for peak in intersections]
thetas = np.deg2rad(thetas)
# Draw lines on the original image
lines_image = draw_lines(gray_image, rhos, thetas)

cv2.imshow('Lines Detected', lines_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
