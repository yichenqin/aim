# %%
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
# %%
cwd = os.getcwd()
print("Current working directory:", cwd)
# %%
def mark_cue_ball(image):
    """
    Takes an image of a pool table (NumPy array) and returns it with the cue ball marked by a green circle.
    
    Args:
        image (np.ndarray): Input image in BGR format (e.g., from cv2.imread()).
    
    Returns:
        np.ndarray: The input image with the cue ball marked.
    """
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Apply median blur to reduce noise
    blurred = cv2.medianBlur(gray, 5)
    
    # Step 3: Detect circles using HoughCircles
    # Parameters: dp=1 (resolution), minDist=100 (min distance between circles),
    # param1=50 (edge threshold), param2=30 (circle accumulator threshold),
    # minRadius=10, maxRadius=50 (tune based on ball size in pixels)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
        param1=50, param2=30, minRadius=10, maxRadius=50
    )
    
    if circles is not None:
        # Convert circles to integers
        circles = np.round(circles[0, :]).astype("int")
        
        # Step 4: Find the brightest circle (cue ball)
        max_intensity = 0
        best_circle = None
        for (x, y, r) in circles:
            # Extract region of interest (ROI) for this circle
            roi = gray[max(0, y-r):min(gray.shape[0], y+r), max(0, x-r):min(gray.shape[1], x+r)]
            if roi.size > 0:  # Ensure ROI is valid
                intensity = np.mean(roi)
                if intensity > max_intensity:
                    max_intensity = intensity
                    best_circle = (x, y, r)
        
        # Step 5: Mark the best circle if found
        if best_circle is not None:
            x, y, r = best_circle
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)  # Green circle, thickness 4
    
    return image
# %%
image_path = 'C:\\Users\\qyc\\Downloads\\IMG_4761.jpg'
image = cv2.imread(image_path)
marked_image = mark_cue_ball(image)

# %%
# Save or display the result
cv2.imwrite('marked_pool_table.jpg', marked_image)
cv2.imshow('Marked Image', marked_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
# %%
type(image)
# %%
image.shape
# %%
img = cv2.imread(image_path)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# %%
# generate histograms for each channel
plt.figure(figsize=(12, 4)) 
colors = ('b', 'g', 'r')
channel_ids = (0, 1, 2)
for channel_id, color in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
        img[:, :, channel_id], bins=256, range=(0, 256)
    )
    plt.plot(bin_edges[0:-1], histogram, color=color)
# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
channels = ['Blue', 'Green', 'Red']
for i in range(3):
    axs[i].imshow(img[:, :, i], cmap='gray')
    axs[i].set_title(f'{channels[i]} Channel')
    axs[i].axis('off')
plt.show()

# %%
img2 = img.copy()
img2[1500:1550,:,0] = 155
img2[:,2500:2550,0] = 155

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
channels = ['Blue', 'Green', 'Red']
for i in range(3):
    axs[i].imshow(hsv[:, :, i], cmap='gray')
    axs[i].set_title(f'{channels[i]} Channel')
    axs[i].axis('off')
plt.show()
# %%
hsv[1500:1550,2500:2550,0].max()
hsv[1500:1550,2500:2550,0].min()

# %%

lower_white = np.array([20-5,  19-5, 112])
upper_white = np.array([22+5, 30+5, 174])
#mask = cv2.inRange(img, lower_white, upper_white)
mask0 = np.logical_and((hsv[:,:,0] > lower_white[0]),
               (hsv[:,:,0] < upper_white[0]))
mask1 = np.logical_and((hsv[:,:,1] > lower_white[1]),
               (hsv[:,:,1] < upper_white[1]))
mask2 = np.logical_and((hsv[:,:,2] > lower_white[2]),
               (hsv[:,:,2] < upper_white[2]))
plt.imshow(mask0, cmap='viridis')
plt.colorbar(label='Intensity')
plt.imshow(mask1, cmap='viridis')
plt.colorbar(label='Intensity')
plt.imshow(mask2, cmap='viridis')
plt.colorbar(label='Intensity')


# %%
import cv2
import numpy as np

def mark_cue_ball(image_path, show_result=False):
    # 1. Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # 2. Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 3. Define white color range (tune if needed)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 4. Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    # 5. Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No white objects found.")
        return img
    
    # 6. Select the most circular and largest contour (likely the cue ball)
    cue_ball_contour = None
    max_score = 0
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:  # ignore very small objects
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        # circularity near 1 → perfect circle
        score = circularity * area
        if score > max_score:
            max_score = score
            cue_ball_contour = c
    
    if cue_ball_contour is not None:
        (x, y), radius = cv2.minEnclosingCircle(cue_ball_contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # 7. Draw circle and center mark
        cv2.circle(img, center, radius, (0, 0, 255), 3)
        cv2.circle(img, center, 3, (255, 0, 0), -1)
    else:
        print("Cue ball not found.")
    
    # 8. Show or return result
    if show_result:
        cv2.imshow("Cue Ball Marked", img)
    
    return img
result = mark_cue_ball("C:\\Users\\qyc\\Downloads\\IMG_4761.jpg", show_result=True)
#cv2.imwrite("pool_table_marked.jpg", result)
# %%
result.shape
# %%
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
channels = ['Blue', 'Green', 'Red']
for i in range(3):
    axs[i].imshow(result[:, :, i], cmap='gray')
    axs[i].set_title(f'{channels[i]} Channel')
    axs[i].axis('off')
plt.show()
# %%
image_path="C:\\Users\\qyc\\Downloads\\IMG_4761.jpg"
# %%
import cv2
import numpy as np

def detect_pool_table_surface(image_path):
    # Load image
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define green color range (tune as needed)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create mask for green felt
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found.")
        return img, None

    # Get the largest contour (table surface)
    table_contour = max(contours, key=cv2.contourArea)

    # Draw contour on original image
    output = img.copy()
    cv2.drawContours(output, [table_contour], -1, (0, 0, 255), 3)

    # Optionally get corner points
    epsilon = 0.02 * cv2.arcLength(table_contour, True)
    approx = cv2.approxPolyDP(table_contour, epsilon, True)
    for point in approx:
        cv2.circle(output, tuple(point[0]), 10, (255, 0, 0), -1)

    return output, approx

# %%
result, corners = detect_pool_table_surface(image_path)
cv2.imshow("Detected Table Surface", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%

import cv2x`x```
import numpy as np

def detect_pool_balls(image_path, debug=False):
    # 1. Load and preprocess the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Reduce noise
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)

    # 3. Edge detection (helps with circle detection)
    edges = cv2.Canny(gray_blur, 50, 150)
    if debug:
        cv2.imshow("Detected Edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # 4. Detect circles using Hough Transform
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,         # Minimum distance between detected centers
        param1=100,         # Canny upper threshold
        param2=30,          # Detection threshold (lower = more sensitive)
        minRadius=8,        # Adjust for your image scale
        maxRadius=25
    )

    result = img.copy()
    ball_centers = []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y, r = i
            ball_centers.append((x, y, r))
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)  # outline
            cv2.circle(result, (x, y), 2, (0, 0, 255), 3)  # center

    if debug:
        cv2.imshow("Detected Balls", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result, ball_centers
# %%
image_path="C:\\dropbox_thinkpad_local_account\\Dropbox\\temp\\1760152598834.jpg"
result_img, balls = detect_pool_balls(image_path, debug=True)
print(f"Detected {len(balls)} balls:")
for (x, y, r) in balls:
    print(f"Ball at ({x}, {y}) with radius {r}")
# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
channels = ['Blue', 'Green', 'Red']
for i in range(3):
    axs[i].imshow(result[:, :, i], cmap='gray')
    axs[i].set_title(f'{channels[i]} Channel')
    axs[i].axis('off')
plt.show()

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_balls_by_edges(image_path, debug=False):
    # 1. Load and preprocess
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # 2. Edge detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 3. Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img.copy()
    ball_contours = []

    # 4. Analyze contours
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:  # skip small noise
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if 0.7 < circularity <= 1.2:  # good circular shapes
            # Fit a circle to the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if 60 < radius < 100:  # filter out unlikely sizes
                center = (int(x), int(y))
                radius = int(radius)
                ball_contours.append((center, radius))
                cv2.circle(result, center, radius, (0, 255, 0), 2)
                cv2.circle(result, center, 2, (0, 0, 255), -1)

    if debug:s
        # cv2.imshow("Gray", gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imshow("Edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("Detected Balls", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result, ball_contours, contours, edges
# %%
image_path="C:\\dropbox_thinkpad_local_account\\Dropbox\\temp\\1760152598834.jpg"
result, balls, contours, edges = detect_balls_by_edges(image_path, debug=True)
print(f"Detected {len(balls)} balls.")
for (center, radius) in balls:
    print(f"Ball at {center} with radius {radius}")
# %%
# adjust figure size for better visibility
plt.figure(figsize=(10, 8))
plt.imshow(edges)
# %%
type(contours[0])

# %%
contours[2].shape
# %%
contours[1].shape
# %%
plt.scatter(contours[1][:,0,0], contours[1][:,0,1])
# %%
plt.imshow(edges,'viridis')
plt.colorbar(label='Intensity')
# %%

# %%
import cv2
import numpy as np

def detect_pool_balls_robust(image_path,
                             min_area=200, max_area=3000,
                             circularity_thresh=0.7,
                             hough_dp=1.2, hough_min_dist=20,
                             hough_param1=50, hough_param2=15,
                             min_radius=5, max_radius=30):
    """
    Robust pool ball detection using edges, handling incomplete edges.

    Args:
        image_path (str): Path to the input image.
        min_area (int): Minimum contour area.
        max_area (int): Maximum contour area.
        circularity_thresh (float): Circularity threshold for contours.
        hough_dp (float): Hough circle dp parameter.
        hough_min_dist (float): Minimum distance between circle centers in Hough.
        hough_param1 (int): Higher threshold for Canny inside Hough.
        hough_param2 (int): Accumulator threshold for circle detection in Hough.
        min_radius (int): Minimum radius for Hough circles.
        max_radius (int): Maximum radius for Hough circles.

    Returns:
        result_img (np.ndarray): Image with detected balls drawn.
        final_circles (list of (x, y, r)): Detected ball coordinates and radii.
    """
    
    # 1️⃣ Load image and preprocess
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1)
    
    # 2️⃣ Edge detection
    edges = cv2.Canny(gray, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 3️⃣ Morphological closing to connect broken edges
    kernel = np.ones((3, 3), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 4️⃣ Contour-based detection
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_circles = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > circularity_thresh:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            contour_circles.append((int(x), int(y), int(radius)))
    
    # 5️⃣ Hough Circle detection on edge map
    hough_circles = []
    circles = cv2.HoughCircles(edges_closed, cv2.HOUGH_GRADIENT, dp=hough_dp, minDist=hough_min_dist,
                               param1=hough_param1, param2=hough_param2,
                               minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            hough_circles.append((int(x), int(y), int(r)))
    
    # 6️⃣ Merge contour and Hough detections (avoid duplicates)
    final_circles = contour_circles.copy()
    for hc in hough_circles:
        duplicate = False
        for fc in final_circles:
            dist = np.sqrt((hc[0]-fc[0])**2 + (hc[1]-fc[1])**2)
            if dist < max(hc[2], fc[2]):  # overlapping, consider duplicate
                duplicate = True
                break
        if not duplicate:
            final_circles.append(hc)
    
    # 7️⃣ Draw detected balls
    for (x, y, r) in final_circles:
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)  # circle
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)  # center

    return img, final_circles


# Example usage:
image_path="C:\\dropbox_thinkpad_local_account\\Dropbox\\temp\\1760152598834.jpg"
result_img, balls = detect_pool_balls_robust(image_path)
cv2.imshow("Detected Pool Balls", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
plt.figure(figsize=(10, 8))
plt.imshow(edges_closed, 'viridis')
plt.scatter(contours_circle[:,0], contours_circle[:,1], color='red')
# plot a scatter plot using yellow color for the first contour
for i in range(len(contours)):
    plt.scatter(contours[i][:,0,0],contours[i][:,0,1], color='yellow')

# %%
# convert contour_circles to array
contours_circle = np.array(contour_circles)

# %%
import cv2
import numpy as np

def mark_pool_balls(image_path, output_path=None, min_radius=5, max_radius=30, param1=50, param2=30, min_distance=50):
    """
    Detects and marks the locations of pool balls in an image of a pool table using circle detection.
    
    Args:
        image_path (str): Path to the input image file.
        output_path (str, optional): Path to save the marked image. If None, displays the image instead.
        min_radius (int): Minimum radius of detected circles (default: 5).
        max_radius (int): Maximum radius of detected circles (default: 30).
        param1 (int): Higher threshold for the Canny edge detector (default: 50).
        param2 (int): Accumulator threshold for the Hough circles (default: 30).
        min_distance (int): Minimum distance between the centers of detected circles (default: 50).
    
    Returns:
        list: List of tuples (x, y, radius) for the detected ball centers and radii.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_distance,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    # Convert circles to integers and draw them if detected
    marked_image = image.copy()
    detected_balls = []
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Draw the circle outline
            cv2.circle(marked_image, (x, y), r, (0, 255, 0), 50)
            # Draw a small circle at the center
            cv2.circle(marked_image, (x, y), 3, (0, 0, 255), -1)
            detected_balls.append((x, y, r))
    
    # Save or display the result
    if output_path:
        cv2.imwrite(output_path, marked_image)
        print(f"Marked image saved to {output_path}")
    else:
        cv2.imshow("Marked Pool Balls", marked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return detected_balls

# %%
# Example usage:
import os
os.getcwd()
balls = mark_pool_balls("C:\\dropbox_thinkpad_local_account\\Dropbox\\temp\\1760152598834.jpg", output_path='marked_pool_table.jpg')
print("Detected ball positions:", balls)

# %%
