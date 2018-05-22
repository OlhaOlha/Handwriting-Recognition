import cv2
from matplotlib import pyplot as plt
import os
import numpy as np

from PIL import Image, ImageFont, ImageDraw

# Folder with original images
pathIn = "C:\\Users\\Janak\\Desktop\\Handwriting Recognition\\image-data\\"

# Folders for output
pathOut1 = "C:\\Users\\Janak\\Desktop\\Handwriting Recognition\\Output\\OTSU\\"
pathOut2 = "C:\\Users\\Janak\\Desktop\\Handwriting Recognition\\Output\\GAU\\"
pathOut3 = "C:\\Users\\Janak\\Desktop\\Handwriting Recognition\\Output\\MEAN\\"
pathOut4 = "C:\\Users\\Janak\\Desktop\\Handwriting Recognition\\Output\\Inverse\\"
pathOut5 = "C:\\Users\\Janak\\Desktop\\Handwriting Recognition\\Output\\Sauvola\\"

pathOut7 = "C:\\Users\\Janak\\Desktop\\Handwriting Recognition\\Output\\Tozero\\"
pathOut8 = "C:\\Users\\Janak\\Desktop\\Handwriting Recognition\\Output\\Triangle\\"
pathOut9 = "C:\\Users\\Janak\\Desktop\\Handwriting Recognition\\Output\\Trunc\\"

# # Create a graph for one image P344-Fg001-R-C01-R01-fused.jpg
# img = cv2.imread(pathIn + "P344-Fg001-R-C01-R01-fused.jpg", 0)  # 0 - grayscale
# # cv2.imshow('My image', img)
#
# ret, imgBin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
# # cv2.imwrite('pathOut1/otsuFirst.png', imgBin)
#
# # Original image
# plt.subplot(3, 1, 1)
# plt.imshow(img, cmap='gray')
# plt.title('Original Image')
# plt.xticks([])
# plt.yticks([])
#
# # Histogram
# plt.subplot(3, 1, 2)
# plt.hist(img.ravel(), 256)
# plt.axvline(x=ret, color='r', linestyle='dashed', linewidth=2)
# plt.title('Histogram')
# plt.xticks([])
# plt.yticks([])
#
# # Binarized image
# plt.subplot(3, 1, 3)
# plt.imshow(imgBin, cmap='gray')  # show binarized image
# plt.title('Otsu thresholding')
# plt.xticks([])
# plt.yticks([])
# plt.savefig(pathOut1 + "/otsuFirstGraph.png", dpi=500)
# plt.show()

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200


# Filter by Area.
params.filterByArea = True
params.minArea = 1500

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create()

# Open the folder with input images
listing = os.listdir(pathIn)

for file in listing:
    if not file.startswith('.'):  # check for hidden files

        # Read an image
        img = cv2.imread(pathIn + file, 0)

        # OTSU Binarization
        ret, imgi = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #Detectblobs.
        keypoints = detector.detect(imgi)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(imgi, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show blobs
       # cv2.imshow("Keypoints", im_with_keypoints)
        #cv2.waitKey(0)
        cv2.imwrite(pathOut1 + file, im_with_keypoints )
#
        # GAUSSIAN Binarization
        imgi = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(pathOut2 + file, imgi)

        # MEAN Binarization
        imgi = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(pathOut3 + file, imgi)

        # Inverse Binarization
        ret, imgi = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imwrite(pathOut4 + file, imgi)

        # Sauvola Binarization
        imgi = cv2.adaptiveThreshold(img, 255, cv2.CALIB_CB_ADAPTIVE_THRESH , cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(pathOut5 + file, imgi)


        #tozero
        ret, imgi = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
        cv2.imwrite(pathOut7 + file, imgi)

        #triangle
        ret, imgi = cv2.threshold(img, 0, 255, cv2.THRESH_TRIANGLE + cv2.THRESH_BINARY)
        cv2.imwrite(pathOut8 + file, imgi)

        #trunc
        ret, imgi = cv2.threshold(img, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
        cv2.imwrite(pathOut9 + file, imgi)

