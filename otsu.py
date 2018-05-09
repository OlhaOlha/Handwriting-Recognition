import cv2
from matplotlib import pyplot as plt
import os

from PIL import Image, ImageFont, ImageDraw

# Folder with original images
pathIn = "/Users/olgamulava/Documents/Uni/5Groningen/HandwritingRecognition/Code/ImageData/"

# Folders for output
pathOut1 = "/Users/olgamulava/Documents/Uni/5Groningen/HandwritingRecognition/Code/OutputOTSU/"
pathOut2 = "/Users/olgamulava/Documents/Uni/5Groningen/HandwritingRecognition/Code/OutputGAU/"
pathOut3 = "/Users/olgamulava/Documents/Uni/5Groningen/HandwritingRecognition/Code/OutputMEAN/"

# # Create a graph for one image P344-Fg001-R-C01-R01-fused.jpg
# img = cv2.imread(pathIn + "P344-Fg001-R-C01-R01-fused.jpg", 0)  # 0 - grayscale
# cv2.imshow('My image', img)
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


# Open the folder with input images
listing = os.listdir(pathIn)

for file in listing:
    if not file.startswith('.'):  # check for hidden files

        # Read an image
        img = cv2.imread(pathIn + file, 0)

        # OTSU Binarization
        ret, imgi = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(pathOut1 + file, imgi)

        # GAUSSIAN Binarization
        imgi = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(pathOut2 + file, imgi)

        # MEAN Binarization
        imgi = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(pathOut3 + file, imgi)


image1 = cv2.imread(pathOut1 + "P344-Fg001-R-C01-R01-fused.jpg", 0)
# cv2.imshow('My image', image1)
# cv2.waitKey()

# Template is one char
template = cv2.imread("CorrectData/" + "Bet.png", 0)
w, h = template.shape[::-1]

# finding the location of a template image in a larger image
res = cv2.matchTemplate(image1, template, cv2.TM_CCORR_NORMED)

# n
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(image1, top_left, bottom_right, (100, 100, 50), 2)
cv2.imwrite("my.png", image1)
# cv2.imshow("TM_CCORR_NORMED", image1)
cv2.waitKey()

print(top_left, bottom_right)

# # plt.subplot(121)
# # plt.imshow(res, cmap='gray')
# plt.xticks([])
# plt.yticks([])
# # plt.subplot(122)
# plt.imshow(image1, cmap='gray')
#
# plt.show()
