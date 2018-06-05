import time
import cv2

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

#clf = joblib.load("digits_cls.pkl")    #mnist trained classifier
img = cv2.imread('C:\\Users\\Janak\\Desktop\\Handwriting Recognition\\Input\\d1cde2cb-75bc-432f-a7eb-521a99824939.jpg', 0)
winW, winH = (22, 40)

blur = cv2.GaussianBlur(img, (5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

for (x, y, window) in sliding_window(img, stepSize=10, windowSize=(winW, winH)):
    if (window.shape[0] != winH or window.shape[1] != winW):
        continue

    # since we do not have a classifier, we'll just draw the window
    clone = img.copy()
    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    cv2.imshow("Window", clone)
    cv2.waitKey(1)
    time.sleep(0.95)