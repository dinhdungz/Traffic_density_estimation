import cv2
import time
import numpy as np

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
# print(np.shape(img))
width, height = np.shape(img)
mean = 112
sum_var = 0
t1 = time.time()
for i in range(width):
    for j in range(height):
        sum_var += (img[i, j] - mean)**2
var = sum_var/(width*height)

t2 = time.time()
var1 = np.var(img)
t3 = time.time()

print(f"var loop - {var}")
print(f"var numpy - {var1}")
print(f"time loop - {t2-t1}")
print(f"time numpy - {t3-t2}")
