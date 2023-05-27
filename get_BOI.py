import cv2

img = cv2.imread('lane.png')

ROI = [(404, 362), (168, 559), (470, 369), (310, 585)]
top_left = ROI[0]
bottom_left = ROI[1]
top_right = ROI[2]
bottom_right = ROI[3]

cv2.line(img, top_left, bottom_left, (255,0,0), 2) 
cv2.line(img, top_right, bottom_right, (255,0,0), 2) 
# cv2.rectangle(img, (209, 527), (335, 555), (0,255,0), 1)
cv2.imshow('lane', img)
cv2.waitKey(0)
cv2.destroyAllWindows()