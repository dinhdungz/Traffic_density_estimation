import get_ROI as ROI
import get_BOI as BOI
import cv2

img = cv2.imread('lane.png')
lanes = ROI.get_lanes(img)
lanes_image = ROI.draw_lanes(img, lanes)

area = BOI.get_area(lanes)

area_image = BOI.draw_lanes(lanes_image, area)

cv2.imshow('lane', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()