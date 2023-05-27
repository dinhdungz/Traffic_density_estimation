# importing the module
import cv2

lanes = []
lane = []
def click_event(event, x, y, flags, params):
	global lanes, lane
	
	# checking for left mouse clicks
	if event == cv2.EVENT_LBUTTONDOWN:
		print(x, ' ', y)
		lane.append((x,y))
		# font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.circle(img, (x,y), 3, (0, 0, 255), -1)
		cv2.imshow('image', img)
		if len(lane) == 4:
			lanes.append(lane)
			# print(f"{lanes} -- 1")
			lane = []

img = cv2.imread('lane.png')

def get_lanes(frame):
	cv2.imshow('image', frame)
	cv2.setMouseCallback('image', click_event)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

get_lanes(img)

def draw_lanes(frame, lanes):
	for lane in lanes:
		top_left = lane[0]
		bottom_left = lane[1]
		top_right = lane[2]
		bottom_right = lane[3]
		cv2.line(frame, top_left, bottom_left, (255,0,0), 2) 
		cv2.line(frame, top_right, bottom_right, (255,0,0), 2)
	cv2.imshow('lane', frame)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

draw_lanes(img, lanes)
