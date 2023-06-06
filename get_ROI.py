import cv2

def click_event(event, x, y, flags, params):
	lanes = params['lanes']
	lane = params['lane']
	img = params['image']

	if event == cv2.EVENT_LBUTTONDOWN:
		lane.append((x, y))
		cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
		cv2.imshow('image', img)
		if len(lane) == 4:
			lanes.append(lane)
			lane = []

def get_lanes(frame):
	cv2.imshow('image', frame)
	lanes = []
	lane = []
	params = {'lanes': lanes, 'lane': lane, 'image': frame}
	cv2.setMouseCallback('image', click_event, params)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# Convert lane coordinates to nested list format
	lanes_nested = [lane[i:i+4] for i in range(0, len(lanes[0]), 4)]

	return lanes_nested

def draw_lanes(frame, lanes):
	print(lanes) 
	for lane in lanes:
		top_left = lane[0]
		bottom_left = lane[1]
		top_right = lane[2]
		bottom_right = lane[3]
		cv2.line(frame, top_left, bottom_left, (255,0,0), 2) 
		cv2.line(frame, top_right, bottom_right, (255,0,0), 2)
	return frame
