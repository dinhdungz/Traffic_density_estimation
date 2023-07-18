# Traffic density estimation
The estimation of road traffic density plays a vital role in transportation systems, as it provides valuable insights for traffic management, infrastructure planning, and congestion control. Accurate estimation of traffic density enables authorities to make informed decisions, optimize traffic flow, and improve overall road safety. Traffic density estimation based on block variance is a simple but highly accurate approach.

## Process
### 1. Select lane
Select 4 points of lane in order top left --> bottom left --> top right --> bottom right. If the road has more than 2 lanes, choose in order (top left, bottom left) from left to right

### 2. Block generate
There are a few parameters you need to pay attention to:
1. Path of video
2. Number of blocks for each lane
3. Number of frames for background initialization
4. Increment of blocks (Because distant vehicles are small and nearby are large, the blocks in the back need to be larger than the previous ones in a certain ratio)

### Result
Each frame is classified as light, medium, or heavy. We choose the category to which the maximum number of frames from the video sequence has been classified.
