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

### 3. Result
Each frame is classified as light, medium, or heavy. We choose the category to which the maximum number of frames from the video sequence has been classified.


https://github.com/dinhdungz/traffic_density_estimation/assets/102871487/96ef1a1b-892d-42dc-8aef-e36cd3913448

## How to use
1. Run main.py with parameters
2. Select lane

## Reference
1. Garg, K., Lam, S.-K., Srikanthan, T., & Agarwal, V. (2016). Real-time road traffic density estimation using block variance. 2016 IEEE Winter Conference on Applications of Computer Vision (WACV). doi:10.1109/wacv.2016.7477607
2. Garg, K., Ramakrishnan, N., Prakash, A., & Srikanthan, T. (2019). Rapid and Robust Background Modeling Technique for Low-Cost Road Traffic Surveillance Systems. IEEE Transactions on Intelligent Transportation Systems, 1–12. doi:10.1109/tits.2019.2917560
