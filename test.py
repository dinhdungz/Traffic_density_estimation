import util
import os

path = "./data/medium"
videos = os.listdir(path)

n_heavy = 0
n_medium = 0
n_light = 0

heavy_fail = []
light_fail = []
for i in range(len(videos)):
    result = util.estimate(f'{path}/{videos[i]}',20, 6, 2)
    if result == 'heavy':
        n_heavy +=1
        heavy_fail.append(videos[i])
    elif result == "medium":
        n_medium +=1
        
    else:
        n_light += 1
        light_fail.append(videos[i])


print(f"heavy - {n_heavy}")
print(f"medium - {n_medium}")
print(f"light - {n_light}")
print(f"heavy_fail - {heavy_fail}")
print(f"light_fail - {light_fail}")




