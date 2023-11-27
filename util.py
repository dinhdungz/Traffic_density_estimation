import numpy as np
import cv2
import math
import get_ROI as ROI
import get_BOI as BOI 
import os 
from datetime import datetime
import csv
import matplotlib.pyplot as plt

def create_variable(N_lane, N_block, N):

    BOIs_var = create_3d_list(N_lane, N_block, N)
    BOIs_mean = create_3d_list(N_lane, N_block, N)
    
    # lambda size N_lane X N_block
    lambda_f = [[100 for i in range(N_block)] for j in range(N_lane)]
    lambda_b = [[100 for i in range(N_block)] for j in range(N_lane)]
    p_f = [[0.4 for i in range(N_block)] for j in range(N_lane)]

    lambda_fm = [[30 for i in range(N_block)] for j in range(N_lane)]
    lambda_bm= [[10 for i in range(N_block)] for j in range(N_lane)]
    p_m = [[0.4 for i in range(N_block)] for j in range(N_lane)]

    return BOIs_var, BOIs_mean, lambda_f, lambda_b, p_f, lambda_fm, lambda_bm, p_m


def draw_BOI(img, coords_blocks, block_occupied):
    # draw block
    N_block = len(coords_blocks)
    for i in range(N_block):
        if i in block_occupied:
            cv2.rectangle(img, coords_blocks[i][0], coords_blocks[i][1], (0,0,255), 1)
        else:
            cv2.rectangle(img, coords_blocks[i][0], coords_blocks[i][1], (0,255,0), 1)

def get_mean_variance(img, coords_block):
    # return mean, variance of block
    # Order of coordinate y --> x
    y_min, x_min = coords_block[0]
    y_max, x_max = coords_block[1]
    mean = np.mean(img[x_min: x_max, y_min: y_max])
    variance = np.var(img[x_min: x_max, y_min: y_max])
    return mean, variance

def get_Vov(list_var):
    # return variance of variance of block with some frame
    Vov = np.var(list_var)
    return Vov

def update_model(lambda_b , delta_v, lambda_bm, delta_m):
    # update lambda b, lambda bm, g_mean , g_var
    if delta_v < lambda_b:
        lr_b = 0.01
    else:
        lr_b = 0.05
    
    lambda_b = (1 - lr_b) * lambda_b + lr_b* delta_v
    if lambda_b < 100:
        lambda_b = 100
    elif lambda_b > 500:
        lambda_b = 500
    
    if delta_m < lambda_bm:
        lr_bm = 0.01
    else:
        lr_bm = 0.1

    lambda_bm = (1 - lr_bm) * lambda_bm + lr_bm* delta_m
    if lambda_bm < 10:
        lambda_bm = 10
    elif lambda_bm > 40:
        lambda_bm = 40

    return lambda_b, lambda_bm

def update_p(occupied, p_f, direct):
    # upadate prior probability
    if direct == 'front':
        for pos in occupied:
            p_f[pos] = 0.5
            if pos + 1 < len(p_f):
                p_f[pos + 1] = 0.6
        p_f[0] = 0.5
        
    else:
        for pos in occupied:
            p_f[pos] = 0.5
            if pos > 0:
                p_f[pos-1] = 0.6
        p_f[-1] = 0.5
    
    return p_f

def gen_block(frame, N_block, increment):
    # generate blocks
    lanes = ROI.get_lanes(frame)
    lanes_image = ROI.draw_lanes(frame, lanes)

    area = BOI.get_area(lanes)

    boi_image, BOIs_coor = BOI.get_BOI(area, lanes_image,N_block, increment)

    with open("name.csv", "w") as f:
        write = csv.writer(f)
        write.writerows(BOIs_coor)

    cv2.imshow('lane', boi_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return BOIs_coor

def create_3d_list(x, y, z):

    lst = []
    for i in range(x):
        lst_2d = []
        for j in range(y):
            lst_1d = []
            for k in range(z):
                lst_1d.append(0)
            lst_2d.append(lst_1d)
        lst.append(lst_2d)
    return lst

def mean_model(x, mean, var_mean):
    p_x = math.exp((mean - x)/(2*var_mean)) /math.sqrt(2*math.pi*var_mean)

    return p_x

def view_density(frame, density):
    total_rate = (sum(density)/len(density))
    y_coor = 100
    height, width, channels = np.shape(frame)
    for i in range(len(density)):
        cv2.putText(frame, f'Lane {i + 1}: {density[i]}%', (100, y_coor),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2 )
        y_coor += 20
    cv2.putText(frame, f'Total: {int(total_rate)}%', (int(width/2) - 50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 

def setup_video_writer(cap, name):
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps =  cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'{name}.avi', fourcc, fps, (width,height))

    return out

def init_bg(frame, count, N, BOIs_coor, BOIs_var, BOIs_mean):

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.float32)/25
    img = cv2.filter2D(img,-1,kernel)

    N_lane = len(BOIs_coor)
    N_block = len(BOIs_coor[0])
    init_background = True

    if count < N:
        for lane in range(N_lane):
            for i in range(N_block):
                mean, var = get_mean_variance(img, BOIs_coor[lane][i])
                BOIs_var[lane][i][count] = var
                BOIs_mean[lane][i][count] = mean
        count += 1
        cv2.putText(frame, 'Init background', (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2 )
    else:
        pas = True
        for lane in range(N_lane):
            for i in range(N_block):
                Vov = get_Vov(BOIs_var[lane][i])
                if Vov <= 100:
                    continue
                else:
                    pas = False
                    BOIs_var[lane][i].pop(0)
                    BOIs_mean[lane][i].pop(0)
                    mean, var = get_mean_variance(frame, BOIs_coor[lane][i])
                    BOIs_var[lane][i].append(var)
                    BOIs_mean[lane][i].append(mean)
        cv2.putText(frame, 'Init background', (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2 )
        
        if pas:
            init_background = False

    return init_background, count

def  cal_rate(frame, BOIs_coor, N_lane, N_block, g_var, g_mean, lambda_b, lambda_f, lambda_bm,lambda_fm, p_f, p_m, direct, pre_density):

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.float32)/25
    img = cv2.filter2D(img,-1,kernel)

    density = []
    for lane in range(N_lane):
        n = 0
        occupied = []
        for i in range(N_block):
            # cv2.putText(frame, str(i), BOIs_coor[i][0],cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2 )
            mean, var = get_mean_variance(img, BOIs_coor[lane][i])
            delta_v = abs(var - g_var[lane][i])
            delta_m = abs(mean - g_mean[lane][i])

            p_vb = math.exp(-delta_v/lambda_b[lane][i])
            p_vf = 1 - math.exp(-delta_v/lambda_f[lane][i])
            p_fv =  (p_vf * p_f[lane][i])/(p_vb * (1 - p_f[lane][i]) + p_vf * p_f[lane][i])

            p_mb = math.exp(-delta_m/lambda_bm[lane][i])
            p_mf = 1 - math.exp(-delta_m/lambda_fm[lane][i])
            p_fm = (p_mf * p_m[lane][i])/(p_mb * (1 - p_m[lane][i]) + p_mf * p_m[lane][i])
    
            if p_fv > 0.7 :
                n += 1
                occupied.append(i) 
                cv2.putText(frame, f'var: {int(var)} - {int(g_var[lane][i])} lambda {int(lambda_b[lane][i])} - {int(lambda_f[lane][i])}', BOIs_coor[lane][i][1],cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2 )
                if pre_density >= 0.3:
                    lr_f = 0.0002
                else:
                    lr_f = 0.001
                lambda_f[lane][i] = (1 - lr_f) * lambda_f[lane][i] + lr_f * delta_v
                if lambda_f[lane][i] > 2000:
                    lambda_f[lane][i] = 2000
                elif lambda_f[lane][i] < 100:
                    lambda_f[lane][i] = 100

            elif p_fm > 0.7:
                n += 1
                occupied.append(i) #int(var)} - {int(g_var[lane][i])
                # cv2.putText(frame, f'mean: {int(mean)} - {int(g_mean[lane][i])} lambda {int(lambda_bm[lane][i])} - {int(lambda_fm[lane][i])}', BOIs_coor[lane][i][1],cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2 )
                lambda_fm[lane][i] = (1 - 0.01) * lambda_fm[lane][i] + 0.01 * delta_m
                if lambda_fm[lane][i] > 80:
                    lambda_fm[lane][i] = 80
                elif lambda_fm[lane][i] < 20:
                    lambda_fm[lane][i] = 20

            else:
                cv2.putText(frame, f'var: {int(var)} - {int(g_var[lane][i])} lambda {int(lambda_b[lane][i])} - {int(lambda_f[lane][i])}', BOIs_coor[lane][i][1],cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2 )
                lambda_b[lane][i], lambda_bm[lane][i]= update_model(lambda_b[lane][i], delta_v,lambda_bm[lane][i], delta_m)
                if p_f[lane][i] == 0.4:
                    g_mean[lane][i] = (1 - 0.01) * g_mean[lane][i] + 0.01 * mean
                    g_var[lane][i] = (1 - 0.01) * g_var[lane][i] + 0.01 * var

        d_lane = int(n * 100/N_block) 
        density.append(d_lane)
        draw_BOI(frame, BOIs_coor[lane], occupied)
    
        p_f[lane] = [0.4 for i in range(N_block)]
        p_m[lane] = [0.4 for i in range(N_block)]

        p_f[lane] = update_p(occupied, p_f[lane], direct)
        p_m[lane] = update_p(occupied, p_m[lane], direct)

    rate = int(sum(density)/len((density)))

    return rate, density, lambda_b, lambda_f, lambda_bm,lambda_fm, p_f, p_m, g_var, g_mean

def save(rate, name):
    
    data_path = "./Result"
    if name in os.listdir(data_path):
        pass
    else:
        os.mkdir(f"{data_path}/{name}")
    
    time = datetime.now()

    day = time.day
    month = time.month 
    year = time.year 

    hour = time.hour
    minute = time.minute

    filename = f"{data_path}/{name}/{day}_{month}_{year}.csv"
    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
     
        # writing the fields
        csvwriter.writerow([hour, minute, rate])

def  visualize(name, day, hour_start, hour_end, minute_start, minute_end):
    #, hour_start, hour_end, minute_start, minute_end
    data_path = "./Result"
    
    result = []
    with open(f"{data_path}/{name}/{day}", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            if hour_start == hour_end:
               if int(row[0]) == hour_start and int(row[1]) >= minute_start and int(row[1]) <= minute_end:
                   result.append(int(row[2]))
            if hour_start < hour_end:
                if int(row[0]) == hour_start and int(row[1]) >= minute_start:
                    result.append(int(row[2]))
                elif int(row[0]) == hour_end and int(row[1]) <= minute_end:
                    result.append(int(row[2]))
                
                if hour_start < int(row[0]) < hour_end :
                    result.append(int(row[2]))

    plt.plot(result)
    plt.show()


def estimate(video_path, N_block, N, increment, direct = "front"):
    # Read video
    cap = cv2.VideoCapture(video_path)
    out = setup_video_writer(cap, "test_02")

    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()

    init_background = True
    # Coords of all block (lane x N_block x coords)
    BOIs_coor = gen_block(frame, N_block, increment)
    N_lane = len(BOIs_coor)

    BOIs_var, BOIs_mean, lambda_f, lambda_b, p_f, lambda_fm, lambda_bm, p_m = create_variable(N_lane, N_block, N)
    
    count = 0
    n_heavy = 0
    n_medium = 0
    n_light = 0
    n_frame = 0
    desity = 0

    rate_in_minute = []
    t1 = datetime.now().minute

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Initialization background
        if init_background:
            init_background, count = init_bg(frame, count, N, BOIs_coor, BOIs_var, BOIs_mean)
            g_mean = np.mean(BOIs_mean, axis = 2)
            g_var = np.mean(BOIs_var, axis = 2)

        else:
            rate, density, lambda_b, lambda_f, lambda_bm,lambda_fm, p_f, p_m, g_var, g_mean = cal_rate(frame, BOIs_coor, N_lane, N_block, g_var, g_mean, lambda_b, lambda_f, lambda_bm,lambda_fm, p_f, p_m, direct, desity)
            
            rate_in_minute.append(rate)
            if n_frame % 20 == 0:
                v_density = density
            view_density(frame, v_density)
            
            n_frame += 1
            if rate > 65:
                n_heavy += 1
            elif 40 <= rate <= 65:
                n_medium += 1
            else:
                n_light += 1

            t2 = datetime.now().minute
            if t2 > t1:
                rate_mean = int(np.mean(rate_in_minute))
                save(rate_mean, "test_02")
                rate_in_minute = []
                t1 = t2
                if t1 == 59:
                    t1 = -1

        cv2.imshow("video",frame)
        out.write(frame)
        if cv2.waitKey(1) == ord('q'):
            break
 
    if n_frame == 0:
        return 
    
    if max(n_light, n_medium, n_heavy) == n_heavy:
        result = 'heavy'
    elif n_medium + n_heavy > n_light:
        result = 'medium'
    else:
        result = 'light'

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return result

if __name__ == "__main__":
    visualize("test", "9_10_2023.csv", 9, 10, 36, 20)