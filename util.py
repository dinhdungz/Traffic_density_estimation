import numpy as np
import cv2
import math
import get_ROI as ROI
import get_BOI as BOI 

def draw_BOI(img, coordinates, occupied):
    # draw block
    for i in range(len(coordinates)):
        if i in occupied:
            cv2.rectangle(img, coordinates[i][0], coordinates[i][1], (0,0,255), 1)
        else:
            cv2.rectangle(img, coordinates[i][0], coordinates[i][1], (0,255,0), 1)

def get_mean_variance(img, coordinate_block):
    # return mean, variance of block
    # Order of coordinate y --> x
    y_min, x_min = coordinate_block[0]
    y_max, x_max = coordinate_block[1]
    mean = np.mean(img[x_min: x_max, y_min: y_max])
    variance = np.var(img[x_min: x_max, y_min: y_max])
    return mean, variance

def get_Vov(list_var):
    # return variance of variance of block with some frame
    Vov = np.var(list_var)
    return Vov

def mean_model(x, g_mean, g_var):
    # return gau distribution
    p = math.exp(-(x - g_mean)**2/(2*g_var))/(math.sqrt(2*math.pi*g_var))
    return p

def update_model(lambda_b , delta_v, mean, var, g_mean, g_var):
    # update lambda b, lr b
    if delta_v < lambda_b:
        lr_b = 0.001
    else:
        lr_b = 0.01
    if lambda_b < 500:
        lambda_b = (1 - lr_b) * lambda_b + lr_b* delta_v
    
    if mean_model(mean, g_mean, g_var) < mean_model(g_mean + 3*g_var, g_mean, g_var):
        lr_am = 0.01
    else:
        lr_am = 0.1
    
    g_mean = (1 - lr_am)*g_mean + lr_am * mean
    g_var = (1 - lr_am)*g_var + lr_am * var #(mean - g_mean)**2

    return lambda_b, g_mean, g_var

def update_pf(occupied, p_f, direct = 'front'):
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

def gen_boi(frame, N_BOI, increment):
    # generate blocks
    lanes = ROI.get_lanes(frame)
    lanes_image = ROI.draw_lanes(frame, lanes)

    area = BOI.get_area(lanes)

    boi_image, BOIs_coor = BOI.get_BOI(area, lanes_image,N_BOI, increment)
    
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

def view_density(frame, density):
    total_rate = (sum(density)/len(density))
    y_coor = 100
    height, width, channels = np.shape(frame)
    for i in range(len(density)):
        cv2.putText(frame, f'Lane {i + 1}: {density[i]}%', (100, y_coor),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2 )
        y_coor += 20
    cv2.putText(frame, f'Total: {int(total_rate)}%', (int(width/2) - 50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 

def estimate(video_path, N_BOI, N, increment):
    # Read video
    cap = cv2.VideoCapture(video_path)

    # Setup video writer
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps =  cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (width,height))

    # Number frame in init background
    init_background = True

    ret, frame = cap.read()
    ret, frame = cap.read()

    BOIs_coor = gen_boi(frame, N_BOI, increment)

    N_lane = len(BOIs_coor)
    # Mean and var blocks for init background size N x N_ Block
    BOIs_var = create_3d_list(N_lane, N_BOI, N)
    BOIs_mean = create_3d_list(N_lane, N_BOI, N)
    

    # lambda size N_lane X N_block
    lambda_f = [[100 for i in range(N_BOI)] for j in range(N_lane)]
    lambda_b = [[100 for i in range(N_BOI)] for j in range(N_lane)]
    p_f = [[0.4 for i in range(N_BOI)] for j in range(N_lane)]

    # Mean and var for classify object
    g_mean = []
    g_var = []

    v_density = []
    count = 0
    n_heavy = 0
    n_medium = 0
    n_light = 0
    n_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        # frame = imutils.resize(frame, width=800)
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5,5),np.float32)/25
        img = cv2.filter2D(image,-1,kernel)
        

        # Initialization background
        if init_background:
            if count < N:
                for lane in range(N_lane):
                    for i in range(N_BOI):
                        mean, var = get_mean_variance(img, BOIs_coor[lane][i])
                        BOIs_var[lane][i][count] = var
                        BOIs_mean[lane][i][count] = mean
                count += 1
                cv2.putText(frame, 'Init background', (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2 )
            else:
                pas = True
                for lane in range(N_lane):
                    for i in range(N_BOI):
                        Vov = get_Vov(BOIs_var[lane][i])
                        if Vov <= 100:
                            continue
                        else:
                            pas = False
                            BOIs_var[lane][i].pop(0)
                            BOIs_mean[lane][i].pop(0)
                            mean, var = get_mean_variance(img, BOIs_coor[lane][i])
                            BOIs_var[lane][i].append(var)
                            BOIs_mean[lane][i].append(mean)
                cv2.putText(frame, 'Init background', (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2 )
                
                if pas:
                    init_background = False
                    g_mean = np.mean(BOIs_mean, axis = 2)
                    g_var = np.mean(BOIs_var, axis = 2)
                    
                    cv2.putText(frame, 'Done init background', (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2 )

        else:
            # Classify object
            
            density = []
            for lane in range(N_lane):
                n = 0
                occupied = []
                for i in range(N_BOI):
                    # cv2.putText(frame, str(i), BOIs_coor[i][0],cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2 )
                    mean, var = get_mean_variance(img, BOIs_coor[lane][i])
                    delta_v = abs(var - g_var[lane][i])
                    # delta_m = abs(mean - g_mean[lane][i])
                    p_vb = math.exp(-delta_v/lambda_b[lane][i])
                    p_vf = 1 - math.exp(-delta_v/lambda_f[lane][i])
                    p_fv =  (p_vf * p_f[lane][i])/(p_vb * (1 - p_f[lane][i]) + p_vf * p_f[lane][i])
                    if p_fv > 0.7 :
                        n += 1
                        occupied.append(i) #int(var)} - {int(g_var[lane][i])
                        cv2.putText(frame, f'{int(mean)} - {int(g_mean[lane][i])} ', BOIs_coor[lane][i][1],cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2 )
                        if lambda_f[lane][i] < 500:
                            lambda_f[lane][i] = (1 - 0.01) * lambda_f[lane][i] + 0.01 * delta_v
                    elif p_f[lane][i] < 0.5:
                        lambda_b[lane][i], g_mean[lane][i], g_var[lane][i] = update_model(lambda_b[lane][i], delta_v, mean,var, g_mean[lane][i], g_var[lane][i])
                    else:
                        if p_f[lane][i] > 0.5 and (mean_model(mean, g_mean[lane][i], g_var[lane][i] ) < mean_model(g_mean[lane][i] + 3*g_var[lane][i] , g_mean[lane][i], g_var[lane][i]) ) :
                            n +=1
                            # cv2.putText(frame, f'{int(delta_m)}', BOIs_coor[lane][i][1],cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
                            # cv2.putText(frame, f'{int(var)} - {int(g_var[lane][i])}', BOIs_coor[lane][i][1],cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2 )
                            occupied.append(i)
                        else:
                            lambda_b[lane][i], g_mean[lane][i], g_var[lane][i] = update_model(lambda_b[lane][i], delta_v, mean,var, g_mean[lane][i], g_var[lane][i])
                    
                d_lane = int(n * 100/N_BOI) 
                density.append(d_lane)
                draw_BOI(frame, BOIs_coor[lane], occupied)
            
                p_f[lane] = [0.4 for i in range(N_BOI)]
                p_f[lane] = update_pf(occupied, p_f[lane])

            rate = int(sum(density)/len((density)))
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

        cv2.imshow("video",frame)
        out.write(frame)
        if cv2.waitKey(1) == ord('q'):
            break
 
    if n_frame == 0:
        return 
    
    if max(n_light, n_medium, n_heavy) == n_heavy:
        result = 'heavy'
    elif max(n_light, n_medium, n_heavy) == n_medium:
        result = 'medium'
    else:
        result = 'light'

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return result

if __name__ == "__main__":
    pf = [0.5, 0.5, 0.5, 0.5, 0.5]
    occupied = [0,1]
    pf = update_pf(occupied, pf)
    print(pf)