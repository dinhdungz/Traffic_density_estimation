import numpy as np
import cv2
import math

# Imread video
cap = cv2.VideoCapture('1.mp4')

# Setup video writer
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps =  cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width,height))

# Number frame in init background
N = 6 
init_background = True
#Block coordinates
BOIs_coor = [[(492, 356), (511, 403)], [(412, 423), (452,477)], [(210, 573),(265, 670)]]

count = 0
BOIs_var = [[0 for i in range(N)] for j in range(len(BOIs_coor))]
BOIs_mean = [[0 for i in range(N)] for j in range(len(BOIs_coor))]
s = 0
lambda_f = [100 for i in range(len(BOIs_coor))]
lambda_b = [100 for i in range(len(BOIs_coor))]
lr_f = [0.01 for i in range(len(BOIs_coor))]
lr_b = [0.01 for i in range(len(BOIs_coor))]
lr_am = [0.01 for i in range(len(BOIs_coor))]
p_f = [0.4 for i in range(len(BOIs_coor))]
g_mean = []
g_var = []

def draw_BOI(img, coordinates):
    # draw block
    for block in coordinates:
        cv2.rectangle(img, block[0], block[1], (0,255,0), 1)

def get_mean_variance(img, coordinate_block):
    # Get mean, variance of block
    x_min, y_min = coordinate_block[0]
    x_max, y_max = coordinate_block[1]
    mean = np.mean(img[x_min: x_max, y_min: y_max])
    variance = np.var(img[x_min: x_max, y_min: y_max])
    return mean, variance

def get_Vov(list_var):
    # get variance of variance of block with some frame
    Vov = np.var(list_var)
    return Vov

def proba_mean(x, mean, var, lr):
    mean = (1 - lr) * mean + lr * x
    var = (1 - lr) * var + lr * (mean - x)**2 

    return math.exp((mean - x)/(2*var))/math.sqrt(2*math.pi*var)

def update_model(i, delta_v, mean):
    if delta_v < lambda_b[i]:
        lr_b[i] = 0.01
    else:
        lr_b[i] = 0.1
    lambda_b[i] = (1 - lr_b[i]) * lambda_b[i] + lr_b[i] * delta_v
    if proba_mean(mean, BOIs_mean[i][0], BOIs_var[i][0], lr_am[i]) > proba_mean(BOIs_mean[i][0] + 3*BOIs_var[i][0], BOIs_mean[i][0], BOIs_var[i][0], lr_am[i]):
        lr_am[i] = 0.01
    else:
        lr_am[i] = 0.1
    
    # g_mean = (1 - lr_am[i]) * g_mean + lr_am[i] * mean
    # g_var = (1 - lr_am[i]) * g_var + lr_am[i]*(mean - g_mean)**2

def update_p_f(positions, p_f):
    for pos in positions:
        p_f[pos] = 0.5
        if pos + 1 < len(p_f):
            p_f[pos + 1] = 0.6
    p_f[0] = 0.5
    return p_f



while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    s+=1
    n_o = 0
    # Initialization background
    if init_background:
        if count < N:
            for i in range(len(BOIs_coor)):
                mean, var = get_mean_variance(img, BOIs_coor[i])
                BOIs_var[i][count] = var
                BOIs_mean[i][count] = mean
            count +=1
            cv2.putText(frame, 'Init background', (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2 )
            cv2.imshow("video",img)
            out.write(frame)
        else:
            pas = True
            for i in range(len(BOIs_coor)):
                Vov = get_Vov(BOIs_var[i])
                if Vov <= 100:
                    continue
                else:
                    pas = False
                    BOIs_var[i].pop(0)
                    BOIs_mean[i].pop(0)
                    mean, var = get_mean_variance(img, BOIs_coor[i])
                    BOIs_var[i].append(var)
                    BOIs_mean[i].append(mean)
                    cv2.putText(frame, 'Init background', (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2 )
                    cv2.imshow("video",img)
                    out.write(frame)

            if pas:
                init_background = False
                g_mean = np.mean(BOIs_mean, axis = 1)
                g_var = np.mean(BOIs_var, axis = 1)
                print("Done init background")
                cv2.putText(frame, 'Done init background', (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2 )
                cv2.imshow("video",img)
                out.write(frame)

    else:
        positions = []
        for i in range(len(BOIs_coor)):
            mean, var = get_mean_variance(img, BOIs_coor[i])
            delta_v = abs(var - g_var[i])
            p_vb = math.exp(-delta_v/lambda_b[i])
            p_vf = 1 - math.exp(-delta_v/lambda_f[i])
            p_fv =  (p_vf * p_f[i])/(p_vb * (1 - p_f[i]) + p_vf * p_f[i])   # Edit
            p_m = proba_mean(mean, g_mean[i], g_var[i], lr_am[i])
            if p_fv > 0.7 :
                n_o += 1
                positions.append(i)
                lambda_f[i] = (1 - lr_f[i]) * lambda_f[i] + lr_f[i] * delta_v
            elif p_fv < 0.5:
                update_model(i, delta_v, mean)
                if get_Vov(BOIs_var[i]) < 100:
                    g_var[i] = 0.01 * var + (1 - 0.01)*g_var[i]
                    g_mean[i] = 0.01 * var + (1 - 0.01)*g_var[i]
            else:
                if p_fv > 0.5 and proba_mean(mean, g_mean[i], g_var[i],lr_am[i]) > proba_mean(g_mean[i] + 3*g_var[i], g_mean[i], g_var[i], lr_am[i]):
                    n_o +=1
                    positions.append(i)
                else:
                    update_model(i, delta_v, mean)

        p_f = [0.4 for i in range(len(BOIs_coor))]
        p_f = update_p_f(positions, p_f)
        rate = round(n_o/len((BOIs_coor)), 2)
        cv2.putText(frame, f'{rate * 100}%', (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2 )
    draw_BOI(frame, BOIs_coor)
    cv2.imshow("video",img)
    out.write(frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()