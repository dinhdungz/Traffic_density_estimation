import numpy as np
import cv2
import imutils
import math
# img = cv2.imread('BOI_w_750.jpg', cv2.IMREAD_GRAYSCALE)
cap = cv2.VideoCapture('highway.mp4')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (750,421))
N = 4
init_background = True
BOIs_coor = [[(329,64), (386, 98)], [(296, 130), (386,168)], [(270, 181),(389, 239)]]
count = 0
BOIs_var = [[0 for i in range(N)] for j in range(len(BOIs_coor))]
BOIs_mean = [[0 for i in range(N)] for j in range(len(BOIs_coor))]
s = 0
lambda_f = [100 for i in range(len(BOIs_coor))]
lambda_b = [100 for i in range(len(BOIs_coor))]
lr_f = [0.01 for i in range(len(BOIs_coor))]
lr_b = [0.01 for i in range(len(BOIs_coor))]
lr_am = [0.01 for i in range(len(BOIs_coor))]

def draw_BOI(img, coordinates):
    for block in coordinates:
        cv2.rectangle(img, block[0], block[1], (0,255,0), 2)

def get_mean_variance(img, coordinate_block):
    x_min, y_min = coordinate_block[0]
    x_max, y_max = coordinate_block[1]
    sum_mean = 0
    sum_var = 0
    for i in range(x_min, x_max +1):
        for j in range(y_min, y_max+1):
            sum_mean += img[i, j]
    mean = sum_mean/((x_max - x_min)*(y_max - y_min))

    for i in range(x_min, x_max +1):
        for j in range(y_min, y_max+1):
            sum_var += (img[i, j] - mean)**2
    
    variance = sum_var/((x_max - x_min)*(y_max - y_min))
    return mean, variance

def get_Vov(list_var):
    sum_mean = sum(list_var)
    sum_var = 0

    mean = sum_mean/(len(list_var))

    for var in list_var:
        sum_var += (var - mean)**2
    Vov = sum_var/(len(list_var))
    return Vov

def proba_mean(x, mean, var):
    return math.exp((mean - x)/(2*var))/math.sqrt(2*math.pi*var)

def update_model(i, delta_v, mean):
    if delta_v < lambda_b[i]:
        lr_b[i] = 0.01
    else:
        lr_b[i] = 0.1
    lambda_b[i] = (1 - lr_b[i]) * lambda_b[i] + lr_b[i] * delta_v
    if proba_mean(mean, BOIs_mean[i][0], BOIs_var[i][0]) > proba_mean(BOIs_mean[i][0] + 3*BOIs_var[i][0], BOIs_mean[i][0], BOIs_var[i][0]):
        lr_am[i] = 0.01
    else:
        lr_am[i] = 0.1
    
    BOIs_mean[i][0] = (1 - lr_am[i]) * BOIs_mean[i][0] + lr_am[i] * mean
    BOIs_var[i][0] = (1 - lr_am[i]) * BOIs_var[i][0] + lr_am[i]*(mean - BOIs_mean[i][0])**2



while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = imutils.resize(gray, width = 750)
    
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
            cv2.putText(img, 'Init background', (40, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2 )
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
                    cv2.putText(img, 'Init background', (40, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2 )

            if pas:
                init_background = False
                print("Done init background")
                cv2.putText(img, 'Done init background', (40, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2 )

    else:
        
        for i in range(len(BOIs_coor)):
            mean, var = get_mean_variance(img, BOIs_coor[i])
            delta_v = abs(var - BOIs_var[i][0])
            p_vb = math.exp(-delta_v/lambda_b[i])
            p_vf = 1 - math.exp(-delta_v/lambda_f[i])
            p_fv = p_vf/(p_vf + p_vb)
            p_m = proba_mean(mean, BOIs_mean[i][0], BOIs_var[i][0])
            if p_fv > 0.7 :
                n_o += 1
                lambda_f[i] = (1 - lr_f[i]) * lambda_f[i] + lr_f[i] * delta_v
            elif p_fv < 0.5:
                update_model(i, delta_v, mean)
                if get_Vov(BOIs_var[i]) < 100:
                    BOIs_var[i][0] = 0.01 * var + (1 - 0.01)*BOIs_var[i][0]
                    BOIs_mean[i][0] = 0.01 * var + (1 - 0.01)*BOIs_mean[i][0]
            else:
                if p_fv > 0.5 and proba_mean(mean, BOIs_mean[i][0], BOIs_var[i][0]) > proba_mean(BOIs_mean[i][0] + 3*BOIs_var[i][0], BOIs_mean[i][0], BOIs_var[i][0]):
                    n_o +=1
                else:
                    update_model(i, delta_v, mean)
        rate = round(n_o/len((BOIs_coor)), 2)
        cv2.putText(img, f'{rate * 100}%', (40, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2 )
    draw_BOI(img, BOIs_coor)
    cv2.imshow("video",img)
    # out.write(img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
