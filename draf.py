import numpy as np
import cv2
import math
import get_ROI as ROI
import get_BOI as BOI
import time

# Imread video
cap = cv2.VideoCapture('./Data Road Traffic/normal_3.mp4')

# Setup video writer
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_2.avi', fourcc, fps, (width, height))

# Number frame in init background
N = 6
N_BOI = 20
init_background = True
gen_BOI = True
# Block coordinates
BOIs_coor = []
count = 0
n = 0

# Mean and var blocks for init background size N x N_ Block
BOIs_var = [[0 for i in range(N_BOI)] for j in range(N)]
BOIs_mean = [[0 for i in range(N_BOI)] for j in range(N)]
s = 0

# Hệ số lambda size N_block
lambda_f = np.array([100 for i in range(N_BOI)], ndmin=2)
lambda_b = np.array([100 for i in range(N_BOI)], ndmin=2)
lr_f = np.array([0.01 for i in range(N_BOI)], ndmin=2)
lr_b = np.array([0.01 for i in range(N_BOI)], ndmin=2)
lr_am = np.array([0.01 for i in range(N_BOI)], ndmin=2)
p_f = np.array([0.4 for i in range(N_BOI)], ndmin=2)

# Mean and var for classify object
g_mean = np.zeros((20, 1))
g_var = np.zeros((20, 1))


def draw_BOI(img, coordinates, filter):
    # draw block
    for i in range(len(filter)):
        if filter[i]:
            cv2.rectangle(img, coordinates[i][0],
                          coordinates[i][1], (0, 0, 255), 1)
        else:
            cv2.rectangle(img, coordinates[i][0],
                          coordinates[i][1], (0, 255, 0), 1)


def write_text(img, filter, vars, g_var):
    # filter = np.reshape(filter,(1,N_BOI))
    for i in range(len(filter)):
        if filter[i]:
            cv2.putText(
                img,
                f'{int(vars[i])} - {int(g_var[i])}',
                BOIs_coor[i][1],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,
                 0,
                 0),
                2)


def get_mean_var(img, coordinates_block):
    # return mean, variance of blocks
    # Order of coordinate
    means = np.zeros((N_BOI, 1))
    vars = np.zeros((N_BOI, 1))
    for i in range(len(coordinates_block)):
        y_min, x_min = coordinates_block[i][0]
        y_max, x_max = coordinates_block[i][1]
        means[i] = np.mean(img[x_min: x_max, y_min: y_max])
        vars[i] = np.var(img[x_min: x_max, y_min: y_max])
    return means, vars


def get_Vov(vars):
    # return variance of variance of block with some frame
    Vov = np.var(vars, axis=0)
    return Vov


def proba_mean(x, mean, var, lr):
    # return gau distribution
    mean = (1 - lr) * mean + lr * x
    var = (1 - lr) * var + lr * (mean - x)**2
    p = math.exp(-(x - mean)**2 / (2 * var)) / (math.sqrt(2 * math.pi * var))
    return p


def update_model(i, delta_v, mean):
    if delta_v < lambda_b[i]:
        lr_b[i] = 0.01
    else:
        lr_b[i] = 0.1
    if lambda_b[i] < 500:
        lambda_b[i] = (1 - lr_b[i]) * lambda_b[i] + lr_b[i] * delta_v
    if proba_mean(
            mean,
            BOIs_mean[i][0],
            BOIs_var[i][0],
            lr_am[i]) > proba_mean(
            BOIs_mean[i][0] +
            3 *
            BOIs_var[i][0],
            BOIs_mean[i][0],
            BOIs_var[i][0],
            lr_am[i]):
        lr_am[i] = 0.01
    else:
        lr_am[i] = 0.1

    g_mean[i] = (1 - lr_am[i]) * g_mean[i] + lr_am[i] * mean
    g_var[i] = (1 - lr_am[i]) * g_var[i] + lr_am[i] * (mean - g_mean[i])**2


def update_p_f(positions, p_f):
    # upadate prior probability
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
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.float32) / 25
    img = cv2.filter2D(image, -1, kernel)
    s += 1
    n_o = 0
    # Generate ROI and BOI
    if gen_BOI:
        lanes = ROI.get_lanes(frame)
        lanes_image = ROI.draw_lanes(frame, lanes)

        area = BOI.get_area(lanes)
        area_image = BOI.draw_lanes(lanes_image, area)

        boi_image, BOIs_coor = BOI.get_BOI(area, area_image, N_BOI, 2)
        cv2.imshow('lane', boi_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        gen_BOI = False

    # Initialization background
    if init_background:
        if count < N:
            means, vars = get_mean_var(img, BOIs_coor)
            BOIs_mean[count] = np.reshape(means, (N_BOI,))
            BOIs_var[count] = np.reshape(vars, (N_BOI,))
            count += 1
            cv2.putText(frame, 'Init background', (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            pas = True
            Vov = get_Vov(BOIs_var)
            means, vars = get_mean_var(img, BOIs_coor)
            filter = Vov > 100
            compare = filter
            if compare.any():
                BOIs_mean.pop(0)
                BOIs_var.pop(0)

                BOIs_var.append(np.reshape(vars, (N_BOI,)))
                BOIs_mean.append(np.reshape(means, (N_BOI,)))
                pas = False

            cv2.putText(frame, 'Init background', (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if pas:
                init_background = False
                g_mean = np.mean(BOIs_mean, axis=0, keepdims=True)
                g_var = np.mean(BOIs_var, axis=0, keepdims=True)
                cv2.putText(frame, 'Done init background', (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        # Classify object
        positions = []
        t1 = time.time()
        means, vars = get_mean_var(img, BOIs_coor)
        delta_v = np.absolute(vars - g_var.T)
        delta_m = np.absolute(means - g_mean.T)
        p_vb = np.exp(-delta_v / lambda_b.T)
        p_vf = 1 - np.exp(-delta_v / lambda_f.T)
        p_fv = (p_vf * p_f.T) / (p_vb * (1 - p_f.T) + p_vf * p_f.T)

        f_1 = p_fv > 0.7
        # print(f_1)
        write_text(frame, f_1, vars, g_var.T)
        # if lambda_f[filter] < 2000:
        #     lambda_f[filter] = (1 - lr_f[filter]) * lambda_f[filter] + lr_f[filter] * delta_v[filter]

        n_block = len(f_1[f_1])
        rate = round(n_block / N_BOI, 1)
        cv2.putText(frame, f'{rate * 100}%', (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        draw_BOI(frame, BOIs_coor, f_1)
        t2 = time.time()
        print(f"time - {t2-t1}")
        # for i in range(N_BOI):
        #     # cv2.putText(frame, str(i), BOIs_coor[i][0],cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2 )
        #     mean, var = get_mean_var(img, BOIs_coor[i])
        #     delta_v = abs(var - g_var[i])
        #     delta_m = abs(mean - g_mean[i])
        #     p_vb = math.exp(-delta_v/lambda_b[i])
        #     p_vf = 1 - math.exp(-delta_v/lambda_f[i])
        #     p_fv =  (p_vf * p_f[i])/(p_vb * (1 - p_f[i]) + p_vf * p_f[i])
        #     # p_m = proba_mean(mean, g_mean[i], g_var[i], lr_am[i])
        #     if p_fv > 0.7 :
        #         n_o += 1
        #         positions.append(i)
        #         cv2.putText(frame, f'{round(var, 1)} - {round(g_var[i], 1)}', BOIs_coor[i][1],cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2 )
        #         if lambda_f[i] < 2000:
        #             lambda_f[i] = (1 - lr_f[i]) * lambda_f[i] + lr_f[i] * delta_v
        #     elif p_f[i] < 0.5 and i == 0:
        #         update_model(i, delta_v, mean)
        #         cv2.putText(frame, f'{round(var, 1)} - {round(g_var[i], 1)}', BOIs_coor[i][1],cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
        #         if get_Vov(BOIs_var[i]) < 100:
        #             g_var[i] = 0.01 * var + (1 - 0.01)*g_var[i]
        #             g_mean[i] = 0.01 * mean + (1 - 0.01)*g_mean[i]
        #     else:
        #         # p2 = proba_mean(g_mean[i] + 2*math.sqrt(g_var[i]), g_mean[i], g_var[i], lr_am[i])
        #         if delta_m > 40 :
        #             n_o +=1
        #             cv2.putText(frame, f'{round(delta_m, 1)}', BOIs_coor[i][1],cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2 )
        #             positions.append(i)
        #         else:
        #             update_model(i, delta_v, mean)

        # p_f = [0.4 for i in range(N_BOI)]
        # p_f = update_p_f(positions, p_f)
        # rate = round(n_o/len((BOIs_coor)), 2)
        # cv2.putText(frame, f'{rate * 100}%', (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2 )
        # draw_BOI(frame, BOIs_coor, positions)
        # t2 = time.time()
        # t = t2 - t1
        # print(f"time - {t}")
    cv2.imshow("video", frame)
    out.write(frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
