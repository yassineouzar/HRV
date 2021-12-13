import matplotlib.pyplot as plt
import os

from shutil import copyfile
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.fftpack import fft, ifft, fftfreq
import scipy.signal as signal
from scipy import sparse
import scipy
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mtcnn import MTCNN
import cv2

import math


def cdf_filter(C_rgb, LPF, HPF, fs, bpf=False):
    """
    Color-distortion filtering for remote photoplethysmography.
    """
    L = C_rgb.shape[0]
    # temporal normalization
    Cn = C_rgb / np.average(C_rgb, axis=0) - 1
    # Hanning Window
    # window = np.hanning(L).reshape(-1,1)
    # Cn = Cn * window
    #  FFT transform
    FF = fft(Cn, n=L, axis=0)
    freq = fftfreq(n=L, d=1 / fs)
    # Characteristic transformation
    H = np.dot(FF, (np.array([[-1, 2, -1]]) / math.sqrt(6)).T)
    # Energy measurement
    if bpf == True:
        # BPF only
        W = 1
    else:
        W = (H * np.conj(H)) / np.sum(FF * np.conj(FF), 1).reshape(-1, 1)
    # band limitation
    FMask = (freq >= LPF) & (freq <= HPF)
    FMask = FMask + FMask[::-1]
    W = W * FMask.reshape(-1, 1)
    # Weighting
    Ff = np.multiply(FF, (np.tile(W, [1, 3])))
    # temporal de-normalization
    C = np.array([(np.average(C_rgb, axis=0)), ] * (L)) * np.abs(np.fft.ifft(Ff, axis=0) + 1)
    return C


def POSMethod(rgb_components, WinSec=1.6, LPF=0.7, HPF=2.5, fs=25, filter=False):
    """
    POS method
    WinSec :was a 32 frame window with 20 fps camera
    (i) L = 32 (1.6 s), B = [3,6]
    (ii) L = 64 (3.2 s), B = [4,12]
    (iii) L = 128 (6.4 s), B = [6,24]
    (iv) L = 256 (12.8 s), B = [10,50]
    (v) L = 512 (25.6 s), B = [18,100]
    """
    N = rgb_components.shape[0]
    H = np.zeros(N)
    l = math.ceil(WinSec * fs)

    # loop from first to last frame
    for t in range(N - l + 1):
        # spatical averagining
        C = rgb_components[t:t + l]
        if filter == True:
            C = cdf_filter(C, LPF, HPF, fs=fs, bpf=True)
            Cn = C / np.average(C, axis=0)
        else:
            # temporal normalization
            # C = cdf_filter.cdf_filter(C, LPF, HPF, fs=fs,bpf=False)
            Cn = C / np.average(C, axis=0)

        # projection (orthogonal to 1)
        S = np.dot(Cn, np.array([[0, 1, -1], [-2, 1, 1]]).T)
        # alpha tuning
        P = np.dot(S, np.array([[1, np.std(S[:, 0]) / np.std(S[:, 1])]]).T)
        # overlap-adding
        H[t:t + l] = H[t:t + l] + (np.ravel(P) - np.mean(P)) / np.std(P)

    return H


def bandpass_butter(arr, cut_low, cut_high, rate, order=2):
    nyq = 0.5 * rate
    low = cut_low / nyq
    high = cut_high / nyq
    b, a = signal.butter(order, [low, high], btype='band', analog=False, output='ba')
    out = signal.filtfilt(b, a, arr)
    return out


def lowpass_butter(arr, cut, rate, order=2):
    nyq = 0.5 * rate
    cut = cut / nyq
    b, a = scipy.signal.butter(order, cut, btype='low', analog=False, output='ba')
    out = scipy.signal.filtfilt(b, a, arr)
    return out

def get_roi(image):
    detector = MTCNN()
    global image1
    detector.detect_faces(image)

    result = detector.detect_faces(image)

    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    bounding_box = result[0]['box']


    keypoints = result[0]['keypoints']

    cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

    image = image[bounding_box[1] : bounding_box[1] + bounding_box[3] , bounding_box[0] : bounding_box[0] + bounding_box[2]]

    left_eye=[]
    left_eye = keypoints['left_eye']
    a1=left_eye[0]
    a2=left_eye[1]
    x1 = a1 - bounding_box[0]

    right_eye=[]
    right_eye = keypoints['right_eye']
    b1=right_eye[0]
    x2 = b1 - bounding_box[0]
    b2=right_eye[1]

    y2= int((a2 - bounding_box[1])/2)

    #cv2.rectangle(image, (x1, 5), (x2, y2) , (0,155,255),2)

    image = image[10 : y2 , x1 : x2]
    #image = img_to_array(image)

    #print(image.shape)

    image1 = np.resize(image,(80,120,3))
    #print(image1.shape, image1.ndim)
    return image1
def load_images_from_one_directory(DIR_DATA):
    list_dir3 = os.listdir(DIR_DATA)

    NB_IMAGES = len(list_dir3)
    IMAGE_WIDTH = 120
    IMAGE_HEIGHT = 80
    imgs = np.zeros(shape=(NB_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

    for k in range(NB_IMAGES):
        #imgs[k] = get_roi(cv2.imread(DIR_DATA + '/' + list_dir3[k]))
        imgs[k] = cv2.imread(DIR_DATA + '/' + list_dir3[k])
        y, x = imgs[k].shape[:2]
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h = []
        w = []
        for m in range(x):
            for l in range(y):

                b, g, r = (imgs[k][l, m])
                if ([b, g, r] >= [30, 30, 30]):
                    w.append(m)
                    h.append(l)
                    # mask = [b,g,r]>=[15,15,15]
        x1, x2, y1, y2 = min(w), max(w), min(h), max(h)
        img = img[y1:y2, x1:x2]

        # print(x1,x2,y1,y2)

        img = cv2.resize(img, (120, 160))
        #imgs[k] = imgs[k][:, :, 1]
        #print(imgs[k].shape)
        print(str(k) + '/' + str(NB_IMAGES))

    return imgs


def append_to_list_hr(path_y, path_save, path_im, path_review, path_save_im):
    list_dir = sorted(os.listdir(path_y))
    fps = 25

    count = 0
    file_count = 0
    # path_to_save = []
    #for i in range(int(len(list_dir))):
    for i in range(0, len(list_dir)):

        list_dir1 = os.listdir(path_y + '/' + list_dir[i])
        Heart_rate_dir1 = []
        for j in range(0,len(list_dir1)):
            imgs = []
            imgs_save = []

            #for j in range(int(len(list_dir1))):
            list_dir2 = os.listdir(path_y + '/' + list_dir[i] + '/' + list_dir1[j])
            heart_rate = [filename for filename in list_dir2 if filename.startswith("Pulse Rate_BPM.t")]

            blood_pressure = [filename for filename in list_dir2 if filename.startswith("BP")]
            blood_pressure_sys = [filename for filename in list_dir2 if filename.startswith("LA Systolic BP_mmHg")]

            path_to_save_hr = os.path.join(path_save + '/' + list_dir[i] + '/' + list_dir1[j])
            path_to_review_hr = os.path.join(path_review + '/' + list_dir[i] + '/' + list_dir1[j])
            path_to_im = path_im + '/' + list_dir[i] + '/' + list_dir1[j]
            list_dir3 = os.listdir(path_to_im)
            path_to_save_im = os.path.join(path_save_im + '/' + list_dir[i] + '/' + list_dir1[j])
            # for im in sorted(list_dir3):
            # imag = os.path.join(path_to_im, im)
            # imgs.append(imag)
            images = load_images_from_one_directory(path_to_im)
            sig = np.zeros(images.shape[0])

            for l in range(images.shape[0]):
                b,g,r = cv2.split(images[l])
                sig[l] = np.mean(g)
                #print(g.shape)
            sampling_rate = 25
            time = np.arange(0, ((len(sig) / sampling_rate) - 1 // sampling_rate), 1 / sampling_rate)
            # sig = sig[0:-(len(sig)-len(heart))]
            # sig_sys = sig_sys[0:-(len(sig_sys) - len(heart))]

            sig_filtred = bandpass_butter(sig, 0.6, 4, 25, 2)
            # detection des pics et des vallies
            indexes_peaks, _ = scipy.signal.find_peaks(-sig_filtred, distance=10, height=0.05)
            indexes_valleys, _ = scipy.signal.find_peaks(sig_filtred, distance=10, height=0.05)
            #plt.plot(sig_filtred, '-b', label='PPG (GT)')
            #plt.plot(indexes_peaks, sig_filtred[indexes_peaks], 'r.')
            #plt.plot(indexes_valleys, sig_filtred[indexes_valleys], 'g.')
            #plt.show()
            # plt.plot(sig_detrend)
            fps = 25

            iHR_peaks = np.gradient(time[indexes_peaks])
            iHR_valleys = np.gradient(time[indexes_valleys])

            iHR_peaks = 60 / iHR_peaks
            iHR_valleys = 60 / iHR_valleys

            hr_p = []
            hr_v = []
            for n in range(len(iHR_peaks)):
                a = iHR_peaks[n]
                b = round((60 / a) * 25)
                vect_p = [iHR_peaks[n]]
                vect_v = [iHR_valleys[n]]

                hri_p = np.repeat(vect_p, b)
                hri_v = np.repeat(vect_v, b)

                hr_p.extend(hri_p)
                hr_v.extend(hri_v)
            print(len(hr_p), len(hr_v))

            plt.plot(hr_p)
            plt.plot(hr_v)

            #plt.show()
            for im in sorted(list_dir3):

                imag = os.path.join(path_to_im, im)
                path_to_im1 = os.path.join(path_save_im + '/' + list_dir[i] + '/' + list_dir1[j] + '/' + im)
                imgs_save.append(path_to_im1)
                imgs.append(imag)
            #print(imgs[int(0):int(20)])

            print(len(imgs))


            for hr in heart_rate:
                Heart_rate_dir = os.path.join(path_y + '/' + list_dir[i] + '/' + list_dir1[j] + '/' + hr)
                print(Heart_rate_dir)

                path_to_save = os.path.join(path_save + '/' + list_dir[i] + '/' + list_dir1[j] + '/' + hr)
                path_to_review = os.path.join(path_review + '/' + list_dir[i] + '/' + list_dir1[j] + '/' + hr)
                hist = os.path.join(path_save + '/' + list_dir[i] + '/' + list_dir1[j] + '/' + "history.txt")

            for BP in blood_pressure:
                BP_dir = os.path.join(path_y + '/' + list_dir[i] + '/' + list_dir1[j] + '/' + BP)
                print(BP_dir)

            with open(Heart_rate_dir, 'r') as file:
                heart = [line.rstrip('\n') for line in file]
            with open(BP_dir, 'r') as file:
                sig = [line.rstrip('\n') for line in file]



            sampling_rate = 1000
            time = np.arange(0, ((len(heart) / sampling_rate) - 1 // sampling_rate), 1 / sampling_rate)
            #sig = sig[0:-(len(sig)-len(heart))]
            #sig_sys = sig_sys[0:-(len(sig_sys) - len(heart))]
            time256 = np.arange(time[0], time[-1], 1 / 1000)

            signal256 = PchipInterpolator(time, sig, extrapolate=False)
            HR256 = PchipInterpolator(time, heart, extrapolate=False)
            sig256 = signal256(time256)
            HR256 = HR256(time256)
            sig_filtred = bandpass_butter(sig256, 0.6, 20, 1000, 2)
            nbr_pt = 30
            v = np.ones((1, round(nbr_pt / 2)))
            b = scipy.signal.convolve(v, v)
            a = sum(b)
            b = b.reshape((-1,))
            sig_liss = scipy.signal.filtfilt(b, a, sig_filtred)

            #plt.plot(sig_liss)
            #plt.plot(sig_filtred)
            #plt.show()

            #detection des pics et des vallies
            indexes_peaks, _ = scipy.signal.find_peaks(-sig_liss, distance=400, height=0.5)
            indexes_valleys, _ = scipy.signal.find_peaks(sig_liss, distance=400, height=0.5)

            fps = 25

            iHR_peaks = np.gradient(time256[indexes_peaks])
            iHR_valleys = np.gradient(time256[indexes_valleys])

            iHR_peaks = 60 / iHR_peaks
            iHR_valleys = 60 / iHR_valleys
            hr_p = []
            hr_v = []
            for n in range(len(iHR_peaks)):
                a = iHR_peaks[n]
                b = round((60 / a) * 25)
                vect_p = [iHR_peaks[n]]
                vect_v = [iHR_valleys[n]]

                hri_p = np.repeat(vect_p, b)
                hri_v = np.repeat(vect_v, b)

                hr_p.extend(hri_p)
                hr_v.extend(hri_v)
            print(len(hr_p), len(hr_v))

            #ti = np.linspace(time256[indexes][0], time256[indexes][-1], len(time256))
            #print(len(ti))
            #t = np.linspace(time[0], time[-1])
            time25 = np.arange(time[0], time[-1], 1 / 25)
            iHR25 = PchipInterpolator(time, heart, extrapolate=False)
            #iHR25 = scipy.interpolate.interp1d(time256[indexes], iHR, kind="nearest")

            iHR25 = iHR25(time25)
            #print(iHR25)
            #plt.step(time256[indexes_peaks],iHR_peaks)
            #plt.step(time256[indexes_valleys], iHR_valleys)

            plt.plot(hr_p)
            plt.plot(hr_v)
            plt.plot(iHR25)
            #plt.step(time256[indexes], iHR)
            plt.show()


path_gt = '/media/ouzar1/Seagate Backup Plus Drive/Physiology'
#path_gt = '/home/ouzar1/Desktop/Dataset1/MMSE_HR'
path_im = '/media/ouzar1/Seagate Expansion Drive/ROI1'
path_save = '/media/ouzar1/Seagate Expansion Drive/HR_train1'
path_save_im = '/media/ouzar1/Seagate Expansion Drive/ROI_train1'

path_review = '/media/ouzar1/Seagate Expansion Drive/HR_train_review'
sampling_rate = 1000

append_to_list_hr(path_gt, path_save, path_im, path_review, path_save_im)

"""
            lamb = 20
            T = len(sig256)
            I = sparse.eye(T)
            D = sparse.spdiags(np.transpose(np.ones((T - 2, 1)) * np.array([[1, -2, 1]])), np.array([0, 1, 2]),
                               T - 2,
                               T)
            sig_detrend = (I - sparse.linalg.inv(I + lamb * lamb * np.transpose(D) * D)) * np.transpose(sig256)
images =load_images_from_one_directory(path_to_im)
            sig_av = []
            for k in range(0, len(images)):
                green_channel = images[k][:, :, 1]
                non_zero = cv2.findNonZero(green_channel)
                spatial_average = np.mean(green_channel)
                sig_av.append(spatial_average.copy())
            plt.plot(sig_av, '-b', label='PPG (GT)')
            plt.show()
            time = np.arange(0, (len(sig_av) / fps) - 1 // fps, 1 / fps)
            time256 = np.arange(time[0], time[-1], 1 / 25)

            signal256 = PchipInterpolator(time, sig_av, extrapolate=True)

            sig_av = signal256(time256)

            lamb = 20
            T = len(sig_av)
            I = sparse.eye(T)
            D = sparse.spdiags(np.transpose(np.ones((T - 2, 1)) * np.array([[1, -2, 1]])), np.array([0, 1, 2]),
                                   T - 2, T)

            sig_detrend_green = (I - sparse.linalg.inv(I + lamb * lamb * np.transpose(D) * D)) * np.transpose(sig_av)

            #sig_detrend_green = (bandpass_butter(sig_detrend_green, 0.7, 4, 25, order=2)) * 10
            plt.plot(sig_detrend_green)
            plt.show()
"""