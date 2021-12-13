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


def load_images_from_one_directory(DIR_DATA):
    list_dir3 = os.listdir(DIR_DATA)

    NB_IMAGES = len(list_dir3)
    IMAGE_WIDTH = 120
    IMAGE_HEIGHT = 160
    imgs = np.zeros(shape=(NB_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

    for k in range(NB_IMAGES):
        imgs[k] = cv2.imread(DIR_DATA + '/' + list_dir3[k])
        print(str(k) + '/' + str(NB_IMAGES))

    return imgs


def append_to_list_hr(path_y, path_save, path_im, path_review, path_save_im):
    list_dir = os.listdir(path_y)
    fps = 25

    count = 0
    file_count = 0
    # path_to_save = []
    #for i in range(int(len(list_dir))):
    for i in range(6, len(list_dir)):

        list_dir1 = os.listdir(path_y + '/' + list_dir[i])
        Heart_rate_dir1 = []
        for j in range(0,len(list_dir)):
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
            time256 = np.arange(time[0], time[-1], 1 / 25)
            signal256 = PchipInterpolator(time, sig, extrapolate=True)
            HR256 = PchipInterpolator(time, heart, extrapolate=True)
            sig256 = signal256(time256)
            HR256 = HR256(time256)

            plt.plot(HR256)
            plt.plot(sig256)
            plt.show()

            lamb = 30
            T = len(sig256)
            I = sparse.eye(T)
            D = sparse.spdiags(np.transpose(np.ones((T - 2, 1)) * np.array([[1, -2, 1]])), np.array([0, 1, 2]),
                               T - 2,
                               T)
            sig_detrend = (I - sparse.linalg.inv(I + lamb * lamb * np.transpose(D) * D)) * np.transpose(sig256)

            fps = 25
            # sig_detrend = preprocessing.scale(sig_detrend)
            # sig_detrend = lowpass_butter(sig_detrend, 0.2, fps, 10)
            # sig_detrend = (bandpass_butter(sig_detrend, 0.7, 3.2, 25, order=2))
            indexes, _ = scipy.signal.find_peaks(sig_detrend, distance=10, height=0.2)
            #plt.plot(sig_detrend, '-b', label='PPG (GT)')
            #plt.plot(indexes, sig_detrend[indexes], 'r.')
            # plt.plot(sig_detrend)
            #plt.show()
            #t = time256
            #print(indexes)
            iHR = np.gradient(time256[indexes])

            iHR = 60 / iHR
            # iHR = list(iHR)
            # print(len(iHR), len(time256))
            # print(time256[indexes])

            fig, ax = plt.subplots(2, 1, figsize=(17, 9))
            ax[0].step(time256, HR256)
            ax[0].set_title('HR de référence pris de la base de données')
            ax[1].step(time256[indexes], iHR)
            ax[1].set_title('HR calculé à partir du signal BP de la base de données')
            plt.show()

            s = input("Enter value: ")

            if s == '1':
                if not os.path.exists(path_to_save):
                    os.makedirs(path_to_save_hr)
                if not os.path.exists(path_to_save_im):
                    os.makedirs(path_to_save_im)
                #copyfile(Heart_rate_dir, path_to_save)
                a,b = input("Enter value: ").split()
                print(len(heart))
                print(heart[int(a*25):int(b*25)])
                heart = heart[int(a*25):int(b*25)]
                imgs = imgs[int(a):int(b)]
                imgs_save = imgs_save[int(a):int(b)]
                print(imgs[int(a):int(b)])
                print(len(imgs))
                with open(path_to_save, 'w') as file:
                    for item in heart:
                        file.write("%s \n" % item)
                for ii in range(len(imgs)):
                    copyfile(imgs[ii], imgs_save[ii])
                print(path_to_save, "is saved")

            elif s == '2':
                if not os.path.exists(path_to_review_hr):
                    os.makedirs(path_to_review_hr)
                copyfile(Heart_rate_dir, path_to_save_im)
                print(path_to_save, "to review")

            else:
                print(path_to_save, "not saved")

                #break

path_gt = '/tmp/.x2go-ouzar1/media/disk/_cygdrive_C_Users_ouzar1_DOCUME1_SHAREF1/BP4D/MMSE_HR_review'
#path_gt = 'D:/HR_estimation/MMSE_HR'
path_im = '/media/ouzar1/Seagate Expansion Drive/ROI1'
path_save = '/media/ouzar1/Seagate Expansion Drive/HR_train1'
path_save_im = '/media/ouzar1/Seagate Expansion Drive/ROI_train1'

path_review = '/media/ouzar1/Seagate Expansion Drive/HR_train_review'
sampling_rate = 1000

append_to_list_hr(path_gt, path_save, path_im, path_review, path_save_im)

"""

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