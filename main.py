import cv2
import numpy as np
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt


def reflection_enhance(norm_img):
    ''' Enhance image to improve specular reflection contrast from other parts of image'''

    hsv = cv2.cvtColor(norm_img, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    for i in range(3):
        norm_img[:, :, i] = (1 - s) * norm_img[:, :, i]
    return norm_img


def histogram_denoise(grayscale):
    '''Reduce noise to prevent variations confused with bumps associated with specular reflections'''

    hist = cv2.calcHist(images=[grayscale], channels=[0], mask=None, histSize=[256], ranges=[0, 1])
    denoised_hist = denoise_wavelet(hist, wavelet_levels=1, method="VisuShrink", channel_axis=1)
    # # show histogram
    # plt.plot(denoised_hist)
    # plt.show()

    return denoised_hist


def find_specular_bump_threshold(w):
    '''Find location of beginning of specular bump'''
    w2 = np.gradient(w, axis=0)
    w2 = thresholding(w2)
    w3 = np.gradient(w2, axis=0)
    w3 = thresholding(w3)
    threshold = np.where(w3 > 0)[0][-1]
    return threshold


def thresholding(w):
    '''Helper method for find_specular_bump_threshold'''
    for i in range(256):
        w[i, 0] = 0 if w[i, 0] <= 0 else 1
    return w


if __name__ == "__main__":
    # get a image from the file
    img = cv2.imread("D:\\GLENDA_v1.5_no_pathology\\no_pathology\\frames\\v_2506_s_0-95\\f_0.jpg")  # 640x360 size

    # step 1: reflection enhancement
    normalized_img = np.float32(cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX))
    enhanced = reflection_enhance(normalized_img)
    # # Code to view results of enhancement #
    # enhanced = 255 * enhanced
    # enhanced = enhanced.astype(np.uint8)
    # cv2.imshow("Window", enhanced)
    # cv2.waitKey(0)

    # step 2: histogram denoising
    enhanced_gray = cv2.cvtColor(normalized_img, cv2.COLOR_RGB2GRAY)
    denoised_hist = histogram_denoise(enhanced_gray)
    # # Code to view results of grayscale #
    # enhanced_gray = 255 * enhanced_gray
    # enhanced_gray = enhanced_gray.astype(np.uint8)
    # cv2.imshow("Window", enhanced_gray)
    # cv2.waitKey(0)

    # step 3: Specular bump thresholding
    threshold = find_specular_bump_threshold(denoised_hist)

    # step 4: Specular Lobe Detection
    specular_spike_mask = np.array((enhanced_gray*255 >= threshold).astype(int), dtype=np.uint8)
    # specular_spike_mask *= 255 # scales pixel intensities to 255 so that it can be seen using cv2.imshow()
    # cv2.imshow("Window", specular_spike_mask)
    # cv2.waitKey(0)

    diamond = np.ones((3, 3), 'uint8')
    diamond[0, 0] = 0
    diamond[0, 2] = 0
    diamond[2, 0] = 0
    diamond[2, 2] = 0
    specular_mask = cv2.dilate(specular_spike_mask, kernel=diamond)
    # specular_mask *= 255
    # cv2.imshow("Window", specular_mask)
    # cv2.waitKey(0)

    # Superimposing mask on real image to see segmentation results
    img[specular_mask > 0] = 0
    cv2.imshow("Window", img)
    cv2.waitKey(0)


