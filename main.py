from weak_detect import *
import matplotlib.pyplot as plt


def get_run_time(function, var):
    '''Get run time of function passed in'''

    start_time = datetime.datetime.now()

    function(var)

    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds() * 1000
    print(execution_time)


def getSIFTfeatures(image, isImage=True, imask=None):
    '''Get keypoints and descriptors of an input image'''
    # reading image
    if isImage:
        img = cv2.imread(image)
    else:
        img = image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # keypoints
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, mask=imask)

    # display keypoints on image
    # img = cv2.drawKeypoints(gray, keypoints, img)
    # cv2.imshow("Window", img)
    # cv2.waitKey(0)

    return keypoints, descriptors


if __name__ == "__main__":
    # get a image from the file
    # img = cv2.imread("D:\\GLENDA_v1.5_no_pathology\\no_pathology\\frames\\v_2506_s_0-95\\f_0.jpg")  # 640x360 size
    # get_run_time(run_weak_detect, img) # Don't forget to disable display window for more accurate runtime

    # Load images and SIFT
    img1 = cv2.imread("frame1.jpg")
    img1, mask1 = run_weak_detect(img1)
    kp1, d1 = getSIFTfeatures(img1, False, 1 - mask1)

    img2 = cv2.imread("frame2.jpg")
    img2, mask2 = run_weak_detect(img2)
    kp2, d2 = getSIFTfeatures(img2, False, 1 - mask2)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(d1, d2)
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:], img2, flags=2)
    cv2.imshow("Window", img3)
    cv2.waitKey(0)

    # TODO: replace matching algo with this: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html

