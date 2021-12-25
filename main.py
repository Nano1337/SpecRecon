from weak_detect import *

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



