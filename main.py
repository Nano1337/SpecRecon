from weak_detect import *
from load_data import *
import os
import random

def get_run_time(function, var):
    '''Get run time of function passed in'''

    start_time = datetime.datetime.now()

    function(var)

    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds() * 1000
    print(execution_time)


if __name__ == "__main__":
    # get an image from the file
    # img = cv2.imread("D:/weak_data/weak_test_images/1.tif")  # 640x360 size
    load_masks()
    # get_run_time(run_weak_detect, img) # Don't forget to disable display window for more accurate runtime





