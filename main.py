from weak_detect import *

def get_run_time(function, var):
    '''Get run time of function passed in'''

    start_time = datetime.datetime.now()

    function(var)

    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds() * 1000
    print(execution_time)


if __name__ == "__main__":
    # get a image from the file
    img = cv2.imread("D:\\GLENDA_v1.5_no_pathology\\no_pathology\\frames\\v_2506_s_0-95\\f_0.jpg")  # 640x360 size

    run_weak_detect(img)

    # get_run_time(run_weak_detect, img) # Don't forget to disable display window for more accurate runtime




