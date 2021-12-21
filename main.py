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
    MIN_MATCH_COUNT = 10;

    FLANN_INDEX_KDTREE = 0;
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5);
    search_params = dict(checks=50)  # maybe change to 100 according to paper

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(d1, d2, k=2)

    # store all good matches as per Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
        matchesMask = mask.ravel().tolist()

        h = img1.shape[0]
        w = img1.shape[1]
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - %d/%d" % (len(good, MIN_MATCH_COUNT)))

    draw_params = dict(matchColor=(0, 255, 0),  # Draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # Draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    # Map corresponding matches using green lines
    # img3 = cv2.resize(img3, (1280, 484))
    # cv2.imshow("Window", img3)
    # cv2.waitKey(0)

    # # Warp source image to destination (frame1 -> frame2) based on homography
    im_out = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))
    # im_out = cv2.resize(im_out, (640, 484))
    # cv2.imshow("Window", im_out)
    # cv2.waitKey(0)

    # Display transformed frame1 overlaid onto frame2
    background = img2
    overlay = im_out
    added_image = cv2.addWeighted(background, 0.4, overlay, 0.35, 0)
    # added_image = cv2.resize(added_image, (640, 484))
    # cv2.imshow('Window', added_image)
    # cv2.waitKey(0)

    # replace areas of specular reflection in frame2 (using mask generated) and take pixels from transformed
    # use frame2's mask bc it's unwarped
    detected_img2, img2_mask = run_weak_detect(img2)
    img_fg = cv2.bitwise_and(im_out, im_out, mask=img2_mask)
    img2_mask_inv = cv2.bitwise_not(img2_mask)
    img2[img2_mask > 0] = 0 #using frame2's mask spec refl locs
    cv2.resize(img2, (640, 484))
    output = cv2.add(img_fg, img2)
    cv2.imshow('Window', output)
    cv2.waitKey(0)

