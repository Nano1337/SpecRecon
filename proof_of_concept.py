def run_poc():
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
    search_params = dict(checks=100)  # maybe change to 100 according to paper

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(d1, d2, k=2)

    # store all good matches as per Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
        matchesMask = mask.ravel().tolist()

        h = img1.shape[0]
        w = img1.shape[1]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
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
    img2[img2_mask > 0] = 0  # using frame2's mask spec refl locs
    cv2.resize(img2, (640, 484))
    output = cv2.add(img_fg, img2)
    cv2.imshow('Window', output)
    cv2.waitKey(0)