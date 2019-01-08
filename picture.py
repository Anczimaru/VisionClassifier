from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2

def show_opened_image(image):

    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def my_canny(img, param=0.33):
	median = np.median(img)

	lower_limit = int(max(0, (1.0-param)*median))
	upper_limit = int(min(255, (1.0+param)*median))
	res = cv2.Canny(img, lower_limit, upper_limit)
	return res


def sobel_x(data_dir, dst_dir, f, debug_mode = 0):
    img = cv2.imread(os.path.join(data_dir,f),0)
    sobel_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    s = "sobel_x_{}".format(f)
    cv2.imwrite(os.path.join(dst_dir,s),sobel_x)


def sobel_y(data_dir, dst_dir, f, debug_mode = 0):
    img = cv2.imread(os.path.join(data_dir,f),0)
    sobel_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    s = "sobel_y_{}".format(f)
    cv2.imwrite(os.path.join(dst_dir,s),sobel_y)


def canny_edge_detector(data_dir, dst_dir, f, auto = 1, debug_mode = 0):

    img = cv2.imread(os.path.join(data_dir,f),-1)
    if auto == 1:
        edges = my_canny(img)
    else:
        edges = cv2.Canny(img,100,200)
    s = "edges_{}".format(f)
    if debug_mode == 1:
        show_opened_image(edges)
        print(edges.shape)
        print(s)
    tmp_img = Image.fromarray(edges)
    tmp_img.save(os.path.join(dst_dir, s))

def laplacian(data_dir, dst_dir, f, debug_mode = 0):
    ddepth = cv2.CV_16S
    kernel_size=5
    img = cv2.imread(os.path.join(data_dir,f),-1)
    # Remove noise by blurring with a Gaussian filter
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(img_gray, ddepth, kernel_size)
    s = "laplacian_{}".format(f)
    if debug_mode == 1:
        print(laplacian.shape)
    cv2.imwrite(os.path.join(dst_dir,s),laplacian)


def segmentation(data_dir, f, debug_mode = 0):
    #ACCORDING TO CV2 documentation
    img = cv2.imread(os.path.join(data_dir, f),-1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    #show_opened_image(sure_bg)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    #show_opened_image(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    #show_opened_image(unknown)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img, markers)

    img[markers == -1] = [255,0,0]
    if debug_mode == 1:
        show_opened_image(img)

def whiteunion3d(img1, img2,debug_mode = 0):
    #Do union of 3d matrixes for white color
    if debug_mode ==1:
        print(img1.shape)
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0,0,0], dtype=np.uint8)
    upper_white = np.array([0,0,255], dtype=np.uint8)

     # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv1, lower_white, upper_white)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img1,img2, mask= mask)
    return res

def feature_matcher(src_dir, dst_dir, f, test_img_path, mode=1):
    print(f)
    try:
        img1 = cv2.imread(test_img_path)          # object we are looking for
        img2 = cv2.imread(os.path.join(src_dir, f)) # trainImage

        # ORB Detector
        orb = cv2.ORB_create()

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if mode == 1:
            dst_dir = os.path.join(dst_dir,"BF")

            # Brute Force Matching
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            matches = bf.match(des1, des2)
            matches = sorted(matches, key = lambda x:x.distance)

            img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

        if mode == 2:
            dst_dir = os.path.join(dst_dir,"FLANN")
            # FLANN parameters
            FLANN_INDEX_LSH = 6
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                           table_number = 6,
                           key_size = 12,
                           multi_probe_level = 1)
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params,search_params)


            matches = flann.knnMatch(des1,des2,k=2)

            # Need to draw only good matches, so create a mask
            matchesMask = [[0,0] for i in range(len(matches))]

            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    matchesMask[i]=[1,0]

            draw_params = dict(matchColor = (0,255,0),
                               singlePointColor = (255,0,0),
                               matchesMask = matchesMask,
                               flags = 0)

            img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
        s = "feature_{}".format(f)
        cv2.imwrite(os.path.join(dst_dir,s),img3)
        return 0
    #Error handling
    except cv2.error as e:
        print("Not able to find matches if levels under 4, ommit")
        print(e)
    except Exception as e:
        print(e)



def downsample(data_dir,f,n):
    return 0
