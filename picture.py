from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2

def show_opened_image(image, caption="image"):

    cv2.imshow(caption,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def my_canny(img, param=0.33):
	median = np.median(img)

	lower_limit = int(max(0, (1.0-param)*median))
	upper_limit = int(min(255, (1.0+param)*median))
	res = cv2.Canny(img, lower_limit, upper_limit)
	return res


def sobel_x(data_dir, dst_dir, f, save=1, debug_mode = 0):
    img = cv2.imread(os.path.join(data_dir,f),0)
    sobel_x = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
    s = "sobel_x_{}".format(f)
    if save == 1:
        cv2.imwrite(os.path.join(dst_dir,s),sobel_x)
    else:
        img_cv = cv2.resize(sobel_x,(sobel_x.shape[1],sobel_x.shape[0]))
        return cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)


def sobel_y(data_dir, dst_dir, f, save=1, debug_mode = 0):
    img = cv2.imread(os.path.join(data_dir,f),0)
    sobel_y = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=5)
    s = "sobel_y_{}".format(f)
    if save == 1:
        cv2.imwrite(os.path.join(dst_dir,s),sobel_y)
    else:
        print(sobel_y.size)
        print(type(sobel_y))
        img_cv = cv2.resize(sobel_y,(sobel_y.shape[1],sobel_y.shape[0]))
        return cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)

def canny_edge_detector(data_dir, dst_dir, f, save=1, auto = 1, debug_mode = 0):

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
    if save == 1:
        tmp_img.save(os.path.join(dst_dir, s))
    else:
        img_cv = cv2.resize(edges,(edges.shape[1],edges.shape[0]))
        return cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)



def laplacian(data_dir, dst_dir, f, save=1, debug_mode = 0):
    ddepth = cv2.CV_8U
    kernel_size=5
    img = cv2.imread(os.path.join(data_dir,f),-1)
    # Remove noise by blurring with a Gaussian filter
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(img_gray, ddepth, kernel_size)
    s = "laplacian_{}".format(f)
    if debug_mode == 1:
        print(laplacian.shape)
    if save == 1:
        cv2.imwrite(os.path.join(dst_dir,s),laplacian)
    else:
        img_cv = cv2.resize(laplacian,(laplacian.shape[1],laplacian.shape[0]))
        return cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)


def White_Intersection_3d(img1, img2,debug_mode = 1):
    #Do union of 3d matrixes for white color

    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0,0,0], dtype=np.uint8)
    upper_white = np.array([0,0,255], dtype=np.uint8)

     # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv1, lower_white, upper_white)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img1,img2, mask= mask)
    return res


def downsample(data_dir,f,n):
    return 0
