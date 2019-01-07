from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2

def show_opened_image(image):

    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def canny_edge_detector(data_dir, dst_dir, f, debug_mode=0):

    img = cv2.imread(os.path.join(data_dir,f),-1)
    edges = cv2.Canny(img,100,200)
    if debug_mode == 1:
        show_opened_image(edges)
    print(edges.shape)
    s = "edges_{}".format(f)
    print(s)
    tmp_img = Image.fromarray(edges)
    tmp_img.save(os.path.join(dst_dir, s))

def segmentation(data_dir,f,debug_mode=0):
    #ACCORDING TO CV2 documentation
    img = cv2.imread(os.path.join(data_dir,f),-1)
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



def downsample(data_dir,f,n):
    return 0
