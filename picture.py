from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
import cv2

root_dir = "."
data_dir = os.path.join(root_dir,'data')
edges_dir = os.path.join(root_dir,"edges")


def show_opened_image(image):

    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def canny_edge_detector(data_dir,f, debug_mode=0):

    img = cv2.imread(os.path.join(data_dir,f),-1)
    edges = cv2.Canny(img,100,200)
    if debug_mode == 1:
        show_opened_image(edges)
    print(edges.shape)
    s = "edges_{}".format(f)
    print(s)
    tmp_img = Image.fromarray(edges)
    tmp_img.save(os.path.join(edges_dir, s))

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

def downsample(data_dir,f,n):
    return 0
