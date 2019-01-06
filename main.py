import picture
import numpy as np
import sys
import os
from PIL import Image
import cv2

#main variables to change to separate file later on
root_dir = "."
data_dir = os.path.join(root_dir,'data_CHIC')
edges_dir = os.path.join(root_dir,"edges")



def main(debug_mode=0):
    avg_img = np.zeros((1200,1800,3))
    FLAG_1ST_MATRIX = 0
    list = os.listdir(edges_dir)
    for f in list:
        a = int("".join(filter(str.isdigit, f)))
        if (a == 1 or a==110):
            print(f)
            temp_img = cv2.imread(os.path.join(edges_dir,f))
            if debug_mode ==1:
                picture.show_opened_image(temp_img)
            if FLAG_1ST_MATRIX == 0:
                avg_img = temp_img
                FLAG_1ST_MATRIX+=1
            else:
                avg_img = whiteunion3d(avg_img,temp_img)
    print("Done")
    picture.show_opened_image(avg_img)


def whiteunion3d(img1, img2):
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



if __name__ == '__main__':

    main()

#RESIZE EXAMPLE
#f = cv2.imread(os.path.join(data_dir,f),-1)
#f = cv2.resize(f, (720,480), interpolation = cv2.INTER_CUBIC)
#cv2.imwrite("new.jpg",f)
#picture.segmentation(data_dir,f,debug_mode)
#picture.canny_edge_detector(data_dir,f,debug_mode)
