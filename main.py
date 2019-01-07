import picture
import numpy as np
import sys
import os
from PIL import Image
import cv2

#main variables to change to separate file later on
root_dir = "."
data_dir = os.path.join(root_dir,'data')
data_CHIC_dir = os.path.join(root_dir,'data_CHIC')
edges_dir = os.path.join(root_dir,"edges")



def main(debug_mode=0):
    #IF NO FILES DO EDGE DETECTION
    if len(os.listdir(edges_dir)) == 0:
        print("Empty edges dir, extracting data")
        list = os.listdir(data_CHIC_dir)
        for f in list:
            picture.canny_edge_detector(data_CHIC_dir, edges_dir, f, debug_mode = debug_mode)

    #get baseline for futher experiment
    baseline_picture = os.path.join (data_dir,"baseline_picture.jpg")
    if not os.path.exists(baseline_picture):
        baseline_avg_CHIC(data_dir,debug_mode=debug_mode)

    print("Done")







def baseline_avg_CHIC(dst_dir, debug_mode=0):
    #script for averaging no-fog photos from downsampled CHIC dataset
    avg_img = np.zeros((1200,1800,3))
    FLAG_1ST_MATRIX = 0
    list = os.listdir(edges_dir)
    for f in list:
        a = int("".join(filter(str.isdigit, f)))
        if (a == 1 or a==110):
            if debug_mode == 1:
                print(f)
            temp_img = cv2.imread(os.path.join(edges_dir,f))
            if debug_mode ==1:
                picture.show_opened_image(temp_img)
            if FLAG_1ST_MATRIX == 0:
                avg_img = temp_img
                FLAG_1ST_MATRIX+=1
            else:
                avg_img = picture.whiteunion3d(avg_img,temp_img,debug_mode=debug_mode)
    print("Extracted baseline picture")
    if debug_mode == 1:
        picture.show_opened_image(avg_img)

    temp_img = Image.fromarray(avg_img)
    temp_img.save(os.path.join(dst_dir,"baseline_picture.jpg"))


def evaluate_union(src_dir, baseline_picture):


    return 0



if __name__ == '__main__':

    main()

#RESIZE EXAMPLE
#f = cv2.imread(os.path.join(data_dir,f),-1)
#f = cv2.resize(f, (720,480), interpolation = cv2.INTER_CUBIC)
#cv2.imwrite("new.jpg",f)
#picture.segmentation(data_dir,f,debug_mode)
#picture.canny_edge_detector(data_dir,f,debug_mode)
