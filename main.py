import picture
import numpy as np
import sys
import os
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from functools import wraps
from time import time
import re
import config
import demo


###############################################################################
def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print ('Elapsed time: {}'.format(end-start))
        return result
    return wrapper


################################################################################


def main():
    for f in config.directories:
        if not os.path.exists(f):
            os.mkdir(f)
            s = "Making directory {}".format(f)
            print("Please place data in proper folders, then run program again")
            return 0

    prepare_data(picture.canny_edge_detector, config.image_dir, config.edges_dir, config.data_dir)

    prepare_data(picture.laplacian, config.image_dir, config.laplacian_dir, config.data_dir)

    prepare_data(picture.sobel_x, config.image_dir, config.sobel_x_dir, config.data_dir)

    prepare_data(picture.sobel_y, config.image_dir, config.sobel_y_dir, config.data_dir)


    with open(os.path.join(config.root_dir,"log_edge.txt"),"w") as log_file:
        print("Evaluating edges")
        evaluate_intersection(config.edges_dir, config.s_base_edge_name, log_file, config.data_dir)

    with open(os.path.join(config.root_dir,"log_laplacian.txt"),"w") as log_file:
        print("Evaluating laplacian")
        evaluate_intersection(config.laplacian_dir, config.s_base_laplace_name, log_file, config.data_dir)

    with open(os.path.join(config.root_dir,"log_sobel_x.txt"),"w") as log_file:
        print("Evaluating sobel_x")
        evaluate_intersection(config.sobel_x_dir, config.s_base_sobel_x_name, log_file, config.data_dir)


    with open(os.path.join(config.root_dir,"log_sobel_y.txt"),"w") as log_file:
        print("Evaluating sobel_y")
        evaluate_intersection(config.sobel_y_dir, config.s_base_sobel_y_name, log_file, config.data_dir)

        log_file.close()


    print("Done")

################################################################################
def prepare_data(function, src_dir, dst_dir, s_base_pic_name, data_dir, debug_mode = 0):
    if len(os.listdir(dst_dir)) == 0:
        s_msg = "Empty {} , preparing data".format(dst_dir)
        print(s_msg)
        list = os.listdir(src_dir)
        for f in list:
            function(src_dir, dst_dir, f, debug_mode = debug_mode)


################################################################################
def baseline_avg():
    for s_base_pic_name in config.baseline_names:
        baseline_picture = os.path.join(data_dir,s_base_pic_name)
        if not os.path.exists(baseline_picture):
            baseline_avg(dst_dir, data_dir, s_base_pic_name, debug_mode=debug_mode)


################################################################################
@timing
def evaluate_intersection(src_dir, baseline_picture,log_file,data_dir, debug_mode = 0):

    #ESTABLISH BASELINE
    #print("Preparing baseline")
    base_img = cv2.imread(os.path.join(data_dir,baseline_picture))
    print(os.path.join(data_dir,baseline_picture))
    base_img_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    base_pixels = 0
    #get number of baseline pixels
    for x in range(base_img_gray.shape[0]):
        for y in range(base_img_gray.shape[1]):
            if base_img_gray[x,y] >=1: base_pixels+=1
    if debug_mode == 1:
        picture.show_opened_image(base_img)
    print("Established baseline_picture")

    #GET DATA
    list = os.listdir(src_dir)
    list.sort() # LIST SORT == IMPORTANT THING!
    similarity_multiplier = 0
    res_list = []
    #EXTRACT intersection
    for f in list:
        temp_pic = cv2.imread(os.path.join(src_dir,f))
        #calculate intersection
        res_pic = cv2.bitwise_and(base_img,temp_pic)
        res_pic_gray = cv2.cvtColor(res_pic, cv2.COLOR_BGR2GRAY)
        if debug_mode == 1:
            picture.show_opened_image(res_pic_gray)
        similarity = 0
        #get similarity
        for x in range(res_pic_gray.shape[0]):
            for y in range(res_pic_gray.shape[1]):
                if res_pic_gray[x,y] >=1: similarity+=1

        similarity_percentage = similarity/base_pixels

        #calculate coefficient
        a = int("".join(filter(str.isdigit, f)))
        if (a == 1 or a==110):
            similarity_multiplier +=similarity_percentage

        res_list.append((f,similarity_percentage))
    similarity_multiplier = 100/(similarity_multiplier/3)
    print(similarity_multiplier)

    #WRITE RESULTS TO LOG FILES
    for i in range(len(res_list)):
        a = res_list[i]
        s_results = ("For picture {} we got result of {}% against baseline \n".format(a[0],a[1]*similarity_multiplier))
        log_file.write(s_results)


    return 0




if __name__ == '__main__':

    main()
