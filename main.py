import picture
import numpy as np
import sys
import os
from PIL import Image
import cv2
from matplotlib import pyplot as plt

#main variables to change to separate file later on
root_dir = "."
data_dir = os.path.join(root_dir,'data')
data_CHIC_dir = os.path.join(data_dir,'data_CHIC')
edges_dir = os.path.join(data_dir,"edges")
laplacian_dir = os.path.join(data_dir,"laplacian")
feature_dir = os.path.join(data_dir, "feature")
feature_dir_bf = os.path.join(feature_dir,"BF")
feature_dir_flann = os.path.join(feature_dir,"FLANN")
directories = (data_dir, edges_dir, laplacian_dir, feature_dir, feature_dir_bf, feature_dir_flann)
s_base_edge_name = "baseline_edge_picture.jpg"
s_base_laplace_name = "baseline_laplace_picture.jpg"
checker_img_path = os.path.join(data_dir,'checker.JPG')

################################################################################
def main(debug_mode=0):
    for f in directories:
        if not os.path.exists(f):
            os.mkdir(f)
            s = "Making directory {}".format(f)
    """
    #IF NO FILES DO EDGE DETECTION
    prepare_data(picture.canny_edge_detector, data_CHIC_dir, edges_dir, s_base_edge_name)

    prepare_data(picture.laplacian,data_CHIC_dir,laplacian_dir, s_base_laplace_name)
    log_file = open("log_edge.txt","w")
    #evaluate_union(edges_dir, s_base_edge_name, log_file)
    log_file.close()
    log_file = open("log_laplacian.txt","w")
    evaluate_union(laplacian_dir,s_base_laplace_name,log_file)
    log_file.close()
    """
    list = os.listdir(data_CHIC_dir)

    for f in list:
        picture.feature_matcher(data_CHIC_dir, feature_dir, f, checker_img_path, mode = 1)
        picture.feature_matcher(data_CHIC_dir, feature_dir, f, checker_img_path, mode = 2)
    print("Done")

################################################################################

def prepare_data(function, src_dir, dst_dir,s_base_pic_name, debug_mode = 0):
    if len(os.listdir(dst_dir)) == 0:
        s_msg = "Empty {} , preparing data".format(dst_dir)
        print(s_msg)
        list = os.listdir(src_dir)
        for f in list:
            function(src_dir, dst_dir, f, debug_mode = debug_mode)

        #get baseline for futher experiment
        baseline_picture = os.path.join (data_dir,s_base_pic_name)
        if not os.path.exists(baseline_picture):
            baseline_avg_CHIC(dst_dir, data_dir, s_base_pic_name, debug_mode=debug_mode)


def baseline_avg_CHIC(src_dir, dst_dir,name, debug_mode=0):
    #script for averaging no-fog photos from downsampled CHIC dataset
    s = "Extracting {} from {} directory".format(name, src_dir)
    print(s)
    avg_img = np.zeros((1200,1800,3))
    FLAG_1ST_MATRIX = 0
    list = os.listdir(src_dir)
    for f in list:
        a = int("".join(filter(str.isdigit, f)))
        if (a == 1 or a==110):
            if debug_mode == 1:
                print(f)
            temp_img = cv2.imread(os.path.join(src_dir,f))
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
    temp_img.save(os.path.join(dst_dir, name))


def evaluate_union(src_dir, baseline_picture,log_file, debug_mode = 0):

    #ESTABLISH BASELINE
    print("Preparing baseline")
    base_img = cv2.imread(os.path.join(data_dir,baseline_picture))
    base_img_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    base_pixels = 0
    for x in range(base_img_gray.shape[0]):
        for y in range(base_img_gray.shape[1]):
            if base_img_gray[x,y] >=1: base_pixels+=1
    if debug_mode == 1:
        picture.show_opened_image(base_img)
    print("Established baseline_picture")

    #GET DATA
    list = os.listdir(src_dir)
    list.sort()
    for f in list:
        #EXTRACT UNION
        temp_pic = cv2.imread(os.path.join(src_dir,f))
        res_pic = cv2.bitwise_and(base_img,temp_pic)
        res_pic_gray = cv2.cvtColor(res_pic, cv2.COLOR_BGR2GRAY)
        if debug_mode == 1:
            picture.show_opened_image(res_pic_gray)
        similarity = 0
        for x in range(res_pic_gray.shape[0]):
            for y in range(res_pic_gray.shape[1]):
                if res_pic_gray[x,y] >=1: similarity+=1

        similarity_percentage = similarity/base_pixels
        s_results = "For picture {} we got result of {}% against baseline \n".format(f,similarity_percentage)
        log_file.write(s_results)


    return 0




if __name__ == '__main__':

    main()

#RESIZE EXAMPLE
#f = cv2.imread(os.path.join(data_dir,f),-1)
#f = cv2.resize(f, (720,480), interpolation = cv2.INTER_CUBIC)
#cv2.imwrite("new.jpg",f)
#picture.segmentation(data_dir,f,debug_mode)
#picture.canny_edge_detector(data_dir,f,debug_mode)
