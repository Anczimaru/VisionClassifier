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

#main variables to change to separate file later on
root_dir = "."
data_dir = os.path.join(root_dir,'data')
data_CHIC_dir = os.path.join(data_dir,'data_CHIC')
edges_dir = os.path.join(data_dir,"edges")
laplacian_dir = os.path.join(data_dir,"laplacian")
sobel_x_dir = os.path.join(data_dir,"sobel_x")
sobel_y_dir = os.path.join(data_dir,"sobel_y")
feature_dir = os.path.join(data_dir, "feature")
feature_dir_bf = os.path.join(feature_dir,"BF")
feature_dir_flann = os.path.join(feature_dir,"FLANN")
directories = (data_dir, edges_dir, laplacian_dir,sobel_x_dir, sobel_y_dir, feature_dir, feature_dir_bf, feature_dir_flann)
s_base_edge_name = "baseline_edge_picture.jpg"
s_base_laplace_name = "baseline_laplace_picture.jpg"
s_base_sobel_x_name = "baseline_sobelx_picture.jpg"
s_base_sobel_y_name = "baseline_sobely_picture.jpg"
checker_img_path = os.path.join(data_dir,'checker.JPG')


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
@timing
def main(debug_mode=0, full_run = 0):
    for f in directories:
        if not os.path.exists(f):
            os.mkdir(f)
            s = "Making directory {}".format(f)
    if (full_run == 1):
        #IF NO FILES DO EDGE DETECTION
        prepare_data(picture.canny_edge_detector, data_CHIC_dir, edges_dir, s_base_edge_name)

        prepare_data(picture.laplacian, data_CHIC_dir, laplacian_dir, s_base_laplace_name)

        prepare_data(picture.sobel_x, data_CHIC_dir, sobel_x_dir, s_base_sobel_x_name)

        prepare_data(picture.sobel_y, data_CHIC_dir, sobel_y_dir, s_base_sobel_y_name)

        with open("log_edge.txt","w") as log_file:
            print("Evaluating edges")
            evaluate_union(edges_dir, s_base_edge_name, log_file)

        with open("log_laplacian.txt","w") as log_file:
            print("Evaluating laplacian")
            evaluate_union(laplacian_dir,s_base_laplace_name,log_file)

        with open("log_sobel_x.txt","w") as log_file:
            print("Evaluating sobel_x")
            evaluate_union(sobel_x_dir,s_base_sobel_x_name,log_file)


        with open("log_sobel_y.txt","w") as log_file:
            print("Evaluating sobel_y")
            evaluate_union(sobel_y_dir,s_base_sobel_y_name,log_file)


        log_file.close()
        list = os.listdir(data_CHIC_dir)

        for f in list:
            picture.feature_matcher(data_CHIC_dir, feature_dir, f, checker_img_path, mode = 1)
            picture.feature_matcher(data_CHIC_dir, feature_dir, f, checker_img_path, mode = 2)
    merge_results()
    show_result_on_picture("Level10(without_fog)")
    show_result_on_picture("Level01")
    show_result_on_picture("Level05")
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

################################################################################
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

@timing
def evaluate_union(src_dir, baseline_picture,log_file, debug_mode = 0):

    #ESTABLISH BASELINE
    #print("Preparing baseline")
    base_img = cv2.imread(os.path.join(data_dir,baseline_picture))
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
    #EXTRACT UNION
    for f in list:
        temp_pic = cv2.imread(os.path.join(src_dir,f))
        #calculate union
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

def show_result_on_picture(index):
    for f in os.listdir(data_CHIC_dir):
        level = find_level_unmerged(f)
        if level == index:
            with open("merged_logs.txt", "r") as log_file:
                lines = log_file.readlines()
                for line in lines:
                    if find_level_merged(line) == level:
                        print(line)
                        classification = find_classification(line)
                        img = cv2.imread(os.path.join(data_CHIC_dir,f))
                        picture.show_opened_image(img, caption = classification)




    return 0






def classify(value):
    if value >= 90:
        s = "Very Good visibility"
    elif ((value < 90) and (value >=75)):
        s = "Good visibility"
    elif ((value < 75) and (value >=50)):
        s = "Foggy"
    elif ((value < 50) and (value >20)):
        s = "Heavy Fog"
    else:
        s = "Extremly Bad Fog"
    return s

def find_percentage(line):
    result = float(re.search("result of (.+?)%", line).group(1))
    return result

def find_classification(line):
    result = re.search(" which means (.+?) \n", line).group(1)
    return result

def find_level_unmerged(line):
    try:
        result = re.search("IM_(.+?).jpg",  line).group(1)
    except AttributeError as e:
        result = "Original(without_fog)"
    return result

def find_level_merged(line):
    try:
        result = re.search("(.+?) average",line).group(1)
    except AttributeError as e:
        result = "Original(without_fog)"
    return result

def merge_results():
    res_list =np.zeros(21) # list for results
    with open("merged_logs.txt", "w") as log_file:
        for f in os.listdir(root_dir):
            if ((f.endswith('.txt')) and (f!="merged_logs.txt")):
                with open(os.path.join(root_dir,f)) as opened_logs:
                    whole_text = list(opened_logs)
                    for i in range(len(whole_text)):
                        try:
                            found = find_percentage(whole_text[i])
                            res_list[i]+=found
                        except Exception as e:
                            print(e)

        res_list= res_list/4
        file_list = os.listdir(data_CHIC_dir)
        file_list.sort()
        i=0
        for f in file_list:
            try:
                found = find_level_unmerged(f)
                description = classify(res_list[i])
                s = ("{} average result of {}% which means {} \n".format(found,res_list[i], description))
                log_file.write(s)
                i+=1
            except Exception as e:
                print(e)


if __name__ == '__main__':

    main()

#RESIZE EXAMPLE
#f = cv2.imread(os.path.join(data_dir,f),-1)
#f = cv2.resize(f, (720,480), interpolation = cv2.INTER_CUBIC)
#cv2.imwrite("new.jpg",f)
#picture.segmentation(data_dir,f,debug_mode)
#picture.canny_edge_detector(data_dir,f,debug_mode)
