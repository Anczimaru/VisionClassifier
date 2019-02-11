import picture
import os
from PIL import Image
import cv2
from functools import wraps
from time import time
import shutil
import config

###############################################################################
#wrapper for time measurment
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
    ###### CHECK IF DIRECTORIES EXIST IF NOT CREATE THEM
    flag_new_dir = 0
    for f in config.directories:
        if not os.path.exists(f):
            flag_new_dir =1
            os.mkdir(f)
            s = "Making directory {}".format(f)
            print(s)
    if (flag_new_dir == 1):
        print("Please place data in proper folders, then run program again")
        return 0


#Preparation of data by first preparing photos obtained by filter transformations
#Then get baseline for given filter by calculating intersection between baseline_original photos

    #Canny
    prepare_data(picture.canny_edge_detector, config.image_dir, config.edges_dir, config.data_dir)
    baseline_avg(picture.canny_edge_detector, config.baseline_src_dir, config.data_dir, config.s_base_edge_name)

    #Laplacian
    prepare_data(picture.laplacian, config.image_dir, config.laplacian_dir, config.data_dir)
    baseline_avg(picture.laplacian, config.baseline_src_dir, config.data_dir, config.s_base_laplace_name)

    #Sobel_x
    prepare_data(picture.sobel_x, config.image_dir, config.sobel_x_dir, config.data_dir)
    baseline_avg(picture.sobel_x, config.baseline_src_dir, config.data_dir, config.s_base_sobel_x_name)

    #Sobel_y
    prepare_data(picture.sobel_y, config.image_dir, config.sobel_y_dir, config.data_dir)
    baseline_avg(picture.sobel_y, config.baseline_src_dir, config.data_dir, config.s_base_sobel_y_name)

#EVALUATE INTERSECTIONS
    #Canny
    with open(os.path.join(config.root_dir,"log_edge.txt"),"w") as log_file:
        print("Evaluating edges")
        evaluate_intersection(config.edges_dir, config.s_base_edge_name, log_file, config.data_dir)
    #Laplacian
    with open(os.path.join(config.root_dir,"log_laplacian.txt"),"w") as log_file:
        print("Evaluating laplacian")
        evaluate_intersection(config.laplacian_dir, config.s_base_laplace_name, log_file, config.data_dir)
    #Sobel_x
    with open(os.path.join(config.root_dir,"log_sobel_x.txt"),"w") as log_file:
        print("Evaluating sobel_x")
        evaluate_intersection(config.sobel_x_dir, config.s_base_sobel_x_name, log_file, config.data_dir)
    #Sobel_y
    with open(os.path.join(config.root_dir,"log_sobel_y.txt"),"w") as log_file:
        print("Evaluating sobel_y")
        evaluate_intersection(config.sobel_y_dir, config.s_base_sobel_y_name, log_file, config.data_dir)
        
        #Close any opened log file
        log_file.close()

    #end message
    print("Done")

################################################################################
def prepare_data(function, src_dir, dst_dir, data_dir, debug_mode = 0):
    src_photos = os.listdir(src_dir)
    dst_photos = os.listdir(dst_dir)
    if (len(dst_photos) < len(src_photos)):
        #check if there is aproppriate no. of photos, if not rm whole dir, and rerun creation procedure
        shutil.rmtree(dst_dir)
        os.mkdir(dst_dir)
        s_msg = "{} , preparing data".format(dst_dir)
        print(s_msg)
        list = os.listdir(src_dir)
        for f in list:
            function(src_dir, dst_dir, f, debug_mode = debug_mode)


################################################################################
def baseline_avg(function, src_dir, dst_dir, name,debug_mode = 0):
    baseline_picture = os.path.join(dst_dir,name)
    if not os.path.exists(baseline_picture):
        #Check if baseline exits, if not create it
        if debug_mode == 1:
            print(function)
        ##### CREATE BASELINE NOW
        FLAG_1ST_MATRIX = 0
        for f in os.listdir(src_dir): #obtain baseline_orginal list of photos
            temp_img = function(src_dir, dst_dir, f, save =0, debug_mode = debug_mode)
            if FLAG_1ST_MATRIX == 0:
                #for first photo just copy it
                avg_img = temp_img
                FLAG_1ST_MATRIX+=1
            else:
                #for every other photo obtain intersection
                avg_img = picture.White_Intersection_3d(avg_img,temp_img, debug_mode = debug_mode)

        print("Extracted baseline picture")
        temp_img = Image.fromarray(avg_img)
        temp_img.save(os.path.join(dst_dir, name))

################################################################################
@timing
def evaluate_intersection(src_dir, baseline_picture,log_file,data_dir, debug_mode = 0):
    #
    base_img = cv2.imread(os.path.join(data_dir,baseline_picture))
    if (debug_mode == 1):
        print(os.path.join(data_dir,baseline_picture))
    #Get single channeled photo
    base_img_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    base_pixels = 0
    #get number of baseline pixels
    for x in range(base_img_gray.shape[0]):
        for y in range(base_img_gray.shape[1]):
            if base_img_gray[x,y] >=1: base_pixels+=1
    if debug_mode == 1:
        picture.show_opened_image(base_img)
    print("Finished counting baseline pixels")

    #GET DATA
    list = os.listdir(src_dir)
    list.sort() # LIST SORT == IMPORTANT THING!
    res_names = []
    res_values = []
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
        #Forward results via lists
        res_names.append(f)
        res_values.append(similarity_percentage)



    #find proper multiplier based on highest match
    multiplier = 100/max(res_values)

    #WRITE RESULTS TO LOG FILES
    for i in range(len(res_names)):
        s_results = ("For picture {} we got result of {}% against baseline \n".format(res_names[i],res_values[i]*multiplier))
        log_file.write(s_results)


    return 0




if __name__ == '__main__':

    main()
