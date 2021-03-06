import picture
import numpy as np
import config
import os
import re
import cv2

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

#Find percentage based on log contruction
def find_percentage(line):
    result = float(re.search("result of (.+?)%", line).group(1))
    return result

#Find classification from merged log file
def find_classification(line):
    result = re.search(" which means (.+?) \n", line).group(1)
    return result

#Find level/unique id in unmerged logs
def find_level_unmerged(line):
    try:
        result = re.search("IM_(.+?).jpg",  line).group(1)
    except AttributeError as e:
        result = "Original(without_fog)"
    return result


#Find unique id in merged log file
def find_level_merged(line):
    try:
        result = re.search("(.+?) average",line).group(1)
    except AttributeError as e:
        result = "Original(without_fog)"
    return result


def merge_results(root_dir, data_dir):
    res_list =np.zeros(len(os.listdir(data_dir))) # list for results
    with open(os.path.join(root_dir,"merged_logs.txt"), "w") as log_file: #open/create logs
        for f in os.listdir(root_dir): #get file list
            if ((f.endswith('.txt')) and (f!="merged_logs.txt")): #get .txt so log files from list
                with open(os.path.join(root_dir,f)) as opened_logs: #open proper logs
                    whole_text = list(opened_logs) #get logs to list, by lines
                    for i in range(len(whole_text)): #iterate over lines
                        try:  
                            found = find_percentage(whole_text[i]) #find percentage
                            res_list[i]+=found #add it to list
                        except Exception as e:
                            print(e)

        res_list= res_list/4 #average results by number of files
        file_list = os.listdir(data_dir) #get list of photos
        file_list.sort() #sort indexes of photos
        i=0 #iterator
        for f in file_list: #save results in proper order
            try:
                found = find_level_unmerged(f)
                description = classify(res_list[i]) #classify obtained results
                s = ("{} average result of {}% which means {} \n".format(found,res_list[i], description))
                log_file.write(s)
                i+=1
            except Exception as e:
                print(e)

def show_result_on_picture(index, src_dir, root_dir= config.root_dir):
    #Show picture with name format ".....IM_index.jpg" where index is unique id of photo 
    for f in os.listdir(src_dir):
        level = find_level_unmerged(f)
        if level == index:
            with open(os.path.join(root_dir,"merged_logs.txt"), "r") as log_file:
                lines = log_file.readlines()
                for line in lines:
                    if find_level_merged(line) == level:
                        print(line)
                        classification = find_classification(line)
                        img = cv2.imread(os.path.join(src_dir,f))
                        picture.show_opened_image(img, caption = classification)



    return 0





def main():


    merge_results(config.root_dir, config.image_dir)
    show_result_on_picture("Level10(without_fog)", config.image_dir)
    show_result_on_picture("Level1", config.image_dir)
    show_result_on_picture("Level5", config.image_dir)
    return 0






if __name__ == "__main__":
    main()
