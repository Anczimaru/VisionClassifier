import os

#Config file for Vision Classifier project


_all_ = ('root_dir',"data_dir", "data_CHIC_dir",
        "edges_dir", "laplacian_dir", "sobel_x_dir", "sobel_y_dir",
        "feature_dir", "directories", " s_base_edge_name", "s_base_laplace_name"
        "s_base_sobel_x_name", "s_base_sobel_y_name")

#PATHS TO DIRECTORIES
root_dir = os.path.join(".","main")
data_dir = os.path.join(root_dir,'data')
image_dir = os.path.join(data_dir,"source_images")
baseline_src_dir = os.path.join(data_dir,"baseline_orginal")
edges_dir = os.path.join(data_dir,"edges")
laplacian_dir = os.path.join(data_dir,"laplacian")
sobel_x_dir = os.path.join(data_dir,"sobel_x")
sobel_y_dir = os.path.join(data_dir,"sobel_y")


#DIR LIST
directories = (root_dir, data_dir, image_dir, baseline_src_dir, edges_dir, laplacian_dir, sobel_x_dir, sobel_y_dir)

#Baseline pictures names
s_base_edge_name = "baseline_edge_picture.jpg"
s_base_laplace_name = "baseline_laplace_picture.jpg"
s_base_sobel_x_name = "baseline_sobelx_picture.jpg"
s_base_sobel_y_name = "baseline_sobely_picture.jpg"

baseline_names = (s_base_edge_name, s_base_laplace_name, s_base_sobel_x_name, s_base_sobel_y_name)
