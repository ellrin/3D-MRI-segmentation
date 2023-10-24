# 3D-MRI-segmentation
# Modules
## Image Dataset Generate -- pre_processing.py
### Function
* normalize()
  * input_array: numpy.array or array like. 
* get_random()
  This function 
  * name_list
  * image_path
  * label_path
  * batch: set the number of using images in each iterater. This parameters will be used only if the steps is `None`.
  * steps: default `None`. set the number of model traininig iterater. .
  * shuffle: default `True`. Notice: This parameter only can shuflle dataset which is less than 1000 images.
  * img_size: default `128`.
