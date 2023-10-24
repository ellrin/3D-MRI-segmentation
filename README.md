# 3D-MRI-segmentation
# Modules
## Image Dataset Generate -- pre_processing.py
### Function
* normalize()
  * input_array: numpy.array() or array-like. 
* get_random()  
  This function is one kind of generator. That means the image will not be read until the program calls for data. Just put the image and label data in a different folder.
  * name_list: list. Image file name list
  * image_path
  * label_path
  * batch: set the number of using images in each iteration. This parameter will be used only if the `steps` is `None`.
  * steps: default `None`. Set the number of model training iterators.
  * shuffle: default `True`. Notice: This parameter only can shuffle dataset which is less than 1000 images.
  * img_size: default `128`.
