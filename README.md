# 3D-MRI-segmentation
# Requries
* numpy
* pandas
* `numpy`
* `tensorflow` version 2.0
* `nibabel` version 3.1.1
# Using
Setup the `info.json` which should be included the `train_tumor_path`, `train_label_path`, etc. Then you should run the `train.py` that will build the segmentation model and save the model automatically. The model weights and loss will be recorded in `weight_dir/model_weight` and `log_dir+/model_log.csv`. `test.ipynb` will use the testing dataset to validate the model performance. This file will give us a quick look.

## Notice ###
The training, validation and testing dataset should be separated into different independence folder.

# Modules
## Model Training Loss -- loss.py
"""
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
"""
* dice_coef
  the dice probability of the training model.
  * y_true, y_pred
  * epsilon: default `0.00001`. A penalty of model training.
* dice_coef_loss  
  The residual between model prediction and ground truth.
  * y_true, y_pred
## Segmentation Model -- model.py  
## Image Dataset Generate -- pre_processing.py
* normalize
  * input_array: numpy.array() or array-like. 
* get_random 
  This function is one kind of generator. That means the image will not be read until the program calls for data. Just put the image and label data in a different folder.
  * name_list: list. Image file name list
  * image_path
  * label_path
  * batch: set the number of using images in each iteration. This parameter will be used only if the `steps` is `None`.
  * steps: default `None`. Set the number of model training iterators.
  * shuffle: default `True`. Notice: This parameter only can shuffle dataset which is less than 1000 images.
  * img_size: default `128`.
