
This is a project to detect lung cancer from CT scan images using Deep learning (CNN)
########Dataset#######################################

Kaggle dataset-https://www.kaggle.com/c/data-science-bowl-2017/data

LUNA dataset-https://luna16.grand-challenge.org/download/

######################################################

LUNA_mask_creation.py- code for extracting node masks from LUNA dataset
LUNA_lungs_segment.py- code for segmenting lungs in LUNA dataset and creating training and testing data
Luna_train.py-  Unet training code

Kaggle_lungs_segment.py- segmeting lungs in Kaggle Data set
kaggle_predict.py - Predicting node masks in kaggle data set using weights from Unet
kaggleSegmentedClassify.py- Classifying kaggle data  from predicted node masks 

