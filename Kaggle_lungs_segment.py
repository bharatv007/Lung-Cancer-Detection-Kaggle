import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
from sklearn.cluster import KMeans
from skimage.transform import resize
import os
import scipy.ndimage
import matplotlib.pyplot as plt

#from __future__ import print_function
import os

# os.environ['KERAS_BACKEND']='theano'
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32,force_device=true,lib.cnmem=0.9"#,nvcc.flags=-D_FORCE_INLINES"
 
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.models import load_model
from IPython.core.debugger import Tracer
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Some constants 
INPUT_FOLDER = '/work/vsankar/projects/kaggle_data/stage1/stage1/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 1

# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    ret = np.ndarray([len(image),1,512,512],dtype=np.float32)
    for i in range (len(image)):
        ret[i,0] = image[i]
    return ret







def get_lungs(imgs_to_process):
    for i in range(len(imgs_to_process)):
        img = imgs_to_process[i,0]
        # Standardize the pixel values
        mean = np.mean(img)
        std = np.std(img)
        img = img - mean
        img = img / std
        # Find the average pixel value near the lungs
        # to renormalize washed out images
        middle = img[100:400, 100:400]
        mean = np.mean(middle)
        max = np.max(img)
        min = np.min(img)
        # To improve threshold finding, I'm moving the
        # underflow and overflow on the pixel spectrum
        img[img == max] = mean
        img[img == min] = mean
        #
        # Using Kmeans to separate foreground (radio-opaque tissue)
        # and background (radio transparent tissue ie lungs)
        # Doing this only on the center of the image to avoid
        # the non-tissue parts of the image as much as possible
        #
        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image
        #
        # I found an initial erosion helful for removing graininess from some of the regions
        # and then large dialation is used to make the lung region
        # engulf the vessels and incursions into the lung cavity by
        # radio opaque tissue
        #
        eroded = morphology.erosion(thresh_img, np.ones([4, 4]))
        dilation = morphology.dilation(eroded, np.ones([10, 10]))
        #
        #  Label each region and obtain the region properties
        #  The background region is removed by removing regions
        #  with a bbox that is to large in either dimnsion
        #  Also, the lungs are generally far away from the top
        #  and bottom of the image, so any regions that are too
        #  close to the top and bottom are removed
        #  This does not produce a perfect segmentation of the lungs
        #  from the image, but it is surprisingly good considering its
        #  simplicity.
        #
        labels = measure.label(dilation)
        label_vals = np.unique(labels)
        regions = measure.regionprops(labels)
        good_labels = []
        for prop in regions:
            B = prop.bbox
            if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
                good_labels.append(prop.label)
        mask = np.ndarray([512, 512], dtype=np.int8)
        mask[:] = 0
        #
        #  The mask here is the mask for the lungs--not the nodes
        #  After just the lungs are left, we do another large dilation
        #  in order to fill in and out the lung mask
        #
        for N in good_labels:
            mask = mask + np.where(labels == N, 1, 0)
        mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation
        img = imgs_to_process[i,0]
        new_size = [512, 512]  # we're scaling back up to the original size of the image
        img = mask * img # apply lung mask
        #
        # renormalizing the masked image (in the mask region)
        #
        new_mean = np.mean(img[mask > 0])
        new_std = np.std(img[mask > 0])
        #
        #  Pulling the background color up to the lower end
        #  of the pixel range for the lungs
        #
        old_min = np.min(img)  # background color
        img[img == old_min] = new_mean - 1.2 * new_std  # resetting backgound color
        img = img - new_mean
        img = img / new_std
        # make image bounding box  (min row, min col, max row, max col)
        labels = measure.label(mask)
        regions = measure.regionprops(labels)
        #
        # Finding the global min and max row over all regions
        #
        min_row = 512
        max_row = 0
        min_col = 512
        max_col = 0
        for prop in regions:
            B = prop.bbox
            if min_row > B[0]:
                min_row = B[0]
            if min_col > B[1]:
                min_col = B[1]
            if max_row < B[2]:
                max_row = B[2]
            if max_col < B[3]:
                max_col = B[3]
        width = max_col - min_col
        height = max_row - min_row
        if width > height:
            max_row = min_row + width
        else:
            max_col = min_col + height
        #
        # cropping the image down to the bounding box for all regions
        # (there's probably an skimage command that can do this in one line)
        #
        img = img[min_row:max_row, min_col:max_col]
        mask = mask[min_row:max_row, min_col:max_col]
        if max_row - min_row < 5 or max_col - min_col < 5:# skipping all images with no god regions
            imgs_to_process[i, 0] = 0
            pass
        else:
            # moving range to -1 to 1 to accomodate the resize function
            mean = np.mean(img)
            img = img - mean
            min = np.min(img)
            img = img  / (max - min)
            imgs_to_process[i,0] = resize(img, [512, 512])
            # new_node_mask = resize(node_mask[min_row:max_row, min_col:max_col], [512, 512])
            # new_node_mask = (new_node_mask > 0.0).astype(np.float32)
            # out_images.append(new_img)
            # out_nodemasks.append(new_node_mask)

# model2 = get_unet()
# model2.load_weights('/home/vsankar/bharat/pretrained/fromscratch_best/weights_halfdata.best.hdf5')


# patients_folder='/work/vsankar/projects/lungCancer/'
df_train = pd.read_csv('/work/vsankar/projects/lungCancer/stage1_labels.csv')
numPatients = len(patients)
PatientsDict = {}
PatientsPredictedDict = {}
t=50
c=0
#patients[2] = '123' 
for ii in range(30):
    c=0
    PatientsPredictedDict = {}
    PatientsDict = {}
    for ij in range(t):
        i= ii*t + ij
        print(i)
        print('c = %d' %(c))
        label = df_train['cancer'][df_train['id']==patients[i]] 
        if label.empty == False:
            first_patient = load_scan(INPUT_FOLDER + patients[i])
            first_patient_pixels = get_pixels_hu(first_patient)


        

            imgs_test = np.ndarray([len(first_patient_pixels),1,512,512],dtype=np.float32)
            imgs_mask_test = np.ndarray([len(first_patient_pixels),1,512,512],dtype=np.float32)
            get_lungs(first_patient_pixels)
            # imgs_test = first_patient_pixels
            # imgs_mask_test = model2.predict(imgs_test, verbose=1)

            PatientsDict[c] = (first_patient_pixels,label)
            PatientsPredictedDict[c] = (imgs_mask_test,label)
            c=c+1
            
        
    #print('saving predict')
    np.save('/work/vsankar/projects/kaggle_segmented/_%d.npy' % (ii),PatientsDict)

    np.save('work/vsankar/projects/kaggle_segmented/PatientsPredictedDict_%d.npy' % (ii),PatientsPredictedDict)
    
