import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage as ndimage
from skimage.transform import resize

import scipy.misc
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam

from keras import backend as K
import matplotlib.pyplot as plt
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans


from skimage import measure, morphology, segmentation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 1.

## loading scans of patient in path
def load_scan(image_path):
    slices = [dicom.read_file(image_path + '/' + s) for s in os.listdir(image_path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    scan_images=np.stack(s._pixel_data_numpy() for s in slices)
    scan_images = scan_images.astype(np.int16)
    return slices,scan_images


def pixels_HU(scans,scan_images):
    intercept=scans[0].RescaleIntercept
    slope=scans[0].RescaleSlope
    # scan_images[scan_images==-2000]=0
    if slope!=1:
        print 'slope is not 1'
        scan_images=slope*scan_images.astype(np.float64)
        scan_images=scan_images.astype(np.int16)

    scan_images+= np.int16(intercept)

    return np.array(scan_images,dtype=np.int16)


def generate_markers(image):
    # Creation of the internal Marker
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    # Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    # Creation of the Watershed Marker matrix
    marker_watershed = np.zeros((512, 512), dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128

    return marker_internal, marker_external, marker_watershed


def seperate_lungs(image):
    # Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(image)

    # Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)

    # Watershed algorithm
    watershed = morphology.watershed(sobel_gradient, marker_watershed)

    # Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(3, 3))
    outline = outline.astype(bool)

    # Performing Black-Tophat Morphology for reinclusion
    # Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    # Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)

    # Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    # Close holes in the lungfilter
    # fill_holes is not used here, since in some slices the heart would be reincluded by accident
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5, 5)), iterations=3)

    # Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    segmented = np.where(lungfilter == 1, image,-2000 * np.ones((512, 512)))

    return segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_unet():
	inputs = Input((1, 512, 512))
	conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(inputs)
	conv1 = Dropout(0.2)(conv1)
	conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool1)
	conv2 = Dropout(0.2)(conv2)
	conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool2)
	conv3 = Dropout(0.2)(conv3)
	conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool3)
	conv4 = Dropout(0.2)(conv4)
	conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool4)
	conv5 = Dropout(0.2)(conv5)
	conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(conv5)

	up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
	conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(up6)
	conv6 = Dropout(0.2)(conv6)
	conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv6)

	up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
	conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up7)
	conv7 = Dropout(0.2)(conv7)
	conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv7)

	up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
	conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up8)
	conv8 = Dropout(0.2)(conv8)
	conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv8)

	up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
	conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up9)
	conv9 = Dropout(0.2)(conv9)
	conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv9)

	conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

	model = Model(input=inputs, output=conv10)
	model.summary()
	model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

	return model



base_folder='/home/bharatv/Desktop/Term2/ECE 657/Kaggle/Lung Cancer/'
df_train = pd.read_csv(base_folder+'/stage1_labels.csv')
# print df_train
patient_list=os.listdir(base_folder+'/sample_images')
patient_list.sort() ## sorting the patients alphabetically,doesnt serve any purpose
cancer_id=[]


series=df_train['cancer'][df_train['id'].isin(np.intersect1d(df_train['id'], np.array(patient_list)))]

patient_list_cancer=np.array(series)
patient_list_cancer=np.insert(patient_list_cancer,[5],[0])
print patient_list_cancer
num=6
image_path=base_folder+'/sample_images/'+patient_list[num]
print 'cancer id', patient_list_cancer[num]
scans,scan_images=load_scan(image_path) #sorted images of patient based on instance number
# scans= dicaom file, scan_images= stacked images
scan_images_hu=pixels_HU(scans,scan_images) #converts to hu scale
print scan_images.shape
plt.figure(1)
j=1
for i in range(1,280,20):
    plt.subplots(22)
    plt.imshow(scan_images[i])
    j=j+1



model=get_unet()
model.load_weights('/home/bharatv/Downloads/weights_halfdata.19.hdf5')
imgs_mask_predict = np.ndarray([len(scan_images), 1, 512, 512], dtype=np.float32)

for i in range(90,len(scan_images)):

    img = scan_images[i]
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std

    middle = img[100:400,100:400]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    img[img==max]=mean
    img[img==min]=mean

    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image

    #
    eroded = morphology.erosion(thresh_img,np.ones([4,4]))
    dilation = morphology.dilation(eroded,np.ones([10,10]))

    #
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
            good_labels.append(prop.label)
    mask = np.ndarray([512,512],dtype=np.int8)
    mask[:] = 0

    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    img=mask*scan_images[i]
    new_mean = np.mean(img[mask > 0])
    new_std = np.std(img[mask > 0])

    #
    old_min = np.min(img)  # background color
    img[img == old_min] = new_mean - 1.2 * new_std  # resetting backgound color
    img = img - new_mean
    img = img / new_std

    labels = measure.label(mask)
    regions = measure.regionprops(labels)

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

    img = img[min_row:max_row, min_col:max_col]
    mask = mask[min_row:max_row, min_col:max_col]
    if max_row - min_row < 5 or max_col - min_col < 5:  # skipping all images with no god regions
        pass
    else:
        mean = np.mean(img)
        img = img - mean
        min = np.min(img)
        max = np.max(img)
        img = img - min / (max - min)
        new_img = resize(img, [512, 512])




    fig = plt.figure(0)
    ax1 = fig.add_subplot(1, 2, 1)

    ax1.imshow(mask*scan_images[i], cmap=plt.cm.bone)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(scan_images[i], cmap=plt.cm.bone)
    plt.show()


for i in range(0,len(scan_images)):
    img=scan_images[i]
    mean = np.mean(img)
    img = img - mean
    min = np.min(img)
    max = np.max(img)
    img = img / (max - min)
    new_img = img
    img_test1 = np.ndarray([1, 1, 512, 512], dtype=np.float32)
    img_test1[0,0] = img
    imgs_mask_predict[i] = model.predict(img_test1, verbose=1)[0]
    # scipy.misc.imsave("predicted_mask_%04d.png" % (i), imgs_mask_predict[i,0])
    fig=plt.figure(0)
    ax1=fig.add_subplot(1, 2, 1)
    # imgs_mask_test[i, 0, :, :] = imgs_mask_test[i, 0, :, :] * pow(10, 20)
    ax1.imshow(np.where(imgs_mask_predict[i, 0, :, :] > 0.01, 1, 0) * img_test1[0, 0, :, :], cmap=plt.cm.bone)
    ax2=fig.add_subplot(1, 2, 2)
    ax2.imshow(img,cmap=plt.cm.bone)
    # extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("./Not_Cancerous/predicted_mask_%04d.png" % (i))
    # plt.show()


