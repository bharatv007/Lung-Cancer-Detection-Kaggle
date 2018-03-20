import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
import scipy.misc

working_path = "/home/bharatv/Link to Luna_Data/full_data_mskextraction/"
dest=working_path
file_list=glob(working_path+"images_*.npy")

for img_file in file_list:

    imgs_to_process = np.load(img_file).astype(np.float64)
    print "on image", img_file
    for i in range(len(imgs_to_process)):
        img = imgs_to_process[i]

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

        eroded = morphology.erosion(thresh_img,np.ones([4,4]))
        dilation = morphology.dilation(eroded,np.ones([10,10]))

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
        imgs_to_process[i] = mask
    np.save(img_file.replace("images","lungmask"),imgs_to_process)


#
#    Here we're applying the masks and cropping and resizing the image
#


file_list=glob(working_path+"lungmask_*.npy")
file_list2=file_list[0:len(file_list)/2]
file_list3=file_list[len(file_list)/2:len(file_list)]
out_images = []
out_nodemasks=[]
for fname in file_list2:
    print "working on file ", fname
    imgs_to_process = np.load(fname.replace("lungmask","images"))
    masks = np.load(fname)
    node_masks = np.load(fname.replace("lungmask","masks"))
    for i in range(len(imgs_to_process)):
        if np.max(masks[i])==1:
            mask = masks[i]
            node_mask = node_masks[i]
            img = imgs_to_process[i]

            img=img*mask;

            new_mean = np.mean(img[mask>0])
            new_std = np.std(img[mask>0])
            #
            #  Pulling the background color up to the lower end
            #  of the pixel range for the lungs
            #
            old_min = np.min(img)       # background color
            img[img==old_min] = new_mean-1.2*new_std   # resetting backgound color
            img = img-new_mean
            img = img/new_std
            #make image bounding box  (min row, min col, max row, max col)
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
            width = max_col-min_col
            height = max_row - min_row
            if width > height:
                max_row=min_row+width
            else:
                max_col = min_col+height
            #
            # cropping the image down to the bounding box for all regions
            # (there's probably an skimage command that can do this in one line)
            #
            img = img[min_row:max_row,min_col:max_col]
            mask =  mask[min_row:max_row,min_col:max_col]
            if max_row-min_row <5 or max_col-min_col<5:  # skipping all images with no god regions
                pass
            else:
                mean = np.mean(img)
                img[mask>0] = img[mask>0] - new_mean
                min = np.min(img[mask>0])
                max = np.max(img[mask>0])
                img = img/new_std
            new_img=img
            new_img = resize(img,[512,512])
            new_img=resize(img,[128,128])

            mean = np.mean(new_img)
            new_img = new_img - mean
            min = np.min(new_img)
            max = np.max(new_img)
            new_img = new_img / (max - min)
            new_node_mask = resize(node_mask[min_row:max_row,min_col:max_col],[512,512])
            new_node_mask=np.where(node_mask>0,1,0)
            out_images.append(new_img)
            out_nodemasks.append(new_node_mask)

num_images = len(out_images)
print num_images
#
#  Writing out images and masks as 1 channel arrays for input into network
#
final_images = np.ndarray([num_images,1,512,512],dtype=np.float32)
final_masks = np.ndarray([num_images,1,512,512],dtype=np.float32)
for i in range(num_images):
    final_images[i,0] = out_images[i]
    final_masks[i,0] = out_nodemasks[i]


rand_i = np.random.choice(range(num_images),size=num_images,replace=False)
test_i = int(0.2*num_images)
np.save("/home/bharatv/Link to Luna_Data/full_data_mskextraction/traintest_data/segmented_lungs_zscore/trainImages_seglungs1.npy",final_images[rand_i[test_i:]])
np.save("/home/bharatv/Link to Luna_Data/full_data_mskextraction/traintest_data/segmented_lungs_zscore/trainMasks_seglungs1.npy",final_masks[rand_i[test_i:]])
np.save("/home/bharatv/Link to Luna_Data/full_data_mskextraction/traintest_data/segmented_lungs_zscore/testImages_seglungs1.npy",final_images[rand_i[:test_i]])
np.save("/home/bharatv/Link to Luna_Data/full_data_mskextraction/traintest_data/segmented_lungs_zscore/testMasks_seglungs1.npy",final_masks[rand_i[:test_i]])


