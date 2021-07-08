import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import imageio
import skimage.io
import nibabel as nib
import glob


def insert_segmenetions_path_to_dict(dataset, new_dataset_output_path, dataset_path, contrast_type):
    for key, value in dataset.items():
        # Get the image path, replace it with the image path from the old dataset
        # and add _roi in order to create the mask path
        path = value['Image'].split('.')                                 # split the path into a list
        path[0] = path[0].replace(new_dataset_output_path, dataset_path) # replace the new path with the old one
        path.insert(1, '_mask.')                                          # append _roi
        path = ''.join(path)
        path = path.replace('_' + contrast_type, '')                                             # join the list elements into a string

        # Add mask path from the old dataset to new dataset dictionary
        dataset[key]['Mask'] = path
        
    return dataset

def histograms_compare(images,image_names,metric=0,name=''):

    image_names = [ i.replace('data/dataset/sygrisampol_images/', '').replace('post-contrast.tif_removed_background.png', '')  for i in image_names ]
    
    histograms = [ cv2.calcHist([x], [0], None, [256], [0, 256]) for x in images]
    
    mat = np.zeros((len(images),len(images)))
    
    
    
    methods = [cv2.HISTCMP_BHATTACHARYYA ]
    
    for i in range(len(images)):
        for j in range(len(images)): 
            mat[i][j]=cv2.compareHist(histograms[i],histograms[j],methods[metric])
            
    fig,ax = plt.subplots()
    
    f = np.around(mat,2)
    
    im = ax.imshow(f)

    ax.set_yticks(np.arange(len(image_names)))
    ax.set_yticklabels(image_names)
    
    ax.set_xticks(np.arange(len(image_names)))
    ax.set_xticklabels(image_names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

    for i in range(len(image_names)):
        for j in range(len(image_names)):
            text = ax.text(j, i, f[i, j], ha="center", va="center", color="w")
  
    ax.set_title("histograms distance with clip limit "+str(name))
    
    return mat

def ssim_compare(image,images,image_names,name=''):

    image_names = [ i.replace('data/dataset/sygrisampol_images/', '').replace('post-contrast.tif_removed_background.png', '')  for i in image_names ]
   
    mat = np.zeros((len(images), len(image)))
    
    for h in range(len(images)):
        for i in range(len(image)):
            mat[h][i] = ssim(image[i],images[h][i])
              
    g = ['clip lim 10','clip_lim 20' ,'clip lim 40']
            
    fig,ax = plt.subplots()
    
    f = np.around(mat,2)
    
    im = ax.imshow(f)

    ax.set_xticks(np.arange(len(image)))
    ax.set_xticklabels(image_names)
    
    ax.set_yticks(np.arange(len(images)))
    ax.set_yticklabels(g)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

    for i in range(len(g)):
        for j in range(len(image_names)):
            text = ax.text(j, i, f[i, j], ha="center", va="center", color="w")
  
    ax.set_title("ssim distance between original and clahe enhanced images")

    
    return mat



def histogram_equalization_CLAHE(images_name, number_bins=256, tile_grid_size=(32, 32), clip_limit=2.0):
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    images  = [ cv2.imread(x, 0)  for x in images_name]
    
    clahe_images = [clahe.apply(x) for x in images]
    #clahe_images  = images
    histograms = [ cv2.calcHist([x], [0], None, [256], [0, 256]) for x in clahe_images]
    
    line =np.arange(0, 256)
    
    plt.figure("histograms with clahe clipLimit= "+str(clip_limit)+" tileGridSize ="+ str(tile_grid_size)+'hist')
    for i in range(0,len(images_name)):
        plt.xlim(0,255)
        plt.ylim(0, 5000)
        plt.plot(line,histograms[i],label=images_name[i])
        plt.title("histograms with clahe clipLimit= "+str(clip_limit)+" tileGridSize ="+ str(tile_grid_size))
        
        plt.show()
    
    plt.figure("histograms with clahe clipLimit= "+str(clip_limit)+" tileGridSize ="+ str(tile_grid_size)+'img')
    
    images_num = int(math.sqrt(len(images_name)))+1
    
    for i in range(0,len(images_name)):
        plt.subplot(images_num,images_num, i+1),plt.imshow(clahe_images[i],'gray')
        plt.show()
   
    
    return clahe_images

def exact_histogram_matching(images_name, ref_img):
        
    images  = [ cv2.imread(x, 0)  for x in images_name]
    
    reference_histogram = ExactHistogramMatcher.get_histogram(cv2.imread(ref_img, 0))
    
    exact_imgs = [ExactHistogramMatcher.match_image_to_histogram(i, reference_histogram) for i in images ]
 
    histograms = [ cv2.calcHist([x.astype('uint8')], [0], None, [256], [0, 256]) for x in exact_imgs]
    
    line =np.arange(0, 256)
    plt.figure('1')
    for i in range(0,len(images_name)):
        plt.xlim(0,255)
        plt.ylim(0, 5000)
        plt.plot(line,histograms[i],label=images_name[i])
        plt.show()
        
    images_num = int(math.sqrt(len(images_name)))+1
    plt.figure('2')
    for i in range(0,len(images_name)):
        plt.subplot(images_num,images_num, i+1),plt.imshow(exact_imgs[i],'gray')
        plt.show()
 
    return exact_imgs

# Histogram Equalization Function
# Reference: https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
def histogram_equalization_2D(img, number_bins=256, display=False):
    # cdf, bins = getCDF(img, display)
    hist, bins = np.histogram(img.flatten(), number_bins, [0,256])
    # cdf: Cumulative Distribution Function
    # numpy.cumsum(): returns the cumulative sum of the elements along a given axis
    cdf = hist.cumsum()
    # Normalize to [0,255], as referenced in https://en.wikipedia.org/wiki/Histogram_equalization
    cdf_normalized = cdf * hist.max()/ cdf.max()

    # The minimum histogram value (excluding 0) by using the Numpy masked array concept
    cdf_m = np.ma.masked_equal(cdf_normalized,0)
    # And apply the histogram equalization equation as given in https://en.wikipedia.org/wiki/Histogram_equalization
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    
    # Look-up table with the information for what is the output pixel value for every input pixel value
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    # Apply the transform
    image_equalized = cdf[img]

    if display:
        # Plot    
        figure2 = plt.figure(2)

        # Original Image
        subplot2 = figure2.add_subplot(1,2,1)
        plt.imshow(img, cmap='gray')
        subplot2.set_title('Original Image')

        # Histogram Equalized Image
        subplot2 = figure2.add_subplot(1,2,2)
        plt.imshow(image_equalized ,cmap='gray')
        subplot2.set_title('Histogram Equalized Image')
        plt.show()

    return image_equalized

 
# Histogram Equalization Function
def histogram_equalization_3D(image, number_bins=256):
    image_equalized = np.zeros(image.shape)
    
    # loop over the slices of the image
    for i in range(image.shape[0]):
        img = image[i, :, :]

        # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
        # get image histogram
        hist, bins = np.histogram(img.flatten(), number_bins)#, [0,256])
        cdf = hist.cumsum() # cumulative distribution function
        cdf = cdf * hist.max()/ cdf.max()#255 * cdf / cdf[-1] # normalize

        # Normalize to [0,255], as referenced in https://en.wikipedia.org/wiki/Histogram_equalization
        cdf_normalized = cdf * hist.max()/ cdf.max()

        # The minimum histogram value (excluding 0) by using the Numpy masked array concept
        cdf_m = np.ma.masked_equal(cdf_normalized,0)
        # And apply the histogram equalization equation as given in https://en.wikipedia.org/wiki/Histogram_equalization
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        
        # Look-up table with the information for what is the output pixel value for every input pixel value
        cdf = np.ma.filled(cdf_m,0).astype('uint8')

        # https://stackoverflow.com/questions/28518684/histogram-equalization-of-grayscale-images-with-numpy/28520445
        # use linear interpolation of cdf to find new pixel values (for 3D images)
        img_eq = np.interp(img.flatten(), bins[:-1], cdf)
        img_eq = img_eq.reshape((image.shape[1], image.shape[2]))

        image_equalized[i, :, :] = img_eq

    return image_equalized


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def merge_slices_into_3D_image(dataset_path, contrast_type):
    dirnames = glob.glob(os.path.join(dataset_path, "*", ""))
        
    for dir in dirnames:
        filenames = glob.glob(os.path.join(dir, "*.tif"))
        first_dimension = second_dimension = third_dimension = 0
        
        # Count the number of slices and get the shape of the image
        # to initialize the dimensions in order to initialize 
        # the following arrays (mask and image representing the 3D images)
        for file in filenames:
            if "_mask" in file:
                third_dimension += 1
            # Execute this only once
            if first_dimension == 0:
                first_dimension = second_dimension = skimage.io.imread(file).shape[0]
       
        mask = np.zeros([first_dimension, second_dimension, third_dimension], dtype=np.uint8)
        image = np.zeros([first_dimension, second_dimension, third_dimension], dtype=np.uint8)

        for file in filenames:
            i = j = 0
            # Avoid already preprocessed images
            if "_mask" in file:
                mask[:,:,i] = skimage.io.imread(file)
                i += 1
            elif contrast_type in file:
                image[:,:,j] = skimage.io.imread(file)
                j += 1


        image_name = file.rsplit(".")[:-1]
        image_name = '.'.join(image_name)
        image_name = image_name + '_' + contrast_type + '-3D.nii'

        # image = nib.Nifti1Image(image, affine=np.eye(4))
        # nib.save(image, image_name)
        imsave(image_name, image)

        mask_name = file.rsplit(".")[:-1]
        mask_name = '.'.join(mask_name)
        mask_name = mask_name + '_' + contrast_type + '-3D_mask.nii'

        # mask = nib.Nifti1Image(mask, affine=np.eye(4))
        # nib.save(mask, mask_name)
        imsave(mask_name, mask)

def imsave(fname, arr):
    sitk_img = sitk.GetImageFromArray(arr, isVector=True)
    sitk.WriteImage(sitk_img, fname)

    # sitk_img = sitk.GetImageFromArray(np.around(arr*255).astype(np.uint8), isVector=True)
    # sitk.WriteImage(sitk_img, fname)

    # plt.imsave(fname, arr, cmap='gray')

    # plt.imsave(fname, np.around(arr*255).astype(np.uint8), cmap='gray')

    # skimage.io.imsave(fname, arr)

    # skimage.io.imsave(fname, arr, plugin='simpleitk')

    # skimage.io.imsave(fname, np.around(arr*255).astype(np.uint8), plugin='simpleitk')

def split_dataset(dataset_path):
    dirnames = glob.glob(os.path.join(dataset_path, "*", ""))
        
    for dir in dirnames:
        filenames = glob.glob(os.path.join(dir, "*.tif"))

        for file in filenames:
            # Avoid already preprocessed images and masks
            if (not ("_pre-contrast"  in file or "_flair" in file or "_post-contrast" in file or "_mask" in file)):
                img = skimage.io.imread(file)

                filename = file.rsplit(".")[:-1]
                filename = '.'.join(filename)
                
                precontrast_img = filename + '_pre-contrast.tif'
                flair_img = filename + '_flair.tif'
                postcontrast_img = filename + '_post-contrast.tif'

                # Avoid creating again file if exists 
                if not os.path.isfile(precontrast_img):
                    skimage.io.imsave(precontrast_img, img[:,:,0])

                if not os.path.isfile(flair_img):
                    skimage.io.imsave(flair_img, img[:,:,1])

                if not os.path.isfile(postcontrast_img):
                    skimage.io.imsave(postcontrast_img, img[:,:,2])

def get_dataset_as_object(dataset_path, contrast_type):
        cases_dict = {}
        dirnames = glob.glob(os.path.join(dataset_path, "*", ""))

        for dir in dirnames:
            filenames = glob.glob(os.path.join(dir, "*.tif"))

            for file in filenames:

                if "_mask" in file:
                    filename = file.rsplit("_")[:-1]
                    filename = '_'.join(filename)
                    filename = filename.rsplit("/")[2:]
                    filename = ''.join(filename)
                    
                    if filename in cases_dict.keys():
                        cases_dict[filename].update({'Mask': file})
                    else:
                        cases_dict[filename] = {'Mask': file}

                elif file.endswith(contrast_type + ".tif"):
                    filename = file.rsplit(".")[:-1]
                    filename = ''.join(filename)
                    filename = file.rsplit("_")[:-1]
                    filename = '_'.join(filename)
                    filename = filename.rsplit("/")[2:]
                    filename = ''.join(filename)

                    if filename in cases_dict.keys():
                        cases_dict[filename].update({'Image': file})
                    else:
                        cases_dict[filename] = {'Image': file}

            if not cases_dict:
                raise FileNotFoundError("Failed to import dataset.")
            
        return cases_dict


def remove_mask_from_image(img, mask):    

    gray_img = rgb2gray(img)
    gray_mask = rgb2gray(mask)

    # blank = np.zeros(img.shape[:2], dtype='uint8')
    # mask = ~cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)

    return cv.bitwise_and(gray_img, gray_mask)
    

def getCDF(img, display=None):
    hist, bins = np.histogram(img.flatten(),256,[0,256])
    # cdf: Cumulative Distribution Function
    # numpy.cumsum(): returns the cumulative sum of the elements along a given axis
    cdf = hist.cumsum()
    # Normalize to [0,255], as referenced in https://en.wikipedia.org/wiki/Histogram_equalization
    cdf_normalized = cdf * hist.max()/ cdf.max()

    if display:
        # Plot
        plt.figure(1)

        # Normalized CDF with red
        plt.plot(cdf_normalized, color = 'r')

        # Histogram with black
        plt.hist(img.flatten(),256,[0,256], color = 'k')
        plt.xlim([0,256])

        # Place labels at the lower right of the plot 
        plt.legend(('Normalized CDF','Histogram'), loc = 'lower right')
        plt.show()

    return cdf, bins


if __name__ == "__main__":
    image_path = 'data/dataset/R01-001.nii'
    mask_path = 'data/dataset/R01-001_roi.nii'
    
    # image = sitk.ReadImage(image_path)
    # image = sitk.GetArrayFromImage(image)
    
    # mask = sitk.ReadImage(mask_path)
    # mask = sitk.GetArrayFromImage(mask)
    image = imageio.imread(image_path)
    
    mask = imageio.imread(mask_path)

    plt.figure(figsize=(20,20))

    plt.subplot(2,2,1)
    plt.imshow(image[12,:,:], cmap="gray")
    plt.title("Brain")

    plt.subplot(2,2,2)
    plt.imshow(mask[12,:,:], cmap="gray")       
    plt.title("Segmentation")

    masked_image = np.zeros(image.shape)
    for i in range(image.shape[0]):
        img = image[i, :, :]
        msk = mask[i, :, :]
        masked_image[i, :, :] = remove_mask_from_image(img, msk)


    plt.subplot(2,2,3)
    plt.imshow(masked_image[12,:,:], cmap='gray')        
    plt.title("Masked Image")

    plt.subplot(2,2,4)
    plt.title('Graylevel Histogram')
    plt.xlabel('Bins')
    plt.ylabel('Number of pixels')
    plt.hist(masked_image[12,:,:].flatten(), 256,[0,256], color = 'b')
    # plt.hist(np.histogram(masked_image.flatten(),256))
    plt.xlim([0,256])

    plt.show()

    cv.waitKey(0)
