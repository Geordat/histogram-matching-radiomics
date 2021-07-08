from radiomics_modules.feature_extraction import FeatureExtractor
from histogram_macthing_modules.histogram_matching import HistogramMatcher

import skimage.io
import utils
import os


def main():
    
    DATASET_PATH = os.path.join('data', 'dataset', 'sygrisampol_images')

    post_contrast_imgs = glob.glob(os.path.join(DATASET_PATH, "*post-contrast.tif_removed_background.png"))
    
    
    reference_img=  cv2.imread(post_contrast_imgs[0],0)
    target_img  = cv2.imread(post_contrast_imgs[5],0)
    
    
    #exact histogram matching
    
    exact_matching_final = utils.exact_histogram_matching(post_contrast_imgs,post_contrast_imgs[0])
    
    # CLAHE 
    
    init_img = [cv2.imread(x, 0)   for x in post_contrast_imgs]
    clip_lim = [5 ,20 , 40 ]
    images = [[],[],[]]
    p=0
    for i in clip_lim:
        images[p] = utils.histogram_equalization_CLAHE(post_contrast_imgs,tile_grid_size=(24,24), clip_limit=i)
        utils.histograms_compare(images[p],post_contrast_imgs,name=i)
        p=p+1
        
    utils.ssim_compare(init_img,images,post_contrast_imgs)
    
    CLAHE_images_final = images[0]
    

    

if __name__ == '__main__':

    PARAMETERS_PATH = os.path.join('radiomics_modules', 'Params.yaml')

    DATASET_PATH = os.path.join('data', 'dataset')
    FEATURES_OUTPUT_PATH = os.path.join('data', 'pyradiomics_extracted_features.csv')

    NEW_DATASET_OUTPUT_PATH = os.path.join('data', 'new_dataset')
    NEW_FEATURES_OUTPUT_PATH = os.path.join('data', 'new_pyradiomics_extracted_features.csv')

    main()
