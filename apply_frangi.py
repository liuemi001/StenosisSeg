from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi
from skimage import exposure
from tqdm import tqdm


def main():
    image_dir = 'C:/Users/emily/Year 2 Classes/ArterySeg/datasets/stenosis/val/images/'
    out_dir = 'C:/Users/emily/Year 2 Classes/ArterySeg/datasets/stenosis/val/filtered_images/'

    # image = image_dir + "1.png"
    # image = Image.open(image)
    # # image = image.convert('L')
    
    # image = np.asarray(image)
    # print("image shape: ", image.shape)
    # plt.imshow(image, interpolation='nearest')
    # plt.show()
    # filtered_image = frangi(image)
    # print(filtered_image)
    # plt.imshow(filtered_image, interpolation='nearest')
    # plt.show()
    # filtered_scaled = (filtered_image * 255).astype(np.uint8)
    # diff = image - filtered_scaled
    # plt.imshow(diff, interpolation='nearest')
    # plt.show()
    # img_adapteq = exposure.equalize_adapthist(image, clip_limit=0.03)
    # plt.imshow(img_adapteq, interpolation='nearest')
    # plt.show()
    # img_eq_frangi = frangi(img_adapteq)
    # plt.imshow(img_eq_frangi, interpolation='nearest')
    # plt.show()
    # img_eq_frangi_diff = img_adapteq - img_eq_frangi
    # plt.imshow(img_eq_frangi_diff, interpolation='nearest')
    # plt.show()
    for image_name in tqdm(os.listdir(image_dir)):
        image_path = image_dir + image_name
        image = Image.open(image_path)
        image = np.asarray(image)
        adapt_eq_img = exposure.equalize_adapthist(image, clip_limit=0.03)
        frangi_adapt = frangi(adapt_eq_img)
        frangi_adapt_diff = adapt_eq_img - frangi_adapt
        # plt.imshow(frangi_adapt_diff, interpolation='nearest')
        # plt.axis('off')
        # plt.show()
        frangi_adapt_diff = (frangi_adapt_diff * 255).astype(np.uint8)
        # print(frangi_adapt_diff)
        # print(frangi_adapt_diff.shape)
        # plt.imshow(frangi_adapt_diff, interpolation='nearest')
        # plt.show()
        processed_img = Image.fromarray(frangi_adapt_diff)
        processed_img.save(out_dir + image_name)




if __name__ == '__main__':
    main()