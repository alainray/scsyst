import os
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm

from utils import get_shapes, get_color_list, color_shapes, assert_dataset_path

# ----------------------------------------------

# Noise function used for the in-distribution dataset
from utils import add_noise_bg, get_label

# Image transformations used for the out-of-distribution dataset
from utils import rotate_90, rotate_180, rotate_270, flip_horizontal, flip_vertical, flip_both, swap_channels, change_color

# ----------------------------------------------

IN_DISTRIBUTION = True       # Generate the in-distribution test set
OUT_DISTRIBUTION = True      # Generate the out-of-distribution test set

# If ALL_LABELED = True,
# N_COLORS has to be a result of 3 times a power of 2
ALL_LABELED = True

N_COLORS = 12   # Number of different colors to use for the labels
SHAPE_SIZE = 3  # Images will have SHAPE_SIZE x SHAPE_SIZE pixels
ON_PIXELS = 7  # If != -1, only generate images with ON_PIXELS colored pixels

# ----------------------------------------------

# change_color transformation not used in this script
TRANSFORMATIONS = {"rot90": rotate_90, 
                    "rot180": rotate_180, 
                    "rot270": rotate_270, 
                    "flip_h": flip_horizontal, 
                    "flip_v": flip_vertical, 
                    "flip_b": flip_both,
                    "swap_01": lambda img: swap_channels(img, 0, 1),
                    "swap_12": lambda img: swap_channels(img, 1, 2),
                    "swap_02": lambda img: swap_channels(img, 0, 2)}

# ----------------------------------------------

def generate_datasets(in_distribution = True, out_distribution = True, path = "dataset",
                      n_colors_train = 2, n_colors_test = 3, n_noise_train = 3, n_noise_test = 4,
                      noise_mean = 0, noise_std = 20, seed = 42):

    # The number of different colors for the same shape in train/in-distribution test is given by n_colors_train
    # The number of different colors for the same shape in test is given by n_colors_out

    # The number of different noisy images for the same shape in train is given by n_noise_train
    # The number of different noisy images for the same shape in test is given by n_noise_test

    np.random.seed(seed)

    # Ensure the path exists
    assert_dataset_path(path)

    # Generate all possible shapes (0 or 1) of size SHAPE_SIZE x SHAPE_SIZE
    shapes = get_shapes(SHAPE_SIZE, ON_PIXELS)
    colors = get_color_list(N_COLORS, ALL_LABELED)

    # Create a dataframe to store the data
    df_data = {'main_color': [], 'main_shape': [], 'set': [],'main_path': []}

    for transformation in TRANSFORMATIONS.keys():
        df_data[transformation + "_path"] = []
        df_data[transformation + "_shape"] = []
        df_data[transformation + "_color"] = []

    print("Generating all possible shapes and colors...")
    for idx, shape in tqdm(enumerate(shapes)):
        shape_new_dim = shape[np.newaxis, :, :]
        colored_shapes = color_shapes(shape_new_dim, colors)

        for i, img in enumerate(colored_shapes):
            color_id, shape_id = get_label(img, color_list = colors)

            pil_img = Image.fromarray(img)
            img_path = os.path.join(path, f'full/{shape_id}_{color_id}.png')
            pil_img.save(img_path)

    print("Generating datasets, this may take a while...")
    for idx, shape in tqdm(enumerate(shapes)):

        shape_new_dim = shape[np.newaxis, :, :]

        selected_idxs = np.random.choice(range(colors.shape[0]), n_colors_train + n_colors_test, replace = False)
        selected_colors = colors[selected_idxs]

        train_shapes = color_shapes(shape_new_dim, selected_colors[:n_colors_train])
        test_shapes = color_shapes(shape_new_dim, selected_colors[n_colors_train:])

        for i, img in enumerate(train_shapes):
            color_id, shape_id = get_label(img, color_list = colors)

            for j in range(n_noise_train):
                noisy_img = add_noise_bg(img, mean = noise_mean, std = noise_std)
                img_path = os.path.join(path, f'train/{idx}_{i}_{j}.png')

                pil_img = Image.fromarray(noisy_img)
                pil_img.save(img_path)

                df_data['main_color'].append(color_id)
                df_data['main_shape'].append(shape_id)
                df_data['set'].append('train')
                df_data['main_path'].append(img_path)

                for transformation in TRANSFORMATIONS.keys():
                    transformation_img = img.copy()
                    img_out = TRANSFORMATIONS[transformation](transformation_img)
                    transformation_color, transformation_shape = get_label(img_out, color_list = colors)
                    df_data[transformation + "_path"].append(os.path.join(path, f'full/{transformation_shape}_{transformation_color}.png'))
                    df_data[transformation + "_shape"].append(transformation_shape)
                    df_data[transformation + "_color"].append(transformation_color)

            if in_distribution:
                # Save n_noise_test other noisy versions of the image in the train folder
                for j in range(n_noise_test):
                    noisy_img = add_noise_bg(img, mean = noise_mean, std = noise_std)
                    img_path = os.path.join(path, f'test_in_dist/{idx}_{i}_{j}.png')

                    pil_img = Image.fromarray(noisy_img)
                    pil_img.save(img_path)

                    df_data['main_color'].append(color_id)
                    df_data['main_shape'].append(shape_id)
                    df_data['set'].append('test_in_dist')
                    df_data['main_path'].append(img_path)

                    for transformation in TRANSFORMATIONS.keys():
                        transformation_img = img.copy()
                        img_out = TRANSFORMATIONS[transformation](transformation_img)
                        transformation_color, transformation_shape = get_label(img_out, color_list = colors)
                        df_data[transformation + "_path"].append(os.path.join(path, f'full/{transformation_shape}_{transformation_color}.png'))
                        df_data[transformation + "_shape"].append(transformation_shape)
                        df_data[transformation + "_color"].append(transformation_color)

        if out_distribution:
            for i, img in enumerate(test_shapes):
                color_id, shape_id = get_label(img, color_list = colors)

                for j in range(n_noise_test):
                    noisy_img = add_noise_bg(img, mean = noise_mean, std = noise_std)
                    img_path = os.path.join(path, f'test_out_dist/{idx}_{i}_{j}.png')

                    pil_img = Image.fromarray(noisy_img)
                    pil_img.save(img_path)

                    df_data['main_color'].append(color_id)
                    df_data['main_shape'].append(shape_id)
                    df_data['set'].append('test_out_dist')
                    df_data['main_path'].append(img_path)

                    for transformation in TRANSFORMATIONS.keys():
                        transformation_img = img.copy()
                        img_out = TRANSFORMATIONS[transformation](transformation_img)
                        transformation_color, transformation_shape = get_label(img_out, color_list = colors)
                        df_data[transformation + "_path"].append(os.path.join(path, f'full/{transformation_shape}_{transformation_color}.png'))
                        df_data[transformation + "_shape"].append(transformation_shape)
                        df_data[transformation + "_color"].append(transformation_color)


    df = pd.DataFrame(df_data)
    df.to_csv(os.path.join(path, "data.csv"), index=False)

if __name__ == "__main__":

    generate_datasets(IN_DISTRIBUTION, OUT_DISTRIBUTION, path="scsyst",noise_std=10, n_noise_train = 20)
    print("Dataset generation complete")