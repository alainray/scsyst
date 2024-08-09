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
from utils import rotate_90, rotate_180, rotate_270, flip_horizontal, flip_vertical, flip_both

# ----------------------------------------------

IN_DISTRIBUTION = True       # Generate the in-distribution test set
OUT_DISTRIBUTION = True      # Generate the out-of-distribution test set

# If ALL_LABELED = True,
# N_COLORS has to be a result of 3 times a power of 2
ALL_LABELED = True

N_COLORS = 12   # Number of different colors to use for the labels
SHAPE_SIZE = 3  # Images will have SHAPE_SIZE x SHAPE_SIZE pixels
ON_PIXELS = -1  # If != -1, only generate images with ON_PIXELS colored pixels

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
    df_data = {'id': [], 'color': [], 'shape': [], 'set': [],'path': []}

    print("Generating datasets, this may take a while...")
    for idx, shape in tqdm(enumerate(shapes)):

        shape_new_dim = shape[np.newaxis, :, :]

        selected_idxs = np.random.choice(range(colors.shape[0]), n_colors_train + n_colors_test, replace = False)
        selected_colors = colors[selected_idxs]
        colored_shapes = color_shapes(shape_new_dim, selected_colors)

        train_shapes = colored_shapes[:n_colors_train]
        test_shapes = colored_shapes[n_colors_train:]

        for i, img in enumerate(train_shapes):
            color, shape = get_label(img, color_list = colors)

            for j in range(n_noise_train):
                noisy_img = add_noise_bg(img, mean = noise_mean, std = noise_std)
                img_path = os.path.join(path, f'train/{idx}_{i}_{j}.png')

                pil_img = Image.fromarray(noisy_img)
                pil_img.save(img_path)

                df_data['id'].append(idx)
                df_data['color'].append(color)
                df_data['shape'].append(shape)
                df_data['set'].append('train')
                df_data['path'].append(img_path)

            if in_distribution:
                # Save n_noise_test other noisy versions of the image in the train folder
                for j in range(n_noise_test):
                    noisy_img = add_noise_bg(img, mean = noise_mean, std = noise_std)
                    img_path = os.path.join(path, f'test_in_dist/{idx}_{i}_{j}.png')

                    pil_img = Image.fromarray(noisy_img)
                    pil_img.save(img_path)

                    df_data['id'].append(idx)
                    df_data['color'].append(color)
                    df_data['shape'].append(shape)
                    df_data['set'].append('test_in_dist')
                    df_data['path'].append(img_path)

        if out_distribution:
            for i, img in enumerate(test_shapes):
                color, shape = get_label(img, color_list = colors)

                for j in range(n_noise_test):
                    noisy_img = add_noise_bg(img, mean = noise_mean, std = noise_std)
                    img_path = os.path.join(path, f'test_out_dist/{idx}_{i}_{j}.png')

                    pil_img = Image.fromarray(noisy_img)
                    pil_img.save(img_path)

                    df_data['id'].append(idx)
                    df_data['color'].append(color)
                    df_data['shape'].append(shape)
                    df_data['set'].append('test_out_dist')
                    df_data['path'].append(img_path)

    df = pd.DataFrame(df_data)
    df.to_csv(os.path.join(path, "data.csv"), index=False)

if __name__ == "__main__":

    generate_datasets(IN_DISTRIBUTION, OUT_DISTRIBUTION)
    print("Dataset generation complete")