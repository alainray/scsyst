import os
import itertools
import numpy as np

def rotate_90(img):
    return np.rot90(img, k=1, axes=(0, 1))

def rotate_180(img):
    return np.rot90(img, k=2, axes=(0, 1))

def rotate_270(img):
    return np.rot90(img, k=3, axes=(0, 1))

def flip_horizontal(img):
    return np.fliplr(img)

def flip_vertical(img):
    return np.flipud(img)

def flip_both(img):
    return np.flipud(np.fliplr(img))

def swap_channels(img, ch_1, ch_2):
    img[..., [ch_1, ch_2]] = img[..., [ch_2, ch_1]]
    return img

def change_color(img, color_list, color_idx):
    # Replace all colored pixels with the new color
    shape = id_to_shape(get_shape(img))
    return color_list[color_idx] * shape[:, :, np.newaxis]

# Get the color ID of an image
def get_color(img, color_list = []):
    # Check if the image contains any of the colors and return the index
    for i, color in enumerate(color_list):
        if np.any(np.all(img == color, axis=-1)):
            return i
    return -1

# Get the shape ID of an image
def get_shape(img):
    # Replace all non zero values with 1
    binary_img = np.where(np.any(img > 0, axis=-1), 1, 0)

    # Flatten the image into a 1D array
    binary_img = binary_img.flatten()

    # Convert the binary array to an integer
    shape = int(''.join(binary_img.astype(str)), 2)

    return shape

# Get both the color and shape of an image
def get_label(img, color_list = []):
    color = get_color(img, color_list)
    shape = get_shape(img)
    return color, shape

# Get the shape associated with an id
def id_to_shape(id, shape_size = 3):
    binary_arr = np.array(list(bin(id)[2:].zfill(shape_size**2))).astype(np.uint8)
    return binary_arr.reshape(shape_size, shape_size)

def get_color_list(n_colors, all_labeled = True):
    # Create a list of colors
    colors = []

    # Calculate the number of steps needed to get at least n_colors
    num_steps = 0
    while True:
        if 3 * num_steps**2 >= n_colors:
            break
        num_steps += 1

    # Get all 2-element combinations with num_steps elements
    step_size = 255 // num_steps
    steps = [i * step_size for i in range(num_steps)]
    combinations = list(itertools.product(steps, repeat = 2))

    # For each combination, create a color and add it to the list
    for combination in combinations:
        colors.append([255, combination[0], combination[1]])        
        colors.append([combination[0], 255, combination[1]])
        colors.append([combination[0], combination[1], 255])

    # Assert that the number of colors is correct if ALL_LABELED is True
    if all_labeled:
        if not len(colors) == n_colors:
            raise ValueError(f"Number of colors is not correct: {n_colors} has to be a result of 3 * m^2, with m an integer value")

    return np.array(colors[:n_colors]).astype(int)

# Generate all possible shapes (binary) of size shape_size x shape_size
def get_shapes(shape_size, on_pixels = -1):
    lst = [list(i) for i in itertools.product([0, 1], repeat = shape_size**2)]
    shapes = np.array(lst).reshape(-1, shape_size, shape_size)

    if on_pixels != -1:
        shapes = shapes[shapes.sum(axis = (1, 2)) == on_pixels]

    # Get shapes with at least one pixel on (not the empty shape)
    on_shapes = shapes[shapes.sum(axis = (1, 2)) > 0]

    return on_shapes

# Color the shapes with the colors, input must have dimensions N_SHAPES x SHAPE_SIZE x SHAPE_SIZE
def color_shapes(shapes, colors):
    # Create a three channel version of the shapes
    img_shapes = np.repeat(shapes[:, :, :, np.newaxis], 3, axis=-1)

    n_shapes = img_shapes.shape[0]
    shape_size = img_shapes.shape[1]
    n_colors = colors.shape[0]

    # Create a 3D array of size N_COLORS * N_SHAPES x SHAPE_SIZE x SHAPE_SIZE x 3
    # where the second and third dimensions are the shape and the last dimension is the RGB color
    data = np.zeros((n_colors * n_shapes, shape_size, shape_size, 3)).astype(int)

    # Fill the data array with the colors and shapes
    for i in range(n_colors):
        for j in range(n_shapes):
            data[i * n_shapes + j] = np.tile(colors[i], (shape_size, shape_size, 1)) * img_shapes[j]

    return data.astype(np.uint8)

# Add gaussian noise to the image
def add_noise_bg(img, mean=0, std=20):
    noise = np.random.normal(mean, std, img.shape)

    # Get pixels which are of value (0, 0, 0)
    mask = np.all(img != [0, 0, 0], axis=-1)

    # Add noise to the background pixels
    noise[mask] = 0
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

# Assert that the dataset path exists, if not create it
def assert_dataset_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Path {path} does not exist but has been created")

    if not os.path.exists(os.path.join(path, "full")):
        os.makedirs(os.path.join(path, "full"))

    if not os.path.exists(os.path.join(path, "train")):
        os.makedirs(os.path.join(path, "train"))

    if not os.path.exists(os.path.join(path, "test_in_dist")):
        os.makedirs(os.path.join(path, "test_in_dist"))

    if not os.path.exists(os.path.join(path, "test_out_dist")):
        os.makedirs(os.path.join(path, "test_out_dist"))