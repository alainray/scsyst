
from itertools import permutations
from itertools import product
import torch
import numpy as np
import random
import math
import seaborn as sns
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, TensorDataset, random_split
from scsyst import SCSyst

def sample_colors_from_span(splits):
    span_lenght = 256
    pixel_values = np.arange(span_lenght)
    idx = np.round(np.linspace(0, len(pixel_values) - 1, splits)).astype(int)
    sampled_values = pixel_values[idx].tolist()
    return np.array(list(product(sampled_values, repeat=3)))

def display_colors(color_array, patches_per_row=10, patch_size=1):
    """
    Display a list of colors using matplotlib in multiple rows and larger patches.

    Parameters:
    color_array (numpy.ndarray): A 2D NumPy array with shape (n, 3), where each row represents an RGB color.
    patches_per_row (int): The number of color patches per row.
    patch_size (int or float): The size of each color patch.
    """
    num_colors = color_array.shape[0]
    num_rows = math.ceil(num_colors / patches_per_row)

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(patches_per_row * patch_size, num_rows * patch_size))

    # Create a patch for each color
    for idx, color in enumerate(color_array):
        # Normalize the RGB values to the range [0, 1]
        normalized_color = color / 255.0
        row = idx // patches_per_row
        col = idx % patches_per_row
        ax.add_patch(
            plt.Rectangle(
                (col * patch_size, row * patch_size), patch_size, patch_size, color=normalized_color))

    # Set the limits and hide the axes
    ax.set_xlim(0, patches_per_row * patch_size)
    ax.set_ylim(0, num_rows * patch_size)
    ax.invert_yaxis()
    ax.axis('off')

    plt.show()

def rgb_to_hex(color):
    return '#{:02x}{:02x}{:02x}'.format(*color)

def n_bit_permutations_as_arrays(n):
    # Calculate the maximum number (2^n - 1)
    max_num = (1 << n) - 1  # This is equivalent to 2^n - 1

    # Generate all numbers from 0 to max_num
    numbers = range(max_num + 1)

    # Initialize an empty list to store permutations as arrays
    permutations_arrays = []

    # Generate permutations of the binary representation of each number
    for num in numbers:
        # Convert the number to its n-bit binary representation
        binary_str = '{0:0{n}b}'.format(num, n=n)

        # Convert binary string to numpy array of integers
        perm_array = np.array([int(bit) for bit in binary_str])

        # Append the numpy array to the list
        permutations_arrays.append(torch.from_numpy(perm_array).float())

    return permutations_arrays

def sample_random_n_bit_number(n):
    # Generate a random n-bit number
    number = random.getrandbits(n)

    # Convert the number to its n-bit binary representation
    binary_str = f'{number:0{n}b}'

    # Convert binary string to a PyTorch tensor of floats
    perm_array = torch.tensor([float(bit) for bit in binary_str])

    return perm_array

def sample_unique_n_bit_numbers(n, N):
    if N > 2**n:
        raise ValueError("N cannot be greater than the total number of unique n-bit numbers")

    unique_numbers = set()
    unique_tensors = []

    while len(unique_numbers) < N:
        perm_array = sample_random_n_bit_number(n)
        number_str = ''.join(map(str, map(int, perm_array.tolist())))
        if number_str not in unique_numbers:
            unique_numbers.add(number_str)
            unique_tensors.append(perm_array)

    return unique_tensors


def create_dataset(args, shapes, colors):
    H = args.dataset_parameters['height']
    W = args.dataset_parameters['width']
    n_shapes = args.dataset_parameters['n_shapes']
    n_colors = args.dataset_parameters['n_colors']
    shape_labels = [i for i, data in enumerate(shapes)]
    shape_labels*=n_colors
    shape_labels = torch.tensor(shape_labels).long()
    final_dataset = []
    colors = colors.view(-1,1,3)
    color_labels = []
    for i,c in enumerate(colors):
        color_labels.extend([i]*n_shapes)
    color_labels = torch.tensor(color_labels).long()
    for c in colors:
        r = shapes @ c             # Create colored shape!
        final_dataset.append(r)
    # Generate random indices
    final_dataset = torch.stack(final_dataset).view(-1,H,W,3).permute(0,3,1,2)
    # Results is shape, ([n_shapes * n_colors] x 3 color channels x H x W)
    return final_dataset, shape_labels, color_labels

def plot_dataset( dataset, colors):
    n_colors = colors.shape[0]
    grid_img = make_grid(dataset, nrow=n_colors, padding=1)
    # Convert grid image to numpy array for display
    grid_img_np = TF.to_pil_image(grid_img)
    grid_img_np.save(f"dataset.png")
    grid_img_np = np.array(grid_img_np)

    # Display the grid image
    plt.figure(figsize=(16, 16))
    plt.imshow(grid_img_np)
    plt.axis('off')
    plt.title('Grid of Images')
    plt.show()

def create_dataloaders(args):
    if args.dataset == "scsyst":
        # 
        SCSyst(split="train")
        # Create DataLoaders
        train_dataset = SCSyst(split="train")
        test_in_dist_dataset = SCSyst(split="test_in_dist")
        test_out_dist_dataset = SCSyst(split="test_out_dist")
        train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=False)
        test_in_dist_loader = DataLoader(test_in_dist_dataset, batch_size=10000, shuffle=False)
        test_out_dist_loader = DataLoader(test_out_dist_dataset, batch_size=10000, shuffle=False)

    '''splits = args.dataset_parameters['color_splits']
    colors = sample_colors_from_span(splits=splits)
    systematic_colors = sample_colors_from_span(splits=2)
    all([(a == colors).all(-1).any() for a in systematic_colors])
    colors = torch.from_numpy(colors/255).float()[1:] # drop black color
    H = args.dataset_parameters['height'] # Height
    W = args.dataset_parameters['width'] # Width
    n_shapes = args.dataset_parameters['n_shapes'] # Number of shapes to use in experiments
    n_colors = args.dataset_parameters['n_colors'] # Number of colors to use in experiments
    
    n = H*W  # Replace with the desired number of bits
    permutations_result = sample_unique_n_bit_numbers(n, n_shapes)
    shapes = torch.stack(permutations_result).unsqueeze(-1) # all shapes possible with H x W blocks
    
    shape_labels = [i for i, data in enumerate(shapes)]
    cs = colors.view(-1,1,3)
    indices = random.sample(range(len(colors)), n_colors) # Generate random indices
    cs = cs[indices]
    # 3, 1, 2 x 1, 2, 4
    
    final_dataset, shape_y, color_y = create_dataset(args, shapes, cs)
    denominator = ((n_shapes) + (n_colors))
    denominator = 1
    y = ((shape_y + 1) + (color_y + 1))/ denominator
    y = y.float().view(-1,1)

    # TODO: Temporary code for creating auxiliary xs and ys
    n_samples = y.shape[0]
    aux_x = torch.randn((n_samples, args.n_tasks-1, 3, H, W)).float()
    aux_y = torch.randn((n_samples, args.n_tasks-1, 1)).float()
    print(final_dataset.shape, aux_x.shape, y.shape, aux_y.shape)
    dataset = TensorDataset(final_dataset, aux_x, y, aux_y)# shape_y, color_y)
    # plot_dataset(final_dataset, colors)
    
    # Define the split ratio
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    # Randomly split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])'''


    return {'train': train_loader, 'test_in_dist': test_in_dist_loader, 'test_out_dist': test_out_dist_loader}