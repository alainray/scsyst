from torch.utils.data import Dataset
import torch
from os.path import join 
import pandas as pd
from PIL import Image
import numpy as np

TRANSFORMATIONS = [ "main",
                    "rot90", 
                    "rot180", 
                    "rot270", 
                    "flip_h", 
                    "flip_v" , 
                    "flip_b",
                    "swap_01",
                    "swap_12",
                    "swap_02"]



class SCSyst(Dataset):
    def __init__(self, mode="sum", split="train", root_path="../datasets"):
        self.root_path = root_path
        self.data_root = join(root_path,"scsyst")
        self.idx_to_task = TRANSFORMATIONS
        self.task_to_idx = {task: i for i, task in enumerate(TRANSFORMATIONS)}
        self.df = pd.read_csv(join(self.data_root,"data.csv"))
        self.mode = mode

        # Step 1: Create the mapping from the 'id' column
        shape_mapping = {val: i for i, val in enumerate(self.df['main_shape'].astype('category').cat.categories)}
        color_mapping = {val: i for i, val in enumerate(self.df['main_color'].astype('category').cat.categories)}


        # Step 3: Replace the values in the column using the mapping
        self.df['shape_y'] = self.df['main_shape'].map(shape_mapping)
        self.df['color_y'] = self.df['main_color'].map(color_mapping)
        self.n_shapes = self.df['shape_y'].max()
        self.n_colors = self.df['color_y'].max()
        self.x = []
        self.y = []
        self.df = self.df[self.df['set']==split] # filter data for requested split



        # For task in TRANSFORMATION
        for task in TRANSFORMATIONS:
            #create_x
            x_task = self.get_images_for_task(task)
            self.x.append(x_task)
            # remap labels according to original labelling
            self.df[f'{task}_shape_new'] = self.df[f'{task}_shape'].map(shape_mapping)
            self.df[f'{task}_color_new'] = self.df[f'{task}_color'].map(color_mapping)
            #create_y
            if self.mode == "sum":
                y_task  = self.df[f'{task}_shape_new'] +  self.df[f'{task}_color_new']
            else:
                y_task  = self.df[f'{task}_shape_new']*self.df[f'{task}_color_new']
            y_task = torch.tensor(y_task.values).float()
            self.y.append(y_task)
        # Stack all auxiliary tasks in one x and one y
        self.x = torch.stack(self.x).float().permute(1,0,2,3,4)
        self.y = torch.stack(self.y).float().permute(1,0)
        self.colors = torch.tensor(self.df['color_y'].values).long()
        self.shapes = torch.tensor(self.df['shape_y'].values).long()

    def get_images_for_task(self, task="main"):
        x = []
        for image_path in self.df[f'{task}_path']:
            image_path = join(self.root_path, image_path)
            with Image.open(image_path) as img:
                image_array = torch.from_numpy(np.array(img))
                x.append(image_array)
        x = torch.stack(x).float()/255.
        x = x.permute(0,3,1,2) # To format (C, H, W)
        return x
    # Function to load image and convert to numpy array
    def load_image_as_array(image_path):
        with Image.open(image_path) as img:
            return np.array(img)
    def __len__(self):
        return len(self.y)

    def __getitem__(self,idx):
        return self.x[idx,0], self.x[idx,1:], self.y[idx,0], self.y[idx,1:], self.shapes[idx], self.colors[idx]
    
if __name__ == "__main__":
    ds = SCSyst(mode="sum", split="train", root_path="../datasets")

    print(ds.x.shape, ds.y.shape, ds.colors.shape, ds.shapes.shape)

    print(ds[0][0].shape, ds[0][1].shape,ds[0][2].shape,ds[0][3].shape,ds[0][4].shape,ds[0][5].shape)


    from torch.utils.data import DataLoader

    dl = DataLoader(ds, batch_size=32)

    a,b,c,d,e,f = next(iter(dl))

    print(a.shape,b.shape,c.shape,d.shape,e.shape,f.shape)
    print(e)

    print(ds.df[['shape_y','color_y']])

