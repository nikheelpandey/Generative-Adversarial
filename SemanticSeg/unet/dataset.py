from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset

class CXRDataset(Dataset):
    '''
    CAUTON: Some masks of the images from img_dir are missing. Hence, only processing those images whose masks are available
    '''
    def __init__(self, image_dir, mask_dir,type="train",split_ratio=0.2, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.masks = os.listdir(mask_dir)
        
        #a very standard "meh" way of train-test split
        if type=="train":
            self.masks = self.masks[:int(len(self.masks)*(1-split_ratio))]

        else:
            self.masks = self.masks[int(len(self.masks)*(1-split_ratio)):]


    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        img_path = os.path.join(self.image_dir, self.masks[index].replace("_mask.png", ".png"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask