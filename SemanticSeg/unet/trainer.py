import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from utils import * #I know it's bad


# hyperparams
lr = 1e-4
dev = "cuda"
batch_size = 16
epochs = 10
workers= 4
img_h = 512
img_w = 512
pin_mem= True
load_model = False


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=dev)
        targets = targets.float().unsqueeze(1).to(device=dev)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


train_transform = A.Compose(
        [
            A.Resize(height=img_h, width=img_w),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

val_transforms = A.Compose(
    [
        A.Resize(height=img_h, width=img_w),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

model = UNet(input_channel=3, output_channel=1).to(dev)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

train_loader, val_loader = get_loaders(
    img_dir,
    mask_dir,
    batch_size,
    train_transform,
    val_transforms,
    workers,
    pin_mem,
)

# if LOAD_MODEL:
#     load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


check_accuracy(val_loader, model, device=dev)
scaler = torch.cuda.amp.GradScaler()

for epoch in range(epochs):
    train_fn(train_loader, model, optimizer, loss_fn, scaler)

    # save model
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer":optimizer.state_dict(),
    }
    save_checkpoint(checkpoint)

    # check accuracy
    check_accuracy(val_loader, model, device=dev)

    # print some examples to a folder
    save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=dev)