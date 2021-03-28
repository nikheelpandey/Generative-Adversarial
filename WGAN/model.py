import os, sys
sys.path.insert(0, os.path.abspath('..'))
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils import Critic, Generator, initalize_weight


device = torch.device("cuda")

lr = 5e-5
batch_size  = 64
img_size    = 64
channels_img= 3
z_dim   =   100
epochs = 5
features_d = 64
features_g = 64

critic_iter = 5

weight_clipping = 0.01



transforms = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(channels_img)],[0.5 for _ in range(channels_img)]),
    ]
)

dataset = datasets.ImageFolder(root="../../dataset",transform=transforms)

loader = DataLoader(dataset,batch_size=batch_size, shuffle=True)

gen = Generator(z_dim,channels_img,features_g).to(device)
disc = Critic(channels_img,features_d).to(device)

initalize_weight(gen)
initalize_weight(disc)

opt_gen = optim.RMSprop(gen.parameters(),lr = lr)
opt_disc = optim.RMSprop(disc.parameters(),lr = lr)

fixed_noise = torch.randn(32,z_dim,1,1).to(device)

writer_fake = SummaryWriter(f"runs_/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs_/GAN_MNIST/real")


step = 0
gen.train()
disc.train()


for epoch in range(epochs):
    for batch_idx, (real,_) in enumerate(loader):
        real = real.to(device)
        current_batch_size = real.shape[0]

        for _ in range(critic_iter):
            noise = torch.randn(current_batch_size,z_dim,1,1).to(device)
            fake = gen(noise)
            disc_real = disc(real).reshape(-1)
            disc_fake = disc(fake).reshape(-1)
            loss_critic = -(torch.mean(disc_real)-torch.mean(disc_fake))
            disc.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_disc.step()


            for p in disc.parameters():
                p.data.clamp_(-weight_clipping,weight_clipping)

        
        output = disc(fake).reshape(-1)
        lossG  = -torch.mean(output)
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()


        if batch_idx%5 == 0:
            gen.eval()
            disc.eval()
            print(
            f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(loader)} \
            Loss D: {loss_critic:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(noise)[:32]
                data = real[:32]
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                "Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                "Real Images", img_grid_real, global_step=step
                )
                step += 1
