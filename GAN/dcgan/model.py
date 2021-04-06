import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils import Discriminator, Generator, initalize_weight


device = torch.device("cuda")

lr = 2e-4
batch_size  = 128
img_size    = 64
channels_img= 3
z_dim   =   100
epochs = 100
features_d = 64
features_g = 64

transforms = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(channels_img)],[0.5 for _ in range(channels_img)]),
    ]
)

dataset = datasets.ImageFolder(root="./dataset",transform=transforms)

loader = DataLoader(dataset,batch_size=batch_size, shuffle=True)

gen = Generator(z_dim,channels_img,features_g).to(device)
disc = Discriminator(channels_img,features_d).to(device)

initalize_weight(gen)
initalize_weight(disc)

opt_gen = optim.Adam(gen.parameters(),lr = lr, betas=(0.5,0.999))
opt_disc = optim.Adam(disc.parameters(),lr = lr, betas=(0.5,0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32,z_dim,1,1).to(device)




writer_fake = SummaryWriter(f"runs_/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs_/GAN_MNIST/real")


step = 0

for epoch in range(epochs):
    for batch_idx, (real,_) in enumerate(loader):
        real = real.to(device)
        # real = real.view(-1,784).to(device)
        noise = torch.randn(batch_size,z_dim,1,1).to(device)

        #Train the discriminator : max log(D(real)) + log (1-D(G(z)))
        fake = gen(noise)

        disc_real = disc(real).reshape(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake).reshape(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD =  (lossD_real+lossD_fake)/2

        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        # Train Generator min log(1-D(G(z))) => max log(D(G(z)))
        output = disc(fake).reshape(-1)
        lossG  = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx%100 == 0:
            print(
            f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(loader)} \
            Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)[:32]
                data = real[:32]
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)
                # img_grid_fal = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                "Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                "Real Images", img_grid_real, global_step=step
                )
                step += 1






























