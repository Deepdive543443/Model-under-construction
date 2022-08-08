from statistics import mean
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

#Dataset
def get_loader(resolution, batch_size, root, num_workers, testing=False):
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    dataset = ImageFolder(root=root, transform=transform)
    if testing >= 0:
        dataset, _ = torch.utils.data.random_split(dataset, [testing, len(dataset) - testing])
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )
    return dataset, loader

#Loss
class pl_reg(nn.Module):
    def __init__(self, batch_shrink, pl_decay, gain, device):
        super().__init__()
        self.batch_shrink = batch_shrink
        self.pl_decay = pl_decay
        self.gain = gain
        self.device = device

        self.pl_mean = torch.zeros([]).to(device)

    def forward(self, gen_imgs, ws):
        batch_size = gen_imgs.shape[0] // self.batch_shrink
        pl_noise = torch.randn_like(gen_imgs[:batch_size]).to(self.device)
        pl_grad = torch.autograd.grad(outputs=pl_noise * gen_imgs[:batch_size].detach(), inputs=ws[:batch_size], create_graph=True, retain_graph=True)[0]

        pl_length = torch.sqrt((pl_grad ** 2).sum(2).mean(1))
        self.pl_mean = self.pl_mean.lerp(pl_length.mean(), self.pl_decay)

        pl_panelty = (pl_length - self.pl_mean) ** 2 #|J*y - a|^2
        return pl_panelty



# Train loop
def train(gen, critic, g_optim, d_optim, g_scaler, d_scaler, dataloader, pl_loss, z_dim, device, g_reg_interval, d_reg_interval, r1_gamma, num_iters):
    # Load data
    loop = tqdm(dataloader, leave=True)

    for idx, (real, _) in enumerate(loop):
        real = real[:,:3,:,:].to(device)
        z = torch.randn(batch_size,z_dim, 1, 1).to(device)
        
        batch_size = real.shape[0]       
        d_loss = 0
        g_loss = 0
        with torch.cuda.amp.autocast():          
            # D Logis
            gen_imgs, ws = gen(z)
            d_fake_score = critic(gen_imgs.detach())
            d_fake_loss = F.softplus(d_fake_score)
            d_real_score = critic(real)
            d_real_loss = F.softplus(-d_real_score)
            # D R1 reg
            if num_iters % d_reg_interval == 0:
                r1_grad = torch.autograd.grad(outputs=torch.sum(d_real_score), inputs = real, create_graph=True, retain_graph=True)[0]
                r1_panalty = torch.sum(r1_grad ** 2, dim=[1,2,3])
                loss_r1 = r1_panalty * (r1_gamma / 2)
                d_loss = torch.mean(d_fake_loss) + (torch.mean(d_real_loss + loss_r1) * d_reg_interval)
            else:
                d_loss = torch.mean(d_fake_loss) + torch.mean(d_real_loss)
        
        d_optim.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(d_optim)
        d_scaler.update()
        
        with torch.cuda.amp.autocast():
            # G Logis
            g_fake_loss = F.softplus(-critic(gen_imgs))
            # G Path length reg
            if num_iters % g_reg_interval == 0:
                pl_reg = pl_loss(gen_imgs, ws)
                g_loss = torch.mean(g_fake_loss) + (torch.mean(pl_reg) * g_reg_interval)
            else:
                g_loss = torch.mean(g_fake_loss)
        
        g_optim.zero_grad()
        g_scaler.scale(g_loss).backward()
        d_scaler.step(g_optim)
        g_scaler.update()
      # Update G ema

    # Save everything