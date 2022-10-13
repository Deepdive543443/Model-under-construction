import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from copy import deepcopy
from utils import Duplicate_checking
import os, shutil

#Dataset
def get_loader(resolution, batch_size, root, num_workers, testing=False):
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    dataset = ImageFolder(root=root, transform=transform)
    if testing > 0:
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
    def __init__(self, batch_shrink, pl_decay, weight, device):
        super().__init__()
        self.batch_shrink = batch_shrink
        self.pl_decay = pl_decay
        self.weight = weight
        self.device = device

        self.pl_mean = torch.zeros([]).to(device)

    def forward(self, gen, z):
        batch_size = z.shape[0] // self.batch_shrink
        gen_imgs, gen_ws = gen(z[:batch_size])
        pl_noise = torch.randn_like(gen_imgs).to(self.device)
        pl_noise = pl_noise / (gen_imgs.shape[2] * gen_imgs.shape[3]) ** 0.5
        pl_grad = torch.autograd.grad(outputs=torch.sum(pl_noise * gen_imgs), inputs=gen_ws, create_graph=True, retain_graph=True)[0]

        pl_length = torch.sqrt((pl_grad ** 2).sum(1, keepdims=True).mean(1))
        # pl_length = torch.sqrt(pl_grad.pow(2).sum(2).mean(1))
        pl_mean = self.pl_mean.lerp(pl_length.mean(), self.pl_decay)
        self.pl_mean = pl_mean.detach()
        pl_panelty = (pl_length - pl_mean) ** 2 #|J*y - a|^2
        return pl_panelty * self.weight


# Train loop
class EMA:
    def __init__(self, model, decay):
        '''
        Yolov5's ema update class without update parameters

        :param model:
        :param decay:
        '''
        self.ema = deepcopy(model.eval())
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        model.eval()
        with torch.no_grad():
            msd = model.state_dict()
            for key, value in self.ema.state_dict().items():
                value *= self.decay
                value += (1 - self.decay) * msd[key].detach()
                # value = torch.lerp(msd[key].detach(), value, self.decay) # This line for some reason doesn't works

    def generate(self, z):
        return self.ema(z)

#Tensorboard Log
class Logger:
    def __init__(self, root = 'tensorboard'):
        #Logger
        self.root = root
        self.global_step = 0
        self.logger = {}

        #Tensorboard
        shutil.rmtree(root)
        os.makedirs(root)
        os.makedirs(os.path.join(root, 'imgs'))
        self.writer = SummaryWriter(root)

    def update(self, key, value):
        try:
            self.logger[key]['step'] += 1
            self.logger[key]['value'] = value
        except:
            self.logger[key] = {'value': value, 'step': 0}

    def step(self):
        self.global_step += 1
        for k, v in self.logger.items():
            self.writer.add_scalar(k, v['value'], global_step=v['step'])

    def save_sample(self, imgs, name):
        # print(name, imgs)
        imgs = make_grid(imgs * 0.5 + 0.5)

        save_image(imgs, fp=self.root+f'/imgs/{name}_{self.global_step}.png')




def train(gen, critic, g_optim, d_optim, g_scaler, d_scaler, dataloader, pl_loss, ema, num_iters, args, logger):#, z_dim, device, g_reg_interval, d_reg_interval, r1_gamma, ):
    # Load data
    loop = tqdm(dataloader, leave=True)
    for idx, (real, _) in enumerate(loop):
        real = real[:,:3,:,:].to(args['device']).requires_grad_(True)
        gen.train()
        critic.train()
        
        batch_size = real.shape[0]
        z = torch.randn(batch_size, args['w_dim']).to(args['device'])

        d_loss = 0
        g_loss = 0

        # with torch.cuda.amp.autocast():
            # D non saturating logistic loss
        gen_imgs, ws = gen(z)

        d_fake_score = critic(gen_imgs.detach())
        d_fake_loss = F.softplus(d_fake_score)
        d_real_score = critic(real)
        d_real_loss = F.softplus(-d_real_score)

        # D R1 reg
        if num_iters % args['d_lazy_reg'] == 0:
            r1_grad = torch.autograd.grad(outputs=torch.sum(d_real_score), inputs = real, create_graph=True, retain_graph=True)[0]
            r1_panalty = torch.sum(r1_grad ** 2, dim=[1,2,3])
            loss_r1 = r1_panalty * (args['r1_gamma'] / 2)
            loss_r1 = torch.mean(d_real_loss + loss_r1) * args['d_lazy_reg']
            d_loss = torch.mean(d_fake_loss) + loss_r1

            #Log
            logger.update('d_loss', d_loss.item())
            logger.update('d_r1_reg', loss_r1.item())
        else:
            d_loss = torch.mean(d_fake_loss) + torch.mean(d_real_loss)
            #Log
            logger.update('d_loss', d_loss.item())

        d_optim.zero_grad()
        # d_scaler.scale(d_loss).backward()
        # d_scaler.step(d_optim)
        # d_scaler.update()

        d_loss.backward()
        d_optim.step()

        # with torch.cuda.amp.autocast():
        # G Logis
        g_fake_loss = F.softplus(-critic(gen_imgs))
        # G Path length reg
        if num_iters % args['g_lazy_reg'] == 0:
            pl_reg = pl_loss(gen, z)
            pl_reg = torch.mean(pl_reg) * args['g_lazy_reg']
            g_loss = torch.mean(g_fake_loss) + pl_reg

            # Log
            logger.update('g_loss', g_loss.item())
            logger.update('g_pl_reg', pl_reg.item())
        else:
            g_loss = torch.mean(g_fake_loss)
            # Log
            logger.update('g_loss', g_loss.item())

        
        g_optim.zero_grad()
        # g_scaler.scale(g_loss).backward()
        # g_scaler.step(g_optim)
        # g_scaler.update()

        g_loss.backward()
        g_optim.step()

        # G_ema
        ema.update(gen)
        # Print Log
        logger.step()
        if idx % args['save_per_iter'] == 0:
            logger.save_sample(real, name='real')
            logger.save_sample(gen_imgs, name='gen')
            logger.save_sample(ema.generate(torch.randn(16,512).to(args['device']))[0], name='ema')
            loop.set_postfix(log=f"Dubs> img_dub:{Duplicate_checking(real.detach())} gen_imgs_dub:{Duplicate_checking(gen_imgs.detach())}  score fake: {Duplicate_checking(d_fake_score.detach())} score real: {Duplicate_checking(d_real_score.detach())}")


if __name__ == "__main__":
    from model import StyleGan2_Generator
    z = torch.randn(4,512).to('cuda')
    g = StyleGan2_Generator(device='cuda').to('cuda')
    g.train()
    imgs, ws = g(z)
    # print(Duplicate_checking(torch.randn(5,3,256,256)))