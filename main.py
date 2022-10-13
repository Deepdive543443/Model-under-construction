import torch
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader
from train import train, get_loader, pl_reg, EMA, Logger
from model import StyleGan2_Generator, StyleGan2_Discriminator
torch.backends.cudnn.benchmarks = True

def main():
    # Parameters scaling
    c_g = args['g_lazy_reg'] / (args['g_lazy_reg'] + 1)
    c_d = args['d_lazy_reg'] / (args['d_lazy_reg'] + 1)

    args['g_learning_rate'] *= c_g
    args['d_learning_rate'] *= c_d

    args['ema_beta'] = 0.5 ** (args['batch_size'] / args['ema_kimg'] * 1e3)

    args['g_beta_1'] = args['g_beta_1'] ** c_g
    args['g_beta_2'] = args['g_beta_2'] ** c_g
    args['d_beta_1'] = args['d_beta_1'] ** c_d
    args['d_beta_1'] = args['d_beta_1'] ** c_d

    print(args)
    # Models
    print('Setting up model......')
    gen = StyleGan2_Generator(device=args['device'], output_resolution=args['resolution']).to(args['device'])
    critic = StyleGan2_Discriminator(output_resolution=args['resolution']).to(args['device'])

    # Optimizer
    print('Setting up optimizer......')
    g_optim = optim.Adam(gen.parameters(), lr=args['g_learning_rate'])
    d_optim = optim.Adam(critic.parameters(), lr=args['d_learning_rate'])
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    #Dataset and loader
    print('Loading data......')
    dataset, loader = get_loader(batch_size=args['batch_size'], resolution=args['resolution'], root=args['path'], testing=args['testing'], num_workers=args['num_worker'])

    #Loss
    pl_loss = pl_reg(args['batch_shrink'], args['pl_decay'], args['pl_lambda'], args['device'])

    #Iters
    num_iters = 0

    #EMA
    ema = EMA(model=gen, decay=args['ema_beta'])

    #Logger
    logger = Logger()

    # Training loop
    for epoch in range(args['epochs']):
        train(gen=gen, critic=critic, g_optim=g_optim, d_optim=d_optim, g_scaler=g_scaler,
              d_scaler=d_scaler, dataloader=loader, pl_loss=pl_loss, num_iters=num_iters, ema=ema,
              logger=logger, args=args)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-p','--path',type=str,  default="E:/exercise/ProgressiveGAN/Anime256", help='path of the image folder')
    ap.add_argument('-w', '--num_worker', type=int, default=0, help='num_worker for dataloader')
    ap.add_argument('-s','--save_per_iter',type=int,  default=20, help='Save model and output per iteration')
    ap.add_argument('-sp','--save_path',type=str,  default="model", help='the path of saving model')
    ap.add_argument('-l','--load_path',type=str,  default="model", help='the path of loading model')

    # Model's hyper parameters
    ap.add_argument('-b','--batch_size',type=int,  default=5, help='batchsize_of_training')
    ap.add_argument('-r','--resolution',type=int,  default=256, help='resolution_of_training')
    ap.add_argument('-zd','--z_dim',type=int,  default=512, help="noise input's dimensions")
    ap.add_argument('-wd','--w_dim',type=int,  default=512, help="mapping's dimensions")
    
    
    # Training parameters(optimizer, loss, regularization)
    ap.add_argument('-t','--testing',type=int,  default=0, help='Performance testing with small amount of data')
    ap.add_argument('-e','--epochs',type=int,  default=300, help='Training epochs')

    ap.add_argument('-glr','--g_learning_rate',type=float,  default=2e-3, help="Generator's learning rate")
    ap.add_argument('-gb1', '--g_beta_1', type=float, default=0.0, help="Generator optimizer's beta 1")
    ap.add_argument('-gb2', '--g_beta_2', type=float, default=0.99, help="Generator optimizer's beta 2")

    ap.add_argument('-dlr','--d_learning_rate',type=float,  default=2e-3, help="Discriminator's learning rate")
    ap.add_argument('-db1', '--d_beta_1', type=float, default=0.0, help="Discriminator optimizer's beta 1")
    ap.add_argument('-db2', '--d_beta_2', type=float, default=0.99, help="Discriminator optimizer's beta 2")

    ap.add_argument('-gl','--g_lazy_reg',type=int,  default=16, help="generator_reg_per_minibatch")
    ap.add_argument('-dl','--d_lazy_reg',type=int,  default=4, help="discriminator_reg_per_minibatch")
    ap.add_argument('-rg','--r1_gamma',type=float,  default=8.0, help="r1_gamma")

    #Path length regularization
    ap.add_argument('-pd', '--pl_decay', type=float, default=0.01, help="path_length_decay")
    ap.add_argument('-pg', '--pl_lambda', type=float, default=2.0, help="scale factor of path length regularization")
    ap.add_argument('-bs', '--batch_shrink', type=int, default=2, help="batch_shrink for path length regularization")

    #G EMA
    ap.add_argument('-ek', '--ema_kimg', type=int, default=20, help="ema per kimg?")

    #Device
    ap.add_argument('-d','--device',type=str,  default="cuda", help="device")

    args = vars(ap.parse_args())
    main()
