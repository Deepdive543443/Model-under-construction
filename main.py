import torch
import torch.optim as optim
import argparse
from torch.utils.data import Dataset, DataLoader
from train import train, get_loader, pl_reg
from model import StyleGan2_Generator, StyleGan2_Discriminator
torch.backends.cudnn.benchmarks = True

def main():
    
    device = args['device']
    resolution = args['resolution']
    z_dim = args['z_dim']

    
    batch_size = args['batch_size']
    g_lr = args['g_learning_rate']
    d_lr = args['d_learning_rate']
    g_lazy_reg = args['g_lazy_reg']
    d_lazy_reg = args['d_lazy_reg']

    root = args['path']
    testing = args['testing']


    
    
    print(args)
    # Models
    gen = StyleGan2_Generator(args['device'], output_resolution=resolution).to(device)
    critic = StyleGan2_Discriminator(output_resolution=resolution).to(device)
    
    #Dataset and loader
    loader, dataset = get_loader(batch_size=batch_size, resolution=resolution, root=args['path'], testing=args['testing'])
    
    #Optimizer
    g_optim = optim.Adam(gen.parameters(), lr=g_lr)
    d_optim = optim.Adam(critic.parameters(), lr=d_lr)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    #Loss
    pl_reg = pl_reg()

    #Iters
    num_iters = 0

    for epoch in range(args['epochs']):
        train(gen=gen, critic=critic, g_optim=g_optim, d_optim=d_optim,
        g_scaler=g_scaler, d_scaler=d_scaler, dataloader=loader, pl_loss=pl_reg, 
        z_dim=z_dim, device=device, g_reg_interval=g_lazy_reg, d_reg_interval=d_lazy_reg, num_iters=num_iters)





if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-p','--path',type=str,  default="", help='path of the image folder')
    ap.add_argument('-s','--save_per_iter',type=int,  default=200, help='Save model and output per iteration')
    ap.add_argument('-sp','--save_path',type=str,  default="/model", help='the path of saving model')
    ap.add_argument('-l','--load_path',type=str,  default="/model", help='the path of loading model')

    # Model's hyper parameters
    ap.add_argument('-b','--batch_size',type=int,  default=8, help='batchsize_of_training')
    ap.add_argument('-r','--resolution',type=int,  default=256, help='resolution_of_training')
    ap.add_argument('-zd','--z_dim',type=int,  default=512, help="noise input's dimensions")
    ap.add_argument('-wd','--w_dim',type=int,  default=512, help="mapping's dimensions")
    
    
    # Training parameters(optimizer, loss)
    ap.add_argument('-t','--testing',type=int,  default=0, help='Performance testing with small amount of data')
    ap.add_argument('-e','--epochs',type=int,  default=300, help='Training epochs')
    ap.add_argument('-glr','--g_learning_rate',type=float,  default=2e-3, help="Generator's learning rate")
    ap.add_argument('-dlr','--d_learning_rate',type=float,  default=2e-3, help="Discriminator's learning rate")
    ap.add_argument('-gl','--g_lazy_reg',type=int,  default=16, help="generator_reg_per_minibatch")
    ap.add_argument('-dl','--d_lazy_reg',type=int,  default=8, help="discriminator_reg_per_minibatch")
    ap.add_argument('-rg','--r1_gamma',type=float,  default=8, help="r1_gamma")



    #Device
    ap.add_argument('-d','--device',type=str,  default="cuda", help="device")

    args = vars(ap.parse_args())
    main()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
