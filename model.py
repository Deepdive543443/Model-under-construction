import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#Part of network
class StyleWSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, demodulate = True):
        super().__init__()
        # Parameters for convolution
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel_size
        self.stride = stride
        self.padding = padding
        self.demodulate = demodulate

        # Equalize learning rate or avoid float16 overflow
        self.scale = (1 / (in_channels * kernel_size ** 2)) ** 0.5

        # Learnable noise injection factor
        self.noise_strength = nn.Parameter(torch.zeros([]))

    def forward(self, x, style, noise=None, eps=1e-8):
        '''
        Modulated convolutional layers, perform weight modulation and demodulate before convolution

        :param x: The input features in shape [b, c, h, w]
        :param style: The input style vector in shape [b, in_channels]
        :param noise: The noise injection in shape of [b, 1, h, w]
        :param eps: A small const to avoid nan
        :return: features map in shape of [b, c, h, w]
        '''
        #Normalize weight and style for float 16
        w = self.weight * self.scale / self.weight.norm(float('inf'), dim=[1,2,3], keepdim=True)
        style = style / style.norm(float('inf'), dim=1, keepdim=True)

        #Obtain batch size and size for upcoming reshape task
        batch_size, _, height, width = x.shape

        # Weight modulation and demodulation
        w = w.unsqueeze(0) * style.view(batch_size, 1, -1, 1, 1)
        # Demodulate won't be execute in ToRGB layer
        if self.demodulate:
            demod = torch.sqrt((w**2).sum([2,3,4]) + eps)
            w = w / demod.view(batch_size, -1, 1, 1, 1)
        # A tricky batch weight mod and demod implementation from official code
        x = x.view(1, -1, height, width)
        w = w.view(-1, self.in_channels, self.kernel, self.kernel)
        x = F.conv2d(x * self.scale, weight=w, stride=self.stride, padding=self.padding, groups=batch_size)
        x = x.view(batch_size, -1, height, width) + self.bias

        #Apply noise injection if noise is not none
        if noise is not None:
            x = x + (noise * self.noise_strength)
        return x

class WSLinear(nn.Module):
    def __init__(self, in_channels, out_channels, lr_scaler = 1):
        super().__init__()
        '''
        Equlized Fully connected layer with learning rate scaler
        '''
        self.linear = nn.Linear(in_channels, out_channels)
        # self.scale = (2 / in_channels) ** 0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        self.weight_gain = lr_scaler/ (in_channels ** 0.5)
        self.bias_gain = lr_scaler

        nn.init.normal_(self.linear.weight)
        nn.init.normal_(self.bias)

    def forward(self, x):
        return self.linear(x * self.weight_gain) + (self.bias.view(1, self.bias.shape[0]) * self.bias_gain)

class WSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, lr_scaler = 1):
        super().__init__()
        '''
        Equlized convolutional layer with learning rate scaler
        '''
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bias = self.conv.bias

        self.conv.bias = None

        self.weight_gain = lr_scaler / (in_channels * kernel_size ** 2) ** 0.5
        self.bias_gain = lr_scaler

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.weight_gain) + (self.bias.view(1, self.bias.shape[0], 1, 1) * self.bias_gain)

class PixelNorm(nn.Module):
    def __init__(self,epsilon = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self,x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)



#Generator
class Mapping_network(nn.Module):
    def __init__(self, num_layers = 8, latent_code_dim = 512, w_dim = 512, lr_scale_factor=0.01):
        '''
        Mapping network with 8 fully connected layers

        :param num_layers: num of fully connected layers
        :param latent_code_size: length of latent code
        :param broadcast: time repeat latents space output
        '''
        super().__init__()
        layers = [PixelNorm()]
        for _ in range(num_layers):
            layers.append(WSLinear(latent_code_dim, w_dim, lr_scaler=lr_scale_factor))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping_net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mapping_net(x)
        return x

class ToRGB(nn.Module):
    def __init__(self, in_channels, out_channels = 3, kernel_size=1, stride=1, padding = 0, w_dim=512):
        '''
        Transfer the feature maps to RGB images

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param w_dim:
        '''
        super().__init__()
        self.gain = 1 / (in_channels * kernel_size ** 2) ** 0.5
        self.affline = WSLinear(w_dim, in_channels)
        # Demodulate activation is not executed in ToRGB layers
        self.mod_conv = StyleWSConv(in_channels, out_channels, kernel_size, stride, padding, demodulate=False)

    def forward(self, x, style):
        style = self.affline(style) * self.gain
        x = self.mod_conv(x, style, noise=None)
        return x

class Synthesis_layer(nn.Module):
    def __init__(self, in_channels, out_channels, device, latent_size = 512):
        super().__init__()
        '''
        
        '''
        self.device = device
        self.style_affline = WSLinear(latent_size, in_channels)
        self.mod_conv = StyleWSConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.out_channels = out_channels
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear') if upsample else None

    def forward(self, x, style):
        style = self.style_affline(style)
        # if self.upsample is not None:
        #     x = self.upsample(x)
        x = self.mod_conv(x, style, torch.randn(x.shape[0], self.out_channels, x.shape[2], x.shape[3]).to(self.device))
        return x

class Synthesis_block(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super().__init__()
        self.layers = nn.ModuleList()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.layers.append(Synthesis_layer(in_channels, out_channels, device=device))
        self.layers.append(Synthesis_layer(out_channels, out_channels, device=device))
        self.toRgb = ToRGB(out_channels)
        self.leacy_relu = nn.LeakyReLU(0.2)

    def forward(self, x, style):
        x = self.upsample(x)
        for layer in self.layers:
            x = self.leacy_relu(layer(x, style))
        return x, self.toRgb(x, style)

class StyleGan2_Generator(nn.Module):
    def __init__(self, device, max_channels=512, channels_multiplier = 32768, output_resolution=256, w_dim = 512, style_mixing_shresh=0.9, w_avg_beta = 0.995):
        super().__init__()
        #Device
        self.device = device
      
        # Mapping Network
        self.mapping_networks = Mapping_network(w_dim, w_dim)
        self.w_avg = torch.zeros(w_dim).to(device)
        self.w_avg_beta = w_avg_beta
        self.style_mixing_shresh = style_mixing_shresh

        #Init
        steps = int(math.log2(output_resolution))
        self.in_channels_dict = { 2**res : min(max_channels, int(channels_multiplier / 2**res)) for res in range(2, steps + 1)}
        self.const_input = nn.Parameter(torch.randn(1, self.in_channels_dict[4], 4, 4))
        self.init = Synthesis_layer(self.in_channels_dict[4], self.in_channels_dict[4], device=device)
        # print(self.in_channels_dict)

        self.blocks = nn.ModuleList()
        for step in range(3, steps + 1):
            self.blocks.append(Synthesis_block(self.in_channels_dict[2 ** (step - 1)], self.in_channels_dict[2 ** step], device=device))

        self.rgb_upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def style_mixing(self, z, ws, shresh):
        if torch.rand([]) >= shresh:
            cutoff = torch.randint(512,())
            ws[:, cutoff:] = self.mapping_networks(torch.randn_like(z))[:, cutoff:]
        return ws

    def update_w_avg(self,ws):
        return torch.lerp(torch.mean(ws.detach(), dim=0), self.w_avg, self.w_avg_beta)

    def forward(self, z):
        batch_size = z.shape[0]
        imgs = torch.zeros(batch_size, 3, 4, 4).to(self.device)
        const = self.const_input.repeat(batch_size, 1, 1, 1)

        # Obtain Latent w and update w average
        w = self.style_mixing(z, self.mapping_networks(z), self.style_mixing_shresh)
        self.w_avg = self.update_w_avg(w)

        x = self.init(const, w)
        for block in self.blocks:
            imgs = self.rgb_upsample(imgs)
            x, img = block(x, w)
            imgs += img
        return torch.tanh(imgs), w



#Discriminator
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample='conv'):
        super().__init__()
        if downsample == 'conv':
            self.skip = WSConv(in_channels, out_channels, 1, 2, 0)
        else:
            self.skip = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                WSConv(in_channels, out_channels, 1, 1, 0)
            )


        self.convs = nn.Sequential(WSConv(in_channels, out_channels, 3, 1, 1),
                                   WSConv(out_channels, out_channels, 3, 2, 1))

    def forward(self, x):
        return self.skip(x) + self.convs(x)

class StyleGan2_Discriminator(nn.Module):
    def __init__(self, img_channels = 3, max_channels=512, channels_multiplier = 32768, output_resolution=256):
        super().__init__()
        steps = int(math.log2(output_resolution))
        self.in_channels_dict = {2 ** res: min(max_channels, int(channels_multiplier / 2 ** res)) for res in
                                 range(1, steps + 1)}
        blocks = []
        blocks.append(WSConv(img_channels, self.in_channels_dict[output_resolution], 1, 1, 0)) #from RGB layers
        blocks.append(nn.LeakyReLU(0.2))

        for step in range(3, steps + 1)[::-1]:
            blocks.append(DiscriminatorBlock(self.in_channels_dict[2 ** step], self.in_channels_dict[2 ** (step - 1)], downsample='Avg'))
            blocks.append(nn.LeakyReLU(0.2))
        self.blocks = nn.Sequential(*blocks)

        self.final_conv = nn.Sequential(
            WSConv(max_channels + 1, max_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            WSLinear(int(max_channels * 4 ** 2), max_channels),
            nn.LeakyReLU(0.2),
            WSLinear(max_channels, 1)
        )

    def minibatch_std(self, x, eps = 1e-8):
        std = torch.mean(torch.std(x + eps, dim=0)).repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, std], dim=1)

    def forward(self, x):
        x = self.blocks(x)
        x = self.minibatch_std(x)
        x = self.final_conv(x)
        return self.fc(x)




if __name__ == "__main__":
    z = torch.randn(4,512)
    d = StyleGan2_Discriminator()
    g = StyleGan2_Generator(device='cpu')
    #
    # imgs, ws = g(z)
    # print(imgs.shape, ws.shape)
    # score = d(imgs)
    # print(score, score.shape)
    i = 0
    for k, v in g.state_dict().items():
        print(k, v.shape)
        i+=1
