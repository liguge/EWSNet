import torch
import torch.nn as nn
from math import pi
import torch.nn.functional as F
# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True


'''
Laplace小波卷积核初始化
'''
def Laplace(p):

    w = 2 * pi * 80
    q = torch.tensor(1 - pow(0.03, 2))
    return 0.08 * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1)))) * (torch.sin(w * (p - 0.1)))

class Laplace_fast(nn.Module):
    def __init__(self, out_channels, kernel_size, stride=2):
        super(Laplace_fast, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.a_ = nn.Parameter(torch.linspace(1, 100, out_channels).view(-1, 1))
        self.b_ = nn.Parameter(torch.linspace(0, 100, out_channels).view(-1, 1))

    def forward(self, waveforms):
        time_disc = torch.linspace(0, 1, steps=int(self.kernel_size))
        p1 = (time_disc.cuda() - self.b_.cuda()) / (self.a_.cuda())
        laplace_filter = Laplace(p1)
        filters = laplace_filter.view(self.out_channels, 1, self.kernel_size).cuda()
        return F.conv1d(waveforms, filters, stride=self.stride, padding=0)


class Laplace_fastv2:

    def __init__(self, out_channels, kernel_size, eps=-0.3):
        super(Laplace_fastv2, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.a_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)    #out_channels-1     通道是整数
        self.b_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.time_disc = torch.linspace(0, self.kernel_size - 1, steps=int(self.kernel_size))

    def forward(self):
        p1 = (self.time_disc - self.b_) / (self.a_ - self.eps)
        filter = Laplace(p1).view(self.out_channels, 1, self.kernel_size)  # (70,1,85)
        return filter

class Laplace_fastv21(nn.Module):

    def __init__(self, out_channels, kernel_size, eps=0.3):
        super(Laplace_fastv21, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.a_ = torch.linspace(0, self.out_channels, self.out_channels).view(-1, 1)
        self.b_ = torch.linspace(0, self.out_channels, self.out_channels).view(-1, 1)
        self.time_disc = torch.linspace(0, self.kernel_size-1, steps=int(self.kernel_size))

    def forward(self):
        p1 = (self.time_disc - self.b_) / (self.a_ + self.eps)
        filter = Laplace(p1).view(self.out_channels, 1, self.kernel_size)  # (70,1,85)
        return filter
# def Laplace1(p):
#     # m = 1000
#     # ep = 0.03
#     # # tal = 0.1
#     # f = 80
#     w = 2 * pi * 80
#     # A = 0.08
#     q = torch.tensor(1 - pow(0.03, 2))
#     #return (0.08 * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1)))) * (torch.sin(w * (p - 0.1))))
#     # y = 0.08 * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1))).sigmoid()) * (torch.sin(w * (p - 0.1)))  # 99.82%
#     return 0.08 * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1)))) * (torch.sin(w * (p - 0.1)))

class Laplace_fastv3(nn.Module):

    def __init__(self, out_channels, kernel_size):
        super(Laplace_fastv3, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.a_ = torch.linspace(1, out_channels, out_channels).view(-1, 1)
        self.b_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.time_disc = torch.linspace(0, 1, steps=int(self.kernel_size))

    def forward(self):
        p1 = (self.time_disc - self.b_) / self.a_
        filter = Laplace(p1).view(self.out_channels, 1, self.kernel_size)  # (70,1,85)
        return filter

# class Laplace_fastv31(nn.Module):
#
#     def __init__(self, out_channels, kernel_size, eps=0.1):
#         super(Laplace_fastv31, self).__init__()
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.eps = eps
#         self.a_ = torch.linspace(1, out_channels, out_channels).view(-1, 1)
#         self.b_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
#         self.time_disc = torch.linspace(0, 1, steps=int(self.kernel_size))
#
#     def forward(self):
#         p1 = (self.time_disc - self.b_) / (self.a_)
#         filter = Laplace(p1).view(self.out_channels, 1, self.kernel_size)  # (70,1,85)
#         return filter
'''
小波核初始化
'''


def Morlet(p, c):
    # y = c * torch.exp((-torch.pow(p, 2) / 2).sigmoid()) * torch.cos(5 * p)  #
   # return c * torch.exp((-torch.pow(p, 2) / 2).sigmoid()) * torch.cos(5 * p)
    return c * torch.exp((-torch.pow(p, 2) / 2)) * torch.cos(5 * p)
   #  return c * torch.exp((-torch.pow(p, 2) / 2).sigmoid()) * torch.cos(5 * p)
   #  return c * torch.exp((-torch.pow(p, 2) / 2).tanh()) * torch.cos(5 * p)
    # return c * torch.exp((2/pi)*((-torch.pow(p, 2) / 2).atan())) * torch.cos(5 * p)
    # return c * torch.exp(F.softmax(-torch.pow(p, 2) / 2, dim=-1)) * torch.cos(5 * p)


class Morlet_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, eps=0.3):
        super(Morlet_fast, self).__init__()
        if kernel_size % 2 != 0:
            kernel_size = kernel_size - 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.a_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.b_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        self.time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1, steps=int((self.kernel_size / 2)))

    def forward(self):
        p1 = (self.time_disc_right - self.b_) / (self.a_ + self.eps)  #
        p2 = (self.time_disc_left - self.b_) / (self.a_ + self.eps)  #
        C = (pow(pi, 0.25)) / torch.sqrt(self.a_ + 0.01)  ##一个值得探讨的点 1e-3  # D = C / self.a_.cuda()
        Morlet_right = Morlet(p1, C)
        Morlet_left = Morlet(p2, C)
        filter = torch.cat([Morlet_left, Morlet_right], dim=1).view(self.out_channels, 1, self.kernel_size)
        return filter


'''
Mexh小波卷积核
'''


def Mexh(p):
    # p = 0.04 * p  # 将时间转化为在[-5,5]这个区间内
    # y = (2 / pow(3, 0.5) * (pow(pi, -0.25))) * (1 - torch.pow(p, 2)) * torch.exp((-torch.pow(p, 2) / 2))
    # return ((2 / pow(3, 0.5) * (pow(pi, -0.25))) * (1 - torch.pow(p, 2)) * torch.exp((-torch.pow(p, 2) / 2)))
    # return ((2 / pow(3, 0.5) * (pow(pi, -0.25))) * (1 - torch.pow(p, 2)) * torch.exp((-torch.pow(p, 2) / 2).sigmoid()))
    # return ((2 / pow(3, 0.5) * (pow(pi, -0.25))) * (1 - torch.pow(p, 2)) * torch.exp((-torch.pow(p, 2) / 2).tanh()))
    # return ((2 / pow(3, 0.5) * (pow(pi, -0.25))) * (1 - torch.pow(p, 2)) * torch.exp((2/pi)*((-torch.pow(p, 2) / 2).atan())))
    # return ((2 / pow(3, 0.5) * (pow(pi, -0.25))) * (1 - torch.pow(p, 2)) * torch.exp(F.softmax(-torch.pow(p, 2) / 2, dim=-1)))
    return (2/pi)*((2 / pow(3, 0.5) * (pow(pi, -0.25))) * (1 - torch.pow(p, 2)) * torch.exp((-torch.pow(p, 2) / 2))).atan()

class Mexh_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, eps=0.3):
        super(Mexh_fast, self).__init__()
        if kernel_size % 2 != 0:
            kernel_size = kernel_size - 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.a_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.b_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        self.time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1, steps=int((self.kernel_size / 2)))

    def forward(self):
        p1 = (self.time_disc_right - self.b_) / (self.a_ + self.eps)  #
        p2 = (self.time_disc_left - self.b_) / (self.a_ + self.eps)  #
        Mexh_right = Mexh(p1)
        Mexh_left = Mexh(p2)
        filter = torch.cat([Mexh_left, Mexh_right], dim=1).view(self.out_channels, 1, self.kernel_size)  # 40x1x250
        return filter


'''
Gaussian小波卷积核
'''


def Gaussian(p):
    # y = D * torch.exp(-torch.pow(p, 2))
    # F0 = (2./pi)**(1./4.) * torch.exp(-torch.pow(p, 2))
    # y = -2 / (3 ** (1 / 2)) * (-1 + 2 * p ** 2) * F0
    # y = (2./pi)**(1./4.) * torch.exp(-torch.pow(p, 2))
    # y = -2 / (3 ** (1 / 2)) * (-1 + 2 * p ** 2) * y
    # y = -((1 / (pow(2 * pi, 0.5))) * p * torch.exp((-torch.pow(p, 2)) / 2))
    return -((1 / (pow(2 * pi, 0.5))) * p * (torch.exp(((-torch.pow(p, 2)) / 2))))
    # return -((1 / (pow(2 * pi, 0.5))) * p * (torch.exp(((-torch.pow(p, 2)) / 2).sigmoid())))
    # return -((1 / (pow(2 * pi, 0.5))) * p * (torch.exp(((-torch.pow(p, 2)) / 2).tanh())))
    # return -((1 / (pow(2 * pi, 0.5))) * p * (torch.exp((2/pi)*(((-torch.pow(p, 2)) / 2).atan()))))
    # return -((1 / (pow(2 * pi, 0.5))) * p * (torch.exp(F.softmax((-torch.pow(p, 2)) / 2, dim=-1))))



class Gaussian_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, eps=0.3):
        super(Gaussian_fast, self).__init__()
        if kernel_size % 2 != 0:
            kernel_size = kernel_size - 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.a_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.b_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        self.time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1, steps=int((self.kernel_size / 2)))

    def forward(self):
        p1 = (self.time_disc_right - self.b_) / (self.a_ + self.eps)  #
        p2 = (self.time_disc_left - self.b_) / (self.a_ + self.eps)  #
        Gaussian_right = Gaussian(p1)
        Gaussian_left = Gaussian(p2)
        filter = torch.cat([Gaussian_left, Gaussian_right], dim=1).view(self.out_channels, 1,
                                                                        self.kernel_size)  # 40x1x250
        return filter


def Shannon(p):
    # y = (torch.sin(2 * pi * (p - 0.5)) - torch.sin(pi * (p - 0.5))) / (pi * (p - 0.5))
    return (torch.sin(2 * pi * (p - 0.5)) - torch.sin(pi * (p - 0.5))) / (pi * (p - 0.5))


class Shannon_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, eps=0.3):
        super(Shannon_fast, self).__init__()
        if kernel_size % 2 != 0:
            kernel_size = kernel_size - 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.a_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.b_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        self.time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1, steps=int((self.kernel_size / 2)))
        # self.time_disc_right = torch.linspace(0, 1, steps=int((self.kernel_size / 2)))
        # self.time_disc_left = torch.linspace(-1, 0, steps=int((self.kernel_size / 2)))

    def forward(self):
        p1 = (self.time_disc_right - self.b_) / (self.a_ + self.eps)  #
        p2 = (self.time_disc_left - self.b_) / (self.a_ + self.eps)  #
        Shannon_right = Shannon(p1)
        Shannon_left = Shannon(p2)
        filter = torch.cat([Shannon_left, Shannon_right], dim=1).view(self.out_channels, 1,
                                                                      self.kernel_size)  # 40x1x250
        return filter

def Sin(p):
    # y = (torch.sin(2 * pi * (p - 0.5)) - torch.sin(pi * (p - 0.5))) / (pi * (p - 0.5))
    return torch.sin(p) / p


class Sin_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, eps=0.3):
        super(Sin_fast, self).__init__()
        if kernel_size % 2 != 0:
            kernel_size = kernel_size - 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.a_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.b_ = torch.linspace(0, out_channels, out_channels).view(-1, 1)
        self.time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        self.time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1, steps=int((self.kernel_size / 2)))
        # self.time_disc_right = torch.linspace(0, 1, steps=int((self.kernel_size / 2)))
        # self.time_disc_left = torch.linspace(-1, 0, steps=int((self.kernel_size / 2)))

    def forward(self):
        p1 = (self.time_disc_right - self.b_) / (self.a_ + self.eps)  #
        p2 = (self.time_disc_left - self.b_) / (self.a_ + self.eps)  #
        Shannon_right = Sin(p1)
        Shannon_left = Sin(p2)
        filter = torch.cat([Shannon_left, Shannon_right], dim=1).view(self.out_channels, 1,
                                                                      self.kernel_size)  # 40x1x250
        return filter
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    input = torch.randn(2, 1, 1024).cuda()
    weight = Laplace_fastv2(out_channels=64, kernel_size=250).forward().cuda()
    weight_t = weight.cpu().detach().numpy()
    y = weight_t[20, :, :].squeeze()
    x = np.linspace(0, 250, 250)
    plt.plot(x, y)
    plt.savefig('wahaha.tiff', format='tiff', dpi=600)
    plt.show()
