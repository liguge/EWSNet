import torch
import torch.nn as nn

class Shrinkagev2(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkagev2, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        # x = torch.abs(x)
        x_abs = x.abs()
        x = self.gap(x)
        x = torch.flatten(x, 1)
        # average = torch.mean(x, dim=1, keepdim=True)  #CS
        average = x    #CW
        x = self.fc(x)
        x = torch.mul(average, x).unsqueeze(2)
        # soft thresholding
        x = x_abs - x
        # zeros = sub - sub
        # n_sub = torch.max(sub, torch.zeros_like(sub))
        x = torch.mul(torch.sign(x_raw), torch.max(x, torch.zeros_like(x)))
        return x

class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        # average = torch.mean(x, dim=1, keepdim=True)  #CS
        average = x    #CW
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2)
        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x

class Shrinkagev3(nn.Module):
    def __init__(self, gap_size, inp, oup, reduction=4):
        super(Shrinkagev3, self).__init__()
        mip = int(max(8, inp // reduction))
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Conv1d(inp, mip, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(mip),
            nn.ReLU(inplace=True),
            nn.Conv1d(mip, oup, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    # def forward(self, x):
    #     x_raw = x
    #     x = torch.abs(x)
    #     x_abs = x
    #     x = self.gap(x)
    #     # x = torch.flatten(x, 1)
    #     # average = torch.mean(x, dim=1, keepdim=True)  #CS
    #     average = x    #CW
    #     x = self.fc(x)
    #     x = torch.mul(average, x)
    #     # x = x.unsqueeze(2)
    #     # soft thresholding
    #     sub = x_abs - x
    #     zeros = sub - sub
    #     n_sub = torch.max(sub, zeros)
    #     x = torch.mul(torch.sign(x_raw), n_sub)
    #     return x
    def forward(self, x):
        x_raw = x
        # x = torch.abs(x)
        x_abs = x.abs()
        x = self.gap(x)
        # average = torch.mean(x, dim=1, keepdim=True)  #CS
        average = x    #CW
        x = self.fc(x)
        x = torch.mul(average, x)
        # soft thresholding
        x = x_abs - x
        # zeros = sub - sub
        # n_sub = torch.max(sub, torch.zeros_like(sub))
        x = torch.mul(torch.sign(x_raw), torch.max(x, torch.zeros_like(x)))
        return x

class Shrinkagev3p(nn.Module):
    def __init__(self, gap_size, inp, oup, reduction=4):
        super(Shrinkagev3p, self).__init__()
        mip = int(max(8, inp // reduction))
        self.a = nn.Parameter(torch.tensor([0.4]))
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        # self.gap = nn.AdaptiveMaxPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Conv1d(inp, mip, kernel_size=1, stride=1),
            nn.BatchNorm1d(mip),
            nn.ReLU(inplace=True),
            nn.Conv1d(mip, oup, kernel_size=1, stride=1),
            nn.Hardsigmoid()
            # nn.Sigmoid()
        )

    # def forward(self, x):
    #     x_raw = x
    #     x = torch.abs(x)
    #     x_abs = x
    #     x = self.gap(x)
    #     # x = torch.flatten(x, 1)
    #     # average = torch.mean(x, dim=1, keepdim=True)  #CS
    #     average = x    #CW
    #     x = self.fc(x)
    #     x = torch.mul(average, x)
    #     # x = x.unsqueeze(2)
    #     # soft thresholding
    #     sub = x_abs - x
    #     zeros = sub - sub
    #     n_sub = torch.max(sub, zeros)
    #     x = torch.mul(torch.sign(x_raw), n_sub)
    #     return x
    def forward(self, x):
        x_raw = x
        # x = torch.abs(x)
        x_abs = x.abs()
        x = self.gap(x)
        # average = torch.mean(x, dim=1, keepdim=True)  #CS
        average = x    #CW
        x = self.fc(x)
        x = torch.mul(average, x)
        # soft thresholding
        # x = x_abs - x
        x = x_abs - self.a * x
        # zeros = sub - sub
        # n_sub = torch.max(sub, torch.zeros_like(sub))
        x = torch.mul(torch.sign(x_raw), torch.max(x, torch.zeros_like(x)))
        return x

class Shrinkagev3p1(nn.Module):
    def __init__(self, gap_size, inp, oup, reduction=32):
        super(Shrinkagev3p1, self).__init__()
        mip = int(max(8, inp // reduction))
        #self.a = nn.Parameter(torch.tensor([0.5]))
        self.a = nn.Parameter(0.5 * torch.ones(1, 10, 1))
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Conv1d(inp, mip, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(mip),
            nn.ReLU(inplace=True),
            nn.Conv1d(mip, oup, kernel_size=1, stride=1, padding=0),
            nn.Hardsigmoid()
            # nn.Sigmoid()
        )

    # def forward(self, x):
    #     x_raw = x
    #     x = torch.abs(x)
    #     x_abs = x
    #     x = self.gap(x)
    #     # x = torch.flatten(x, 1)
    #     # average = torch.mean(x, dim=1, keepdim=True)  #CS
    #     average = x    #CW
    #     x = self.fc(x)
    #     x = torch.mul(average, x)
    #     # x = x.unsqueeze(2)
    #     # soft thresholding
    #     sub = x_abs - x
    #     zeros = sub - sub
    #     n_sub = torch.max(sub, zeros)
    #     x = torch.mul(torch.sign(x_raw), n_sub)
    #     return x
    def forward(self, x):
        x_raw = x
        # x = torch.abs(x)
        x_abs = x.abs()
        x = self.gap(x)
        # average = torch.mean(x, dim=1, keepdim=True)  #CS
        average = x    #CW
        x = self.fc(x)
        x = torch.mul(average, x)
        # soft thresholding
        x = x_abs - self.a * x
        # zeros = sub - sub
        # n_sub = torch.max(sub, torch.zeros_like(sub))
        x = torch.mul(torch.sign(x_raw), torch.max(x, torch.zeros_like(x)))
        return x
class Shrinkagev3p11(nn.Module):
    def __init__(self, gap_size, channel):
        super(Shrinkagev3p11, self).__init__()
        #self.a = nn.Parameter(torch.tensor([0.5]))
        self.a = nn.Parameter(torch.tensor([0.48]))
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        x_raw = x
        # x = torch.abs(x)
        x_abs = x.abs()
        x = self.gap(x)
        x = torch.flatten(x, 1)
        # average = torch.mean(x, dim=1, keepdim=True)  #CS
        average = x    #CW
        x = self.fc(x)
        x = torch.mul(average, x).unsqueeze(2)
        # soft thresholding
        x = x_abs - self.a * x
        # zeros = sub - sub
        # n_sub = torch.max(sub, torch.zeros_like(sub))
        x = torch.mul(torch.sign(x_raw), torch.max(x, torch.zeros_like(x)))
        return x
class Shrinkagev3pp(nn.Module):   #会议文献的软阈值降噪
    def __init__(self, gap_size, inp, oup, reduction=4):
        super(Shrinkagev3pp, self).__init__()
        mip = int(max(8, inp // reduction))
        self.a = nn.Parameter(torch.tensor([0.48]))
        # self.a = nn.Parameter(torch.clamp(torch.tensor([0.4]), min=0, max=1))
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        # self.gap = nn.AdaptiveMaxPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Conv1d(inp, mip, kernel_size=1, stride=1),
            nn.BatchNorm1d(mip),
            nn.ReLU(inplace=True),
            nn.Conv1d(mip, oup, kernel_size=1, stride=1),
            nn.Hardsigmoid()
            # nn.Sigmoid()
        )

    # def forward(self, x):
    #     x_raw = x
    #     x = torch.abs(x)
    #     x_abs = x
    #     x = self.gap(x)
    #     # x = torch.flatten(x, 1)
    #     # average = torch.mean(x, dim=1, keepdim=True)  #CS
    #     average = x    #CW
    #     x = self.fc(x)
    #     x = torch.mul(average, x)
    #     # x = x.unsqueeze(2)
    #     # soft thresholding
    #     sub = x_abs - x
    #     zeros = sub - sub
    #     n_sub = torch.max(sub, zeros)
    #     x = torch.mul(torch.sign(x_raw), n_sub)
    #     return x
    def forward(self, x):
        x_raw = x
        # x = torch.abs(x)
        x_abs = x.abs()
        x = self.gap(x)

        # average = torch.mean(x, dim=1, keepdim=True)  #CS
        average = x    #CW
        x = self.fc(x)
        x = torch.mul(average, x)
        # soft thresholding
        x = x_abs - x
        # x = x_abs - self.a * x
        # sub = torch.max(x1, torch.zeros_like(x1))
        # sub = self.a * sub
        a = torch.clamp(self.a, min=0, max=1)
        x = torch.mul(torch.sign(x_raw), a*(torch.max(x, torch.zeros_like(x))))
        return x
class Shrinkagev3pp1(nn.Module):   #会议文献的软阈值降噪
    def __init__(self, gap_size, channel):
        super(Shrinkagev3pp1, self).__init__()
        self.a = nn.Parameter(torch.tensor([0.48]))
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_raw = x
        x_abs = x.abs()
        x = self.gap(x)
        x = torch.flatten(x, 1)
        # average = torch.mean(x, dim=1, keepdim=True)  #CS
        average = x    #CW
        x = self.fc(x)
        x = torch.mul(average, x).unsqueeze(2)
        # soft thresholding
        x = x_abs - x
        a = torch.clamp(self.a, min=0, max=1)
        x = torch.mul(torch.sign(x_raw), a*(torch.max(x, torch.zeros_like(x))))
        return x
class Shrinkagev3ppp(nn.Module):   #半软阈值降噪
    def __init__(self, gap_size, inp, oup, reduction=4):
        super(Shrinkagev3ppp, self).__init__()
        mip = int(max(8, inp // reduction))
        self.a = nn.Parameter(torch.tensor([0.48]))
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        # self.gap = nn.AdaptiveMaxPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Conv1d(inp, mip, kernel_size=1, stride=1),
            nn.BatchNorm1d(mip),
            nn.ReLU(inplace=True),
            nn.Conv1d(mip, oup, kernel_size=1, stride=1),
            # nn.Hardsigmoid()
            nn.Sigmoid()
        )

    def forward(self, x):
        x_raw = x
        x_abs = x.abs()
        x = self.gap(x)
        average = x
        x = self.fc(x)
        x = torch.mul(average, x)
        # soft thresholding
        # x1 = x_abs - x
        # sub = torch.max(x1, torch.zeros_like(x1))
        sub = torch.max(x_abs - x, torch.zeros_like(x_abs - x))
        mask = sub.clone()
        mask[mask > 0] = 1
        # a = torch.clamp(self.a, min=0, max=1)
        x = sub + (1-self.a) * x
        x = torch.mul(x, mask)
        x = torch.mul(torch.sign(x_raw), x)
        return x

class Shrinkagev3ppp1(nn.Module):
    def __init__(self, gap_size, inp, oup, reduction=4):
        super(Shrinkagev3ppp1, self).__init__()
        mip = int(max(8, inp // reduction))
        self.a = nn.Parameter(torch.tensor([0.4]))
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        # self.gap = nn.AdaptiveMaxPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Conv1d(inp, mip, kernel_size=1, stride=1),
            nn.BatchNorm1d(mip),
            nn.ReLU(inplace=True),
            nn.Conv1d(mip, oup, kernel_size=1, stride=1),
            nn.Hardsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        x_raw = x
        x_abs = x.abs()
        x = self.gap(x)
        average = x
        x = self.fc(x)
        x = torch.mul(average, x)
        # soft thresholding
        sub = torch.max(x_abs - x, torch.zeros_like(x_abs - x))
        x = sub + (1-self.a) * x
        x = x.index_put((sub == 0).nonzero(as_tuple=True), torch.tensor(0.))
        x = torch.mul(torch.sign(x_raw), x)
        return x

class Shrinkagev3ppp2(nn.Module):   #半软阈值降噪
    def __init__(self, gap_size, channel):
        super(Shrinkagev3ppp2, self).__init__()
        self.a = nn.Parameter(torch.tensor([0.48]))
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):

        x_raw = x
        x_abs = x.abs()
        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = x
        # average = torch.mean(x, dim=1, keepdim=True)
        x = self.fc(x)
        x = torch.mul(average, x).unsqueeze(2)
        # soft thresholding
        sub = torch.max(x_abs - x, torch.zeros_like(x_abs - x))
        mask = sub.clone()
        mask[mask > 0] = 1
        # a = torch.clamp(self.a, min=0, max=1)
        x = sub + (1 - self.a) * x
        x = torch.mul(torch.sign(x_raw), torch.mul(x, mask))
        return x

# class Shrinkagev3ppp3(nn.Module):   #半软阈值降噪
#     def __init__(self, gap_size, channel):
#         super(Shrinkagev3ppp3, self).__init__()
#         self.a = nn.Parameter(0.48 * torch.ones(16, channel, 1))
#         self.gap = nn.AdaptiveAvgPool1d(gap_size)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel),
#             nn.BatchNorm1d(channel),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel, channel),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#
#         x_raw = x
#         x_abs = x.abs()
#         x = self.gap(x)
#         x = torch.flatten(x, 1)
#         average = x
#         # average = torch.mean(x, dim=1, keepdim=True)
#         x = self.fc(x)
#         x = torch.mul(average, x).unsqueeze(2)
#         # soft thresholding
#         sub = torch.max(x_abs - x, torch.zeros_like(x_abs - x))
#         mask = sub.clone()
#         mask[mask > 0] = 1
#         # a = torch.clamp(self.a, min=0, max=1)
#         x = sub + (1 - self.a) * x
#         x = torch.mul(torch.sign(x_raw), torch.mul(x, mask))
#         return x

class HShrinkage(nn.Module):   #硬阈值降噪
    def __init__(self, gap_size, channel):
        super(HShrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):

        x_raw = x
        x_abs = x.abs()
        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = x
        # average = torch.mean(x, dim=1, keepdim=True)
        x = self.fc(x)
        x = torch.mul(average, x).unsqueeze(2)
        # soft thresholding
        sub = torch.max(x_abs - x, torch.zeros_like(x_abs - x))
        mask = sub.clone()
        mask[mask > 0] = 1
        x = torch.mul(x_raw, mask)
        return x
class Shrinkagev3ppp31(nn.Module):   #半软阈值降噪
    def __init__(self, gap_size, channel):
        super(Shrinkagev3ppp31, self).__init__()
        self.a = nn.Parameter(torch.tensor([0.48]))
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):

        x_raw = x
        x_abs = x.abs()
        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = x
        # average = torch.mean(x, dim=1, keepdim=True)
        x = self.fc(x)
        x = torch.mul(average, x).unsqueeze(2)
        # soft thresholding
        sub = torch.max(x_abs - x, torch.zeros_like(x_abs - x))
        # mask = sub.clone()
        # mask[mask > 0] = 1
        # a = torch.clamp(self.a, min=0, max=1)
        x = torch.mul(sub, (1 - self.a))
        x = torch.add(torch.mul(torch.sign(x_raw), x), torch.mul(self.a, x_raw))
        return x
if __name__ == '__main__':
    input = torch.randn(2, 3, 1024).cuda()
    model = Shrinkagev3ppp(1, 3, 3).cuda()
    for param in model.parameters():
        print(type(param.data), param.size())
    output = model(input)
    print(output.size())