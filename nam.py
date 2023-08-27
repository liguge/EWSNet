import torch
import torch.nn.functional as F
from Elaplacenet5 import Net
from gdatasave import train_loader
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
torch.cuda.manual_seed(1)
torch.manual_seed(2)

sampling_rate = 50000
window_size = 1024
n_class = 4
channels = [64]


def normalize_freq(fam):
    max_fam, _ = torch.max(fam, dim=-1, keepdim=True)
    return torch.div(fam, max_fam)
# This is from P. Welch's method to compute power spectrum, check [3]
def power_spectrum(t_freq):
    result = torch.abs(torch.fft.rfft(t_freq))**2
    return result / torch.mean(result, dim=2, keepdim=True)


def freq_activation_map(model, input, width, channels, target_label):
    '''
        Param:
            model : Neural Network Object
            input : timeseries data
            width : length of power_spectrum(input)
            channels : # last channel of the model
    '''
    fam = torch.zeros(input.shape[0], 1, width)
    if torch.cuda.is_available():
        fam = fam.cuda()

    with torch.no_grad():
        freq, labels = model(torch.reshape(input, (-1, 1, input.shape[-1])))
        freq = freq.unsqueeze(2)
        labels = labels.unsqueeze(2)
        labels = torch.argmax(F.softmax(labels, dim=1), dim=1)
        labels = torch.where(labels == target_label, 1., 0.)
        labels = torch.unsqueeze(torch.reshape(labels, [-1, 1]).repeat(1, width), 1)
        for c in range(channels):
            sp = freq[:, c, :, :]
            if torch.cuda.is_available():
                sp = sp.cuda()

            sp = power_spectrum(sp)
            sp = sp * labels

            if model.p4[0].weight[target_label, c] > 0:
                fam += model.p4[0].weight[target_label, c] * sp

    return torch.squeeze(torch.sum(fam, dim=0)), torch.sum(labels[:, 0, 0], dim=0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#summary(Net1, input_size=(1, 1024))  # 输出模型具有的参数
model = Net().to(device)
model.load_state_dict(torch.load('./model1001.pt'))
torch.no_grad()
# for img, label in test_loader:
'''
测试集
'''
# Test.Data = Test.Data.type(torch.FloatTensor).to(device)
# label = torch.from_numpy(Test.Label)
# outputs = model(Test.Data)
# outputs = torch.squeeze(outputs).float()
# encoded_data = outputs.cpu().detach().numpy()
# clf = SVC(C=8.94, gamma=3.26)
'''
训练集
'''
# Train.Data = Train.Data.type(torch.FloatTensor).to(device)
# labels = torch.from_numpy(Train.Label)
# outputs = model(Train.Data)
# outputs = torch.squeeze(outputs).float()
# encoded_data = outputs.cpu().detach().numpy()

# clf = SVC(C=10.82, kernel='rbf', gamma=1.19, decision_function_shape='ovr')  # rbf高斯基核函数
# clf.fit(encoded_data, label.cpu().numpy())
# fit_score = clf.score(encoded_data, label.cpu().numpy())
# print(fit_score)

freq_intervals = np.fft.rfftfreq(window_size, d=1/sampling_rate)
total_fam = torch.zeros(n_class, len(freq_intervals))
total_len = torch.zeros(n_class, 1)
if torch.cuda.is_available():
    total_fam = total_fam.cuda()
    total_len = total_len.cuda()

for batch in tqdm(train_loader):
    x, y = batch
    if torch.cuda.is_available():
        x = x.cuda()
    for n in range(n_class):
        tmp_fam, cnt = freq_activation_map(model, x.float(), len(freq_intervals), channels[-1], n)
        total_fam[n, :] += tmp_fam
        total_len[n] += cnt

total_fam /= total_len
print(total_fam.size())
# _, predicted = outputs.max(1)
# print(predicted, Test.Label)
# num_correct = (predicted.cpu() == Test.Label).sum().item()
# acc = num_correct / outputs.cpu().shape[0]
# eval_acc = 0
# eval_acc += fit_score
# print(eval_acc)
rtotal_fam = normalize_freq(total_fam).cpu().detach()
columns_ = np.floor(freq_intervals).astype(int)
plt.plot(columns_, rtotal_fam[0, :])

result = pd.DataFrame(rtotal_fam.numpy(),
                      index = ['0', '1', '2', '3'],
                      columns = columns_,
                      )

new_index = ['3', '2', '1', '0']
print(result.values.shape)
new_index.reverse()

result = result.reindex(new_index)

sns.heatmap(result, cmap='viridis')
plt.show()

