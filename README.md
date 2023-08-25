# Physics-informed Interpretable Wavelet Weight Initialization and Balanced Dynamic Adaptive Threshold for Intelligent Fault Diagnosis of Rolling Bearings

# Updating!

[Sorry]The upload of the code needs to 1-2 weeks, thank you very much for your attention!

[NEWS!]This paper has been accepted by **<font color="blue">[Journal of Manufacturing Systems](https://www.sciencedirect.com/journal/journal-of-manufacturing-systems)</font>**!

[NOTE!!]The code will be gradually and continuously opened!

## Brief introduction  
Intelligent fault diagnosis of rolling bearings using deep learning-based methods has made unprecedented progress. However, there is still little research on weight initialization and the threshold setting for noise reduction. An innovative deep triple-stream network called EWSNet is proposed, which presents a wavelet weight initialization method and a balanced dynamic adaptive threshold algorithm. Initially, an enhanced wavelet basis function is designed, in which a scale smoothing factor is defined to acquire more rational wavelet scales. Next, a plug-and-play wavelet weight initialization for deep neural networks is proposed, which utilizes physics-informed wavelet prior knowledge and showcases stronger applicability. Furthermore, a balanced dynamic adaptive threshold is established to enhance the noise-resistant robustness of the model. Finally, normalization activation mapping is devised to reveal the effectiveness of Z-score from a visual perspective rather than experimental results. The validity and reliability of EWSNet are demonstrated through four data sets under the conditions of constant and fluctuating speeds.

## Highlights

- **A novel deep triple-stream network called EWSNet is proposed for fault diagnosis of rolling bearings under the condition of constant or sharp speed variation.**
- **An enhanced wavelet convolution kernel is designed to improve the trainability, in which a scale smoothing factor is employed to acquire rational wavelet scales.**
- **A plug-and-play and physics-informed wavelet weight initialization is proposed to construct an interpretable convolution kernel, which makes the diagnosis interpretable.**
- **Balanced dynamic adaptive threshold is specially devised to improve the antinoise robustness of the model.**
- **Normalization activation mapping is designed to visually reveal that Z-score can enhance the frequencydomain information of raw signals.**


## Paper
**Physics-informed Interpretable Wavelet Weight Initialization and Balanced Dynamic Adaptive Threshold for Intelligent Fault Diagnosis of Rolling Bearings**  
Chao He<sup>a,b</sup>, Hongmei Shi<sup>a,b</sup>, Jin Si<sup>c</sup> and Jianbo Li<sup>a,b</sup>

<sup>a</sup>School of Mechanical, Electronic and Control Engineering, Beijing Jiaotong University, Beijing 100044, China 

<sup>b</sup>Collaborative Innovation Center of Railway Traffic Safety, Beijing 100044, China 

<sup>c</sup>Beijing Institute of Control and Electronic Technology, Beijing 100038, China  

**Journal of Manufacturing Systems**

## Wavelet initialization

![image](https://user-images.githubusercontent.com/19371493/180359513-b6fd1fb4-4c63-47ad-8d98-b8030d2ca529.png)

## Balanced Dynamic Adaptive Thresholding

![image](https://user-images.githubusercontent.com/19371493/190544070-b8a3a630-6fc4-48d4-9693-53253a40752f.png)

![image](https://user-images.githubusercontent.com/19371493/180358950-fcb9b417-7306-4fc0-b99b-5952c59b941f.png)

## Example:



```python
class CNNNet(nn.Module):

    def __init__(self, init_weights=False):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 60, padding=2)
        self.conv2 = nn.Conv1d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(16, 48, 3, padding=1)
        #self.sage = sage(channel=16, gap_size=1)
        self.conv4 = nn.Conv1d(24, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32*60, 512)
        self.fc2 = nn.Linear(512, 4)
        if init_weights:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if m.kernel_size == (60,):
                    m.weight.data = fast(out_channels=64, kernel_size=60, frequency=100000, eps=0.2, mode='sigmoid').forward()
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.sage(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
   
   
    


```



## Citation

(Please wait for updating ...)

## Ackowledgements
(Please wait for updating ...)
