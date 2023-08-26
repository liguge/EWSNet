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
- **Normalization activation mapping is designed to visually reveal that Z-score can enhance the frequency-domain information of raw signals.**


## Paper
**Physics-informed Interpretable Wavelet Weight Initialization and Balanced Dynamic Adaptive Threshold for Intelligent Fault Diagnosis of Rolling Bearings**  

Chao He<sup>a,b</sup>, Hongmei Shi<sup>a,b,*</sup>, Jin Si<sup>c</sup> and Jianbo Li<sup>a,b</sup>

<sup>a</sup>School of Mechanical, Electronic and Control Engineering, Beijing Jiaotong University, Beijing 100044, China 

<sup>b</sup>Collaborative Innovation Center of Railway Traffic Safety, Beijing 100044, China 

<sup>c</sup>Key laboratory of information system and technology, Beijing institute of control and electronic technology, Beijing 100038, China  

**Journal of Manufacturing Systems**

## Wavelet initialization

![image](https://user-images.githubusercontent.com/19371493/180359513-b6fd1fb4-4c63-47ad-8d98-b8030d2ca529.png)

## Balanced Dynamic Adaptive Thresholding

![image](https://user-images.githubusercontent.com/19371493/190544070-b8a3a630-6fc4-48d4-9693-53253a40752f.png)

![image](https://user-images.githubusercontent.com/19371493/180358950-fcb9b417-7306-4fc0-b99b-5952c59b941f.png)

**where $\alpha$ and $\eta$ are differentiable($\alpha  \in \left( {0,1} \right),\alpha  \ne 0,1$). When $\alpha$=0 or 1, Eq. respectively degenerates into hard threshold and soft threshold, and thus we can adjust $\alpha$ appropriately to make $y$ closer to the genuine wavelet coefficient.**

## Normalization Activation Mapping

**Data normalization can accelerate the process of convergence. Also, Z-score makes CNN get better accuracy. Unlike experimental methods, we notice that Z-score enhances frequency-domain information of signals so that CNN can learn these features better.**

**FAM illustrates the frequency-domain information by utilizing the weights of the classification layer and extracted features, but it can not reveal the influence of normalization methods. Therefore in NAM, the weight of the correct label is $1.0$, and the features are signals processed by the normalization methods and it can visualize which normalization method possesses more frequency-domain knowledge.**

![4eb60742e697c828151434ed263b922](https://github.com/liguge/EWSNet_new/assets/19371493/3e9cddd7-3904-466a-8a59-823f6fea3400)
where ${l_{real}}$ is the real label and  ${l_{target}}$ is the tested label.

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
                    m.weight.data = fast(out_channels=64, kernel_size=60, eps=0.2, mode='sigmoid').forward()
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
