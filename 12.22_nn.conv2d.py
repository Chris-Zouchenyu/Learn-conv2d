import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter
dataset = torchvision.datasets.CIFAR10('./dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)
data_loader = DataLoader(dataset,batch_size=64)
class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)
    
    def forward(self,x):
        x = self.conv1(x)
        return x
    
N = NN()
#print(N)
writer = SummaryWriter('logs')
step = 0
for data in data_loader:
    imgs,targets = data
    output = N(imgs)
    #torch.size([64,3,32,32])
    writer.add_images('input',imgs,step)
    #torch.size([64,6,30,30]) 但是tensorboard只能显示3通道的
    #torch.size([64,6,30,30]) -> [xxx,3,30,30] 未知用-1
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images('output',output,step)
    step+=1
    #print(output.shape)

writer.close()