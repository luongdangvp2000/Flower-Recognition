import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

class VGG16(nn.Module):
    def __init__(self, n_classes):
        super(VGG16, self).__init__()
        self.n_classes= n_classes
        self.conv1_1= nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,padding=1)
        self.conv1_2= nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1)
        
        self.conv2_1= nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,padding=1)
        self.conv2_2= nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,padding=1)

        self.conv3_1= nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,padding=1)
        self.conv3_2= nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,padding=1)
        self.conv3_3= nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,padding=1)
        
        self.conv4_1= nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,padding=1)
        self.conv4_2= nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,padding=1)
        self.conv4_3= nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,padding=1)
        
        self.conv5_1= nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,padding=1)
        self.conv5_2= nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,padding=1)
        self.conv5_3= nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,padding=1)
        
        self.maxpool= nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool= nn.AdaptiveAvgPool2d((1,1))
        
        self.fc1= nn.Linear(512,128)
        self.fc2= nn.Linear(128,self.n_classes)
        #self.fc3= nn.Linear(128, self.n_classes)
        

    def forward(self,x):
        x= F.relu(self.conv1_1(x))
        x= F.relu(self.conv1_2(x))
        x= self.maxpool(x)
        x= F.relu(self.conv2_1(x))
        x= F.relu(self.conv2_2(x))
        x= self.maxpool(x)
        x= F.relu(self.conv3_1(x))
        x= F.relu(self.conv3_2(x))
        x= F.relu(self.conv3_3(x))
        x= self.maxpool(x)
        x= F.relu(self.conv4_1(x))
        x= F.relu(self.conv4_2(x))
        x= F.relu(self.conv4_3(x))
        x= self.maxpool(x)
        x= F.relu(self.conv5_1(x))
        x= F.relu(self.conv5_2(x))
        x= F.relu(self.conv5_3(x))
        x= self.maxpool(x)
        x= x.reshape(x.shape[0], -1)
        x= F.relu(self.fc1(x))
        x= F.dropout(x, 0.5) #prevent overfitting
        x= self.fc2(x)
#         x= F.dropout(x, 0.5)
#         x= self.fc3(x)
    
        return x

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.n_classes = n_classes 
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)   
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)    
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))   
        
        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d(1), #nn.Sequential: combine some networks, to decrease length of model
                                        nn.Flatten(),     
                                        nn.Dropout(0.2),
                                        nn.Linear(512, n_classes))    
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out