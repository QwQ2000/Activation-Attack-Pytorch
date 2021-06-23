from torchvision.models import resnet18
from torch import nn

class ResNet18FeatureExtractor(nn.Module):
    def __init__(self,model):
        super(ResNet18FeatureExtractor,self).__init__()
        self.model = model
    
    def forward(self,x):
        m = self.model
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        #x = m.layer4(x)
        return x