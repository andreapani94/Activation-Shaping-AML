import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

ratio = 1

class BaseResNet18(nn.Module):
    def __init__(self):
        super(BaseResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def forward(self, x):
        return self.resnet(x)
    
class DAResNet18(nn.Module):
    def __init__(self):
        super(DAResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.actmaps_target = []
        self.forward_turn = 'target'

    def forward(self, x_source, x_target=None):
        # unregister other forward hooks
        # register forward hooks
        # unregister forward hooks
        if x_target is not None:
            self.forward_turn == 'target'
            self.resnet(x_target)
        self.forward_turn = 'source'
        return self.resnet(x_source)
    
    def rec_actmaps_hook(self, module, input, output):
        if self.forward_turn == 'target':
            print(f"rec_actmaps_hook triggered for module: {module.__class__.__name__}")
        #print(f"actmaps length: {len(self.actmaps_target)}")
        #self.actmaps_target.append(output.detach())
        return output
    
    def asm_source_hook(self, module, input, output):
            if self.forward_turn == 'source':
                print(f"asm_source_hook triggered for module: {module.__class__.__name__}")
            #print(f"actmaps length: {len(self.actmaps_target)}")
            #mask = self.actmaps_target.pop(0)
            #mask_bin = (mask > 0).float()
            #output_bin = (output > 0).float()
            #return output_bin * mask_bin
            return output

def activation_shaping_hook(module, input, output):
    #print(f"Activation hook triggered for module: {module.__class__.__name__}")
    p = torch.full_like(output, ratio)
    mask = torch.bernoulli(p)
    mask_bin = (mask > 0).float()
    output_bin = (output > 0).float()
    return output_bin * mask_bin



######################################################
# TODO: either define the Activation Shaping Module as a nn.Module
#class ActivationShapingModule(nn.Module):
#...
#
# OR as a function that shall be hooked via 'register_forward_hook'
#def activation_shaping_hook(module, input, output):
#...
#
######################################################
# TODO: modify 'BaseResNet18' including the Activation Shaping Module
#class ASHResNet18(nn.Module):
#    def __init__(self):
#        super(ASHResNet18, self).__init__()
#        ...
#    
#    def forward(self, x):
#        ...
#
######################################################
