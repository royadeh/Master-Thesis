import math
import torch
from torch import nn
from .AIBN import AIBNorm2d
from .TNorm import TNorm
import numpy as np
import torch
import io
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


class AIBNBlock(nn.Module):
    
    def __init__(self, dim, resitual_path=0., layer_scale_init_value=1e-6, 
                 adaptive_weight=None,#new
                 generate_weight=True,#new
                 init_weight=0.1,#new
                 ):
        super(AIBNBlock, self).__init__()#new
        if adaptive_weight is None:#new
            self.adaptive_weight = nn.Parameter(torch.ones(1) * init_weight)#new
        #initializes a depthwise convolution with kernel size 7 and padding 3.
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        #self.norm = LayerNorm(dim, eps=1e-6)
        
        #use AIBN normalizaton
        self.aibnNorm= AIBNorm2d(dim,#new
                             adaptive_weight=self.adaptive_weight,#new
                             generate_weight=generate_weight)#new
        
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()# This line initializes the activation function. GELU (Gaussian Error Linear Units) is a non-linear activation function that is used in some deep learning models
        self.pwconv2 = nn.Linear(4 * dim, dim)# initializes a pointwise convolution (1x1 convolution) with 4 times the output channels as input channel
       
        #This line initializes a scaling factor parameter for the block. If layer_scale_init_value is greater than 0, the scaling factor is initialized with that value multiplied by
        # a tensor of ones with dim elements. Otherwise, the scaling factor is set to None.
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        
        
        self.resitual_path = DropPath(resitual_path) if resitual_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # #This line permutes the dimensions of the output tensor of the depthwise convolution to be in the form of (N, H, W, C)
        #x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)#new
        #x = self.norm(x)# This line applies layer normalization to the output tensor
        
        #because the LayerNorm takes the input shape with (N, H, W, C), we change th eshape of input to (N, C, H, W)
        #for aibnNorm
        x=self.aibnNorm(x)#new
        
         
        #the input shape for Pointwise conv should be (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)#This line applies the first pointwise convolution to the output tensor.
        x = self.act(x)#This line applies the activation function to the output tensor.
        x = self.pwconv2(x)#This line applies the second pointwise convolution to the output tensor.
        
        # This line applies the scaling factor parameter to the output tensor if it is not None.
        if self.gamma is not None:
            x = self.gamma * x
        
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)#new
        #residual connection
        x = input + self.resitual_path(x)
        return x


class ConvNeXt(nn.Module):
    
      def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 init_weight=0.1,#new
                 adaptive_weight=None#new
                 ):
        super().__init__()
    
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        
        #stem
        self.stem = nn.Sequential(
            #4*4, 96, stride 4
            #in_chans is 3 for RBG image i think
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        
        
        # self.downsample_layers.append(stem)
        #for i in range(3):
        #############downsample for stage2
        self.downsample_layer1 = nn.Sequential(
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[0], dims[0+1], kernel_size=2, stride=2),
        )
        # self.downsample_layers.append(downsample_layer1)
        ################downsample for stage3
        self.downsample_layer2 = nn.Sequential(
                LayerNorm(dims[1], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[1], dims[1+1], kernel_size=2, stride=2),
        )
        # self.downsample_layers.append(downsample_layer2)
        ##################downsample for stage4
        self.downsample_layer3 = nn.Sequential(
                LayerNorm(dims[2], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[2], dims[2+1], kernel_size=2, stride=2),
        )
        # self.downsample_layers.append(downsample_layer3)


        #4 stages
        # self.stages = nn.ModuleList() #to store all stages
        
        # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        
        #for i in range(4):
        #######################stage1
        self.stage1 = nn.Sequential(
        #call Block class 
        *[Block(dim=dims[0], drop_path=dp_rates[cur + j], 
        layer_scale_init_value=layer_scale_init_value) for j in range(depths[0])]
        )
        # self.stages.append(stage1)
        # the cur variable is updated to keep track of the current position in the dp_rates list.
        cur += depths[0]
        
        self.tnorm1 = TNorm(dim[0], domain_number)
        ################stage2
        self.stage2 = nn.Sequential(
            #call Block class 
            *[Block(dim=dims[1], drop_path=dp_rates[cur + j], 
            layer_scale_init_value=layer_scale_init_value) for j in range(depths[1])]
        )
        # self.stages.append(stage2)
        # the cur variable is updated to keep track of the current position in the dp_rates list.
        cur += depths[1]
        self.tnorm2 = TNorm(dim[1], domain_number)
        ################stage3
        self.stage3 = nn.Sequential(
            
            
               *[ AIBNBlock(dim=dims[2], resitual_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value,
                 adaptive_weight=None,#new
                 generate_weight=True,#new
                 init_weight=0.1#new
                ) for j in range(depths[2])]
            
            
            # #call Block class 
            # *[Block(dim=dims[2], drop_path=dp_rates[cur + j], 
            # layer_scale_init_value=layer_scale_init_value) for j in range(depths[2])]
        )
        # self.stages.append(stage3)
        # the cur variable is updated to keep track of the current position in the dp_rates list.
        cur += depths[2]
        self.tnorm3 = TNorm(dim[2], domain_number)
        #########################stage4
        self.stage4 = nn.Sequential(
            
            
               *[ AIBNBlock(dim=dims[3], resitual_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value,
                 adaptive_weight=None,#new
                 generate_weight=True,#new
                 init_weight=0.1#new
                ) for j in range(depths[3])]
            
            
            
            # #call Block class 
            # *[Block(dim=dims[3], drop_path=dp_rates[cur + j], 
            # layer_scale_init_value=layer_scale_init_value) for j in range(depths[3])]
        )
        # self.stages.append(stage4)
        # the cur variable is updated to keep track of the current position in the dp_rates list.
        cur += depths[3]
            


       
        #final layers
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)

        # self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x, domain_index=None, convert=False):
        # for i in range(4):
        #     x = self.downsample_layers[i](x)
        #     x = self.stages[i](x)
            
          def forward(self, x, domain_index=None, convert=False):
        if convert:
            selected_domain = np.random.randint(0,
                                                self.domain_number,
                                                size=(x.size(0)))
        else:
            selected_domain = None
        x= self.stem(x)
        x= self.stage1(x)
        x= self.tnorm1(x, domain_index, convert, selected_domain)
        x= self.downsample_layer1(x) 
        x= self.stage2(x)
        x= self.tnorm2(x, domain_index, convert, selected_domain) 
        x= self.downsample_layer2(x)
        x= self.stage3(x)
        x= self.tnorm3(x, domain_index, convert, selected_domain) 
        x= self.downsample_layer3(x)
        x= self.stage4(x)
        
           
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        #x = self.head(x)
        return x
    
    
   


    def load_param(self, model_path):
            param_dict = torch.load(model_path)
            for k, v in param_dict.items():
                if k in self.state_dict().keys():
                    self.state_dict()[k].copy_(v)    









class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for k, v in param_dict.items():
            if k in self.state_dict().keys():
                self.state_dict()[k].copy_(v)


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}



"""
@register_model is a decorator used to register a model function with PyTorch's hub module.
The hub module is a pre-trained model repository maintained by PyTorch, where users can find and use pre-trained models for various tasks. By registering a model 
function with @register_model, users can load the model easily using torch.hub.load() function with a model name as an argument.
"""
@register_model
def convnext_tiny(pretrained,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    #print("model in convnext_tiny",model)
    return model

@register_model
def convnext_small(pretrained,in_22k=True, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

# ConvNeXt-XL model trained on ImageNet-1K
@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model



#The loaded checkpoint may have been trained on ImageNet-22K, which has 21841 classes.
@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        #The state_dict is a Python dictionary object that maps each layer to its parameter tensor(s)
        #and the keys of the state_dict should match the names of the layers in the model.
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        #model.load_param(checkpoint["model"])
        # for k, v in checkpoint.items():
        #     if k in checkpoint().keys():
        #         checkpoint()[k].copy_(v)
        
        model.load_state_dict(checkpoint["model"],strict=False)
    
    return model

