import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
        )
        self.last_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv(x)
        shortcut = self.shortcut_conv(x)
        res_path = torch.add(x1, shortcut)
        return self.last_activation(res_path)

class Recurrent_block(nn.Module):
    def __init__(self,out_channels,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
        )
    def forward(self,x):
        for i in range(self.t):
            if i==0:
                x1 = self.conv(x)
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,in_channels,out_channels,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(out_channels,t=t),
            Recurrent_block(out_channels,t=t)
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class VanillaUNET(nn.Module):
    def __init__(
            self, in_channels =1, out_channels=1, features=[64,128,256,512]
    ):
        super(VanillaUNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList() #used to store the convolutional layers etc
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self,x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # Add a check just in case the original input was not divisible by 16, so we can recover it with the same size
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:]) 

            concat_skip = torch.cat((skip_connection, x), dim=1) # dimensions in order: batch, channel, height, width. Here it's channel.
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)
    
class AttentionUNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64,128,256,512]
    ):
        super(AttentionUNET, self).__init__()
        self.downs = nn.ModuleList() #used to store the convolutional layers etc
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(AttentionBlock(F_g=feature, F_l=feature, n_coefficients=feature//2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self,x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder (upsampling) path
        for i in range(0, len(self.ups), 3):
            x = self.ups[i](x)  # ConvTranspose2d
            skip = self.ups[i+1](gate=x, skip_connection=skip_connections[i//3])  # AttentionBlock
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i+2](x)  # DoubleConv

        return self.final_conv(x)
    
class RR_UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64,128,256,512]
    ):
        super(RR_UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList() #used to store the convolutional layers etc
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(RRCNN_block(in_channels, feature, t=2))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(RRCNN_block(feature*2, feature, t=2))

        self.bottleneck = RRCNN_block(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self,x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # Add a check just in case the original input was not divisible by 16, so we can recover it with the same size
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:]) 

            concat_skip = torch.cat((skip_connection, x), dim=1) # dimensions in order: batch, channel, height, width. Here it's channel.
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)
    
    
class RR_Attention_UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64,128,256,512]
    ):
        super(RR_Attention_UNET, self).__init__()
        self.downs = nn.ModuleList() #used to store the convolutional layers etc
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(RRCNN_block(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(AttentionBlock(F_g=feature, F_l=feature, n_coefficients=feature//2))
            self.ups.append(RRCNN_block(feature*2, feature))

        self.bottleneck = RRCNN_block(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self,x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder (upsampling) path
        for i in range(0, len(self.ups), 3):
            x = self.ups[i](x)  # ConvTranspose2d
            skip = self.ups[i+1](gate=x, skip_connection=skip_connections[i//3])  # AttentionBlock
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i+2](x)  # DoubleConv

        return self.final_conv(x)
    
    
def test():
    x = torch.randn((3,1,128,128))
    model = RR_Attention_UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()