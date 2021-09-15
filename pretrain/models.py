import torch.nn as nn
import torch

class Encoder(nn.Module):
    
    def __init__(self):
        super(Encoder, self).__init__()
        # input shape: (B, 1, T, D)
        self.CNN_BLOCK = nn.ModuleList()
        init_channel = 1
        while (init_channel < 256):
            if init_channel == 16:
                stride = 1
            else: stride = 2
            self.CNN_BLOCK.append(
                nn.Conv2d(in_channels=init_channel, \
                    out_channels=init_channel * 4, \
                    kernel_size=3, padding=1, stride=stride)
            )
            
            if (init_channel <= 16):
                self.CNN_BLOCK.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
                self.CNN_BLOCK.append(nn.BatchNorm2d(init_channel * 4))
                self.CNN_BLOCK.append(nn.ReLU())
            init_channel *= 4
        self.CNN_BLOCK.append(nn.AdaptiveAvgPool2d(1))
        self.model = nn.Sequential(*self.CNN_BLOCK)

    def forward(self, input):
        if (input.shape[1] < 64):
            tmp = torch.zeros(input.shape[0], 64, input.shape[2]).to(input.device)
            tmp[:, :input.shape[1], :] = input
            input = tmp         # padding

        return self.model(input.unsqueeze(1)).flatten(1, 3)





class Decoder_stft(nn.Module):

    def __init__(self, input_dim:int=256):          # outputsize = (80,128)
        super(Decoder_stft, self).__init__()


        self.Linear1 = nn.Linear(input_dim, 64 * 6 * 32)
        # project to (64 * 6 * 8) then reshape to (64, 6, 32)

        self.CNN_BLOCK = nn.ModuleList()
        init_channel = 64
        shrink = 2
        while (init_channel >= 4):
            if (init_channel >= 32): shrink = 2
            else: shrink = 4
            self.CNN_BLOCK.append(nn.ConvTranspose2d(in_channels=init_channel, out_channels=init_channel//shrink, stride=2, padding=1, kernel_size=4))
            if (init_channel > 4):
                self.CNN_BLOCK.append(nn.BatchNorm2d(init_channel//shrink))
                self.CNN_BLOCK.append(nn.ReLU())
            init_channel //= shrink
        self.model = nn.Sequential(*self.CNN_BLOCK)
        # from (4, 48, 256) to (1, 96, 512)

    def forward(self, x):
        projection = self.Linear1(x)
        reshape = projection.reshape(projection.shape[0], 64, 6, 32)

        return self.model(reshape).squeeze(1)     #(B，96, 512)

class Decoder_mel(nn.Module):

    def __init__(self, input_dim:int=256):          # outputsize = (80,128)
        super(Decoder_mel, self).__init__()


        self.Linear1 = nn.Linear(input_dim, 64 * 6 * 8)
        # project to (64 * 6 * 8) then reshape to (64, 6, 8)

        self.CNN_BLOCK = nn.ModuleList()
        init_channel = 64
        shrink = 2
        while (init_channel >= 4):
            if (init_channel >= 32): shrink = 2
            else: shrink = 4
            self.CNN_BLOCK.append(nn.ConvTranspose2d(in_channels=init_channel, out_channels=init_channel//shrink, stride=2, padding=1, kernel_size=4))
            if (init_channel > 4):
                self.CNN_BLOCK.append(nn.BatchNorm2d(init_channel//shrink))
                self.CNN_BLOCK.append(nn.ReLU())
            init_channel //= shrink
        self.model = nn.Sequential(*self.CNN_BLOCK)

    def forward(self, x):
        projection = self.Linear1(x)
        reshape = projection.reshape(projection.shape[0], 64, 6, 8)

        return self.model(reshape).squeeze(1)     #(B，96, 128)

class DEPA(nn.Module):
    def __init__(self, type='stft'):
        super(DEPA, self).__init__()
        decoder = Decoder_stft() if type == 'stft' else Decoder_mel()
        self.model = nn.Sequential(Encoder(), decoder)

    def forward(self, x):
        return self.model(x)

    def extract_embedding(self, input):
        return self.model[0](input)