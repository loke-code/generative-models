import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

'''
    Following is an imitation of the official EnlightenGAN Implementation 
    https://github.com/VITA-Group/EnlightenGAN
'''


def compute_attention_map(image):
    normalized_img = (image + 1.0) / 2.0
    I = torch.max(normalized_img, dim=1, keepdim=True)[0]
    return 1.0 - I

def pad_tensor(input_tensor):
    height_org, width_org = input_tensor.shape[2], input_tensor.shape[3]
    divide = 16 

    if height_org % divide != 0 or width_org % divide != 0:
        width_res = width_org % divide
        height_res = height_org % divide
        
        pad_left = int((divide - width_res) / 2) if width_res != 0 else 0
        pad_right = int((divide - width_res) - pad_left) if width_res != 0 else 0
        
        pad_top = int((divide - height_res) / 2) if height_res != 0 else 0
        pad_bottom = int((divide - height_res) - pad_top) if height_res != 0 else 0
        
        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input_tensor = padding(input_tensor)
    else:
        pad_left = pad_right = pad_top = pad_bottom = 0
        
    return input_tensor, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input_tensor, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input_tensor.shape[2], input_tensor.shape[3]
    return input_tensor[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]

# =======================
# EnlightenGAN Generator
# =======================
class EnlightenGenerator(nn.Module):
    def __init__(self, norm_type='instance', skip_connection=True):
        """
        norm_type: 'batch' (original code) or 'instance' (ours for small batches)
        """
        super(EnlightenGenerator, self).__init__()
        self.skip_connection = skip_connection
        p = 1
        
        def get_norm(channels):
            if norm_type == 'instance':
                return nn.InstanceNorm2d(channels, affine=False)
            elif norm_type == 'batch':
                return nn.BatchNorm2d(channels)
            return nn.Identity()
        
        # Self-Attention Map Downsampling
        self.downsample_1 = nn.MaxPool2d(2)
        self.downsample_2 = nn.MaxPool2d(2)
        self.downsample_3 = nn.MaxPool2d(2)
        self.downsample_4 = nn.MaxPool2d(2)

        # Encoder
        self.conv1_1 = nn.Conv2d(4, 32, 3, padding=p) 
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_1 = get_norm(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_2 = get_norm(32)
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_1 = get_norm(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_2 = get_norm(64)
        self.max_pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_1 = get_norm(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_2 = get_norm(128)
        self.max_pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_1 = get_norm(256)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_2 = get_norm(256)
        self.max_pool4 = nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=p)
        self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn5_1 = get_norm(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn5_2 = get_norm(512)

        # Decoder
        self.upconv5 = nn.Conv2d(512, 256, 3, padding=p)
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=p)
        self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn6_1 = get_norm(256)
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn6_2 = get_norm(256)

        self.upconv6 = nn.Conv2d(256, 128, 3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_1 = get_norm(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_2 = get_norm(128)

        self.upconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_1 = get_norm(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_2 = get_norm(64)

        self.upconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn9_1 = get_norm(32)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv10 = nn.Conv2d(32, 3, 1)

    def forward(self, input_img, gray_map=None):
        if gray_map is None:
            gray_map = compute_attention_map(input_img)

        input_img, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input_img)
        gray_map, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(gray_map)

        gray_2 = self.downsample_1(gray_map)
        gray_3 = self.downsample_2(gray_2)
        gray_4 = self.downsample_3(gray_3)
        gray_5 = self.downsample_4(gray_4)

        x = self.bn1_1(self.LReLU1_1(self.conv1_1(torch.cat((input_img, gray_map), 1))))
        conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
        x = self.max_pool1(conv1)

        x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
        conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
        x = self.max_pool2(conv2)

        x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
        conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
        x = self.max_pool3(conv3)

        x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))
        conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))
        x = self.max_pool4(conv4)

        x = self.bn5_1(self.LReLU5_1(self.conv5_1(x)))
        x = x * gray_5  
        conv5 = self.bn5_2(self.LReLU5_2(self.conv5_2(x)))

        conv5_up = F.interpolate(conv5, scale_factor=2, mode='bilinear', align_corners=False)
        conv4 = conv4 * gray_4  
        up6 = torch.cat([self.upconv5(conv5_up), conv4], 1)
        x = self.bn6_1(self.LReLU6_1(self.conv6_1(up6)))
        conv6 = self.bn6_2(self.LReLU6_2(self.conv6_2(x)))

        conv6_up = F.interpolate(conv6, scale_factor=2, mode='bilinear', align_corners=False)
        conv3 = conv3 * gray_3
        up7 = torch.cat([self.upconv6(conv6_up), conv3], 1)
        x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))
        conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))

        conv7_up = F.interpolate(conv7, scale_factor=2, mode='bilinear', align_corners=False)
        conv2 = conv2 * gray_2
        up8 = torch.cat([self.upconv7(conv7_up), conv2], 1)
        x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
        conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))

        conv8_up = F.interpolate(conv8, scale_factor=2, mode='bilinear', align_corners=False)
        conv1 = conv1 * gray_map
        up9 = torch.cat([self.upconv8(conv8_up), conv1], 1)
        x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
        conv9 = self.LReLU9_2(self.conv9_2(x))

        latent = self.conv10(conv9)

        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        input_img = pad_tensor_back(input_img, pad_left, pad_right, pad_top, pad_bottom)

        if self.skip_connection:
            output = torch.tanh(latent + input_img)
        else:
            output = torch.tanh(latent)

        return output
    
    
# ----------------------- 
# Discriminator PatchGAN
# -----------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_norm=False):
        super(PatchDiscriminator, self).__init__()
        kw = 4
        padw = int(np.ceil((kw-1)/2)) 
        
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw)
            ]
            if use_norm:
                sequence += [nn.BatchNorm2d(ndf * nf_mult)]
            sequence += [nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw)
        ]
        if use_norm:
            sequence += [nn.BatchNorm2d(ndf * nf_mult)]
        sequence += [nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
