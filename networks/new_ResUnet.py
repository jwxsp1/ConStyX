from torch import nn
import torch
from networks.resnet import resnet34, resnet18, resnet50, resnet101, resnet152
import torch.nn.functional as F
from networks.unet import UnetBlock


class ResUnet(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False, mixstyle_layers=[],
                 random_type=None, p=0.5):
        super().__init__()
        if resnet == 'resnet34':
            base_model = resnet34
            feature_channels = [64, 64, 128, 256, 512]
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
            feature_channels = [64, 256, 512, 1024, 2048]
        elif resnet == 'resnet101':
            base_model = resnet101
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        self.mixstyle_layers = mixstyle_layers
        self.res = base_model(pretrained=pretrained, mixstyle_layers=mixstyle_layers, random_type=random_type, p=p)

        self.num_classes = num_classes

        self.up1 = UnetBlock(feature_channels[4], feature_channels[3], 256)
        self.up2 = UnetBlock(256, feature_channels[2], 256)
        self.up3 = UnetBlock(256, feature_channels[1], 256)
        self.up4 = UnetBlock(256, feature_channels[0], 256)

        self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.bnout = nn.BatchNorm2d(32)

        self.seg_head = nn.Conv2d(32, self.num_classes, 1)

    def forward(self, input):
        x, sfs,ori_feature = self.res(input)
        mix_feature=sfs[2].detach()
        x = F.relu(x)

        x = self.up1(x, sfs[3])
        x = self.up2(x, sfs[2])
        x = self.up3(x, sfs[1])
        x = self.up4(x, sfs[0])
        x = self.up5(x)
        head_input = F.relu(self.bnout(x))

        seg_output = self.seg_head(head_input)
        d_style=self.calculate_D_style(mix_feature,ori_feature)
        return seg_output,d_style

    def close(self):
        for sf in self.sfs:
            sf.remove()
    
    def calculate_D_style(self,mix_feature,ori_feature):
        b,c,h,w=mix_feature.shape[0],mix_feature.shape[1],mix_feature.shape[2],mix_feature.shape[3]
        mix_feature,ori_feature=mix_feature.view(b,c,h*w),ori_feature.view(b,c,h*w)
        gram_mix_feature,gram_ori_feature=torch.bmm(mix_feature,mix_feature.transpose(1,2)),torch.bmm(ori_feature,ori_feature.transpose(1,2))
        d_style=F.l1_loss(gram_mix_feature,gram_ori_feature,reduction='none')# (b,c,c)
        return d_style.mean(dim=(1,2),keepdim=True).unsqueeze(dim=1)


if __name__ == "__main__":
    model = ResUnet(resnet='resnet34', num_classes=2, pretrained=False, mixstyle_layers=['layer1'], random_type='Random')
    print(model.res)
    model.cuda().eval()
    input = torch.rand(2, 3, 512, 512).cuda()
    seg_output, x_iw_list, iw_loss = model(input)
    print(seg_output.size())

