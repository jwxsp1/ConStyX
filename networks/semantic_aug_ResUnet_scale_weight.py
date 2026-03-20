from torch import nn
import torch
from networks.resnet_aug import resnet34, resnet18, resnet50, resnet101, resnet152
import torch.nn.functional as F
from networks.unet import UnetBlock
from utils.semantic_aug import EstimatorCV,semantic_aug,style_aug
import time

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
        self.res = base_model(pretrained=pretrained, mixstyle_layers=[], random_type=random_type, p=p)

        self.num_classes = num_classes

        self.up1 = UnetBlock(feature_channels[4], feature_channels[3], 256)
        self.up2 = UnetBlock(256, feature_channels[2], 256)
        self.up3 = UnetBlock(256, feature_channels[1], 256)
        self.up4 = UnetBlock(256, feature_channels[0], 256)

        self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.bnout = nn.BatchNorm2d(32)

        self.seg_head = nn.Conv2d(32, self.num_classes, 1)
        self.estimator=EstimatorCV(32,3,device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, input,target_x,grad,aug=False):
        x, sfs = self.res(input)
        x = F.relu(x)

        x = self.up1(x, sfs[3])
        x = self.up2(x, sfs[2])
        x = self.up3(x, sfs[1])
        x = self.up4(x, sfs[0])
        x = self.up5(x)
        head_input = F.relu(self.bnout(x))
        if aug:
            N,A,H,W=head_input.size(0),head_input.size(1),head_input.size(2),head_input.size(3)
            target_x = target_x.view(N, 1, target_x.size(1), target_x.size(2)).float()
            target_x = target_x.long().squeeze()
            target_x_NHW = target_x.contiguous().view(N * H * W)
            features_NHWxA = head_input.permute(0, 2, 3, 1).contiguous().view(N * H * W, A)

            self.estimator.update_CV(features_NHWxA.detach(), target_x_NHW)

            head_input=semantic_aug(head_input,target_x,self.estimator.CoVariance,3)

            head_input=style_aug(grad,head_input,target_x,3)

        seg_output = self.seg_head(head_input)
        pix_features=head_input.detach()
        return seg_output,pix_features

    def close(self):
        for sf in self.sfs:
            sf.remove()


if __name__ == "__main__":
    model = ResUnet(resnet='resnet34', num_classes=2, pretrained=False, mixstyle_layers=['layer1'], random_type='Random')
    print(model.res)
    model.cuda().eval()
    input = torch.rand(2, 3, 512, 512).cuda()
    seg_output, x_iw_list, iw_loss = model(input)
    print(seg_output.size())

