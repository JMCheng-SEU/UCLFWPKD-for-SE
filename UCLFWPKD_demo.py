import torch
from torch import nn
import torch.nn.functional as F

class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            # nn.BatchNorm2d(mid_channel),
            nn.InstanceNorm2d(mid_channel, affine=True),
            nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
            # nn.BatchNorm2d(out_channel),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU()
        )


        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None


    def forward(self, x, y=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)

        if self.att_conv is not None:
            # upsample residual features
            # y = F.interpolate(y, (shape, w), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * (z[:,0].view(n,1,h,w).contiguous()) + y * (z[:,1].view(n,1,h,w).contiguous()))
        # output
        y = self.conv2(x)

        return y, x



class ABF_Res(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF_Res, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            # nn.BatchNorm2d(mid_channel),
            nn.InstanceNorm2d(mid_channel, affine=True),
            nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
            # nn.BatchNorm2d(out_channel),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU()
        )


        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None


    def forward(self, x, y=None, shape = None):
        n, c, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (h, shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * (z[:,0].view(n,1,h,w).contiguous()) + y * (z[:,1].view(n,1,h,w).contiguous()))
        # output
        y = self.conv2(x)
        return y, x

class ResKD(nn.Module):
    def __init__(
        self, in_channels, out_channels, mid_channel, shapes
    ):
        super(ResKD, self).__init__()
        self.shapes = shapes

        abfs = nn.ModuleList()

        for idx, in_channel in enumerate(in_channels):
            abfs.append(ABF_Res(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))


        self.abfs = abfs[::-1]

    def forward(self, student_features):

        x = student_features[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0])
        results.append(out_features)
        for features, abf, shape in zip(x[1:], self.abfs[1:], self.shapes[1:]):
            out_features, res_features = abf(features, res_features, shape)
            results.insert(0, out_features)

        return results


class ResKD_mid_trans(nn.Module):
    def __init__(
        self, in_channels, out_channels, mid_channel
    ):
        super(ResKD_mid_trans, self).__init__()


        abfs = nn.ModuleList()

        for idx, in_channel in enumerate(in_channels):
            abfs.append(ABF(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))


        self.abfs = abfs[::-1]

    def forward(self, student_features):

        x = student_features
        results = []
        out_features, res_features = self.abfs[0](x)
        results.append(out_features)
        for abf in self.abfs[1:]:
            out_features, res_features = abf(x, res_features)
            results.insert(0, out_features)

        return results

def cosine_pairwise_similarities_perframe(features, eps=1e-6, normalized=True):
    features = features.permute(1, 0, 2)  ##### take timesteps as the first dim
    features_norm = torch.sqrt(torch.sum(features ** 2, dim=2, keepdim=True))
    features = features / (features_norm + eps)
    # features[features != features] = 0
    features_t = features.permute(0, 2, 1)
    similarities = torch.bmm(features, features_t)

    if normalized:
        similarities = (similarities + 1.0) / 2.0
    return similarities

def probabilistic_loss_perframe_withFW(teacher_features, student_features, eps=1e-6):

    tea_avg = torch.mean(teacher_features, dim=2, keepdim=True)
    tea_FW_mask = torch.sigmoid(tea_avg).permute(1,0,2)



    student_s = cosine_pairwise_similarities_perframe(student_features)
    teacher_s = cosine_pairwise_similarities_perframe(teacher_features)

    # Transform them into probabilities
    teacher_s = teacher_s / torch.sum(teacher_s, dim=2, keepdim=True)
    student_s = student_s / torch.sum(student_s, dim=2, keepdim=True)

    loss = (teacher_s - student_s) * (torch.log(teacher_s) - torch.log(student_s)) * tea_FW_mask

    loss = (torch.mean(loss, dim=[1,2])).sum(0)

    return loss

if __name__ == '__main__':


    mid_stu_fea = torch.randn(4, 64, 10, 64) ##### batch channel frame feature

    enc_stu_fea_list = []
    stu_enc_layer_1_out = torch.randn(4, 64, 10, 129)
    enc_stu_fea_list.append(stu_enc_layer_1_out)
    stu_enc_layer_2_out = torch.randn(4, 64, 10, 64)
    enc_stu_fea_list.append(stu_enc_layer_2_out)
    stu_enc_layer_3_out = torch.randn(4, 64, 10, 64)
    enc_stu_fea_list.append(stu_enc_layer_3_out)

    dec_stu_fea_list = []
    stu_dec_layer_1_out = torch.randn(4, 64, 10, 64)
    dec_stu_fea_list.append(stu_dec_layer_1_out)
    stu_dec_layer_2_out = torch.randn(4, 64, 10, 129)
    dec_stu_fea_list.append(stu_dec_layer_2_out)
    stu_dec_layer_3_out = torch.randn(4, 2, 10, 257)
    dec_stu_fea_list.append(stu_dec_layer_3_out)

    mid_tea_fea_list = []
    for index in range(4):
        temp_mid_tea_fea = torch.randn(4, 128, 10, 64)
        mid_tea_fea_list.append(temp_mid_tea_fea)

    enc_tea_fea_list = []
    tea_enc_layer_1_out = torch.randn(4, 128, 10, 129)
    enc_tea_fea_list.append(tea_enc_layer_1_out)
    tea_enc_layer_2_out = torch.randn(4, 128, 10, 64)
    enc_tea_fea_list.append(tea_enc_layer_2_out)
    tea_enc_layer_3_out = torch.randn(4, 128, 10, 64)
    enc_tea_fea_list.append(tea_enc_layer_3_out)

    dec_tea_fea_list = []
    tea_dec_layer_1_out = torch.randn(4, 128, 10, 64)
    dec_tea_fea_list.append(tea_dec_layer_1_out)
    tea_dec_layer_2_out = torch.randn(4, 128, 10, 129)
    dec_tea_fea_list.append(tea_dec_layer_2_out)
    tea_dec_layer_3_out = torch.randn(4, 2, 10, 257)
    dec_tea_fea_list.append(tea_dec_layer_3_out)

    in_channels_mid = [64, 64, 64, 64]

    out_channels_mid = [128, 128, 128, 128]


    mid_channel = 128

    Review_KD_Block_mid = ResKD_mid_trans(in_channels_mid, out_channels_mid, mid_channel)


    in_channels_enc = [64, 64, 64]

    out_channels_enc = [128, 128, 128]

    shapes_enc = [64, 64, 129]

    Review_KD_Block_enc = ResKD(in_channels_enc, out_channels_enc, mid_channel, shapes_enc)

    in_channels_dec = [2, 64, 64]

    out_channels_dec = [2, 128, 128]

    shapes_dec = [64, 129, 257]
    Review_KD_Block_dec = ResKD(in_channels_dec, out_channels_dec, mid_channel, shapes_dec)

    #### KD for Enc
    enc_stu_res_list = Review_KD_Block_enc(enc_stu_fea_list)

    pred_res_enc_loss = 0.0

    for fs, ft in zip(enc_stu_res_list, enc_tea_fea_list):
        BS, CS, T, DS = fs.shape
        BT, CT, T, DT = ft.shape
        ft = ft.detach()

        fs_fea = fs.permute(0, 2, 1, 3)
        ft_fea = ft.permute(0, 2, 1, 3)
        fs_fea = torch.reshape(fs_fea, [BS, T, CS * DS])
        ft_fea = torch.reshape(ft_fea, [BT, T, CT * DT])
        pred_res_enc_loss += probabilistic_loss_perframe_withFW(ft_fea, fs_fea)

    #### KD for Mid
    mid_res_stu_fea_list = Review_KD_Block_mid(mid_stu_fea)
    pred_res_mid_loss = 0.0
    for fs, ft in zip(mid_res_stu_fea_list, mid_tea_fea_list):
        BS, CS, T, DS = fs.shape
        BT, CT, T, DT = ft.shape
        ft = ft.detach()

        fs_fea = fs.permute(0, 2, 1, 3)
        ft_fea = ft.permute(0, 2, 1, 3)
        fs_fea = torch.reshape(fs_fea, [BS, T, CS * DS])
        ft_fea = torch.reshape(ft_fea, [BT, T, CT * DT])
        pred_res_mid_loss += probabilistic_loss_perframe_withFW(ft_fea, fs_fea)


    ###### KD for Dec

    dec_stu_fea_list = dec_stu_fea_list[::-1]

    dec_stu_res_list = Review_KD_Block_dec(dec_stu_fea_list)

    dec_tea_fea_list = dec_tea_fea_list[::-1]
    pred_res_dec_loss = 0.0

    cnt = 0

    for fs, ft in zip(dec_stu_res_list, dec_tea_fea_list):

        BS, CS, T, DS = fs.shape
        BT, CT, T, DT = ft.shape
        ft = ft.detach()

        fs_fea = fs.permute(0, 2, 1, 3)
        ft_fea = ft.permute(0, 2, 1, 3)
        fs_fea = torch.reshape(fs_fea, [BS, T, CS * DS])
        ft_fea = torch.reshape(ft_fea, [BT, T, CT * DT])

        pred_res_dec_loss += probabilistic_loss_perframe_withFW(ft_fea, fs_fea)

    KD_loss = pred_res_mid_loss + pred_res_enc_loss + pred_res_dec_loss

    print(KD_loss)


