from torch.utils.data import Dataset
from albumentations import Compose, Resize, Normalize, HorizontalFlip, VerticalFlip, Rotate, CenterCrop
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import math
import timm
import torch
import torch.nn as nn
from torch.nn import functional as F, Parameter
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


class ShopeeNet(nn.Module):
    def __init__(self,
                 n_classes,
                 model_name='efficientnet_b0',
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0,
                 loss_module='softmax',
                 s=30.0,
                 margin=0.50,
                 ls_eps=0.0,
                 theta_zero=0.785,
                 pretrained=True):
        """
        :param model_name: name of model from pretrainedmodels e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super(ShopeeNet, self).__init__()
        print(f'Building Model Backbone for {model_name} model')

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        final_in_features = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        self.loss_module = loss_module
        if loss_module == 'arcface':
            self.final = ArcMarginProduct(final_in_features, n_classes,
                                          s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
        elif loss_module == 'cosface':
            self.final = AddMarginProduct(final_in_features, n_classes, s=s, m=margin)
        elif loss_module == 'adacos':
            self.final = AdaCos(final_in_features, n_classes, m=margin, theta_zero=theta_zero)
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
        feature = self.extract_feat(x)
        if self.loss_module in ('arcface', 'cosface', 'adacos'):
            logits = self.final(feature, label)
        else:
            logits = self.final(feature)
        return logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)

        return x


class ShopeeDataset(Dataset):
    def __init__(self, csv, train):
        self.csv = csv.reset_index()
        self.train = train
        self.transform = Compose([VerticalFlip(p=0.5),
                                  HorizontalFlip(p=0.5),
                                  Resize(512, 512),
                                  Normalize(),
                                  ])

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        image = cv2.imread(self.csv["filepath"][index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_transf = self.transform(image=image)["image"].astype(np.float32)
        image_transf = torch.tensor(image_transf.transpose(2, 0, 1))

        # Return dataset info
        if self.train == True:
            label_group = torch.tensor(self.csv["label_group"][index])
            return image_transf, label_group

        else:
            return image_transf


class AdaCos(nn.Module):
    def __init__(self, in_features, out_features, m=0.50, ls_eps=0, theta_zero=math.pi / 4):
        super(AdaCos, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.theta_zero = theta_zero
        self.s = math.log(out_features - 1) / math.cos(theta_zero)
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input_dim, label):
        # normalize features
        x = F.normalize(input_dim)
        # normalize weights
        w = F.normalize(self.weight)
        # dot product
        logits = F.linear(x, w)
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        with torch.no_grad():
            b_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            b_avg = torch.sum(b_avg) / input_dim.size(0)
            theta_med = torch.median(theta)
            self.s = torch.log(b_avg) / torch.cos(torch.min(self.theta_zero * torch.ones_like(theta_med), theta_med))
        output *= self.s

        return output


class AddMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input_dim, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input_dim), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class ArcMarginProduct(nn.Module):
    def __init__(self, in_feature=128, out_feature=10575, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        """
        :param in_feature:
        :param out_feature:
        :param s:
        :param m:
        :param easy_margin:
        """
        super(ArcMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output


def edit_title(text):
    title_with_return = ""
    for i, ch in enumerate(text):
        title_with_return += ch
        if (i != 0) & (i % 20 == 0):
            title_with_return += '\n'
    return title_with_return


def display_df(df, path, cols=6, rows=4):
    cols = max(min(cols, df.shape[0]) - 1, df.shape[0])
    rows = max(int(df.shape[0] / rows), 1)
    for k in range(rows):
        plt.figure(figsize=(20, 5))
        for j in range(cols):
            row = cols * k + j

            name = df['filepath'].tolist()[row]
            title = df['title'].tolist()[row]
            title_with_return = edit_title(title)

            img = cv2.imread(str(path / name))
            plt.subplot(1, cols, j + 1)
            plt.title(title_with_return)
            plt.axis('off')
            plt.imshow(img)
        plt.show()


def f1_score_cal(target, predict):
    intersection = len(np.intersect1d(target, predict))
    return 2 * intersection / (len(target) + len(predict))


def plot_image(df):
    img_path = df['filepath'].item()
    title = df['title'].item()
    title_with_return = edit_title(title)

    plt.figure(figsize=(20, 5))
    plt.title(title_with_return)
    img = cv2.imread(img_path)
    plt.imshow(img)
    plt.show()
