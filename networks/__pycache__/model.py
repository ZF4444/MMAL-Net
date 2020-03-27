import torch
from torch import nn
from torchvision.ops import roi_pool
from torch.autograd import Variable
import torch.nn.functional as F
from networks import resnet
from config import pretrain_path, coordinates_cat, window_side, stride, iou_threshs, topN, window_nums, window_nums_sum, ratios, cat_nums,window_milestones, window_nums_sum1, ratios1, cat_nums1
import numpy as np
from random import randint
from time import time
from utils.object_module import select_aggregate
from random import randint
from torchvision import transforms


def nms(scores_np, proposalN, iou_threshs, coordinates):
    if not (type(scores_np).__module__ == 'numpy' and len(scores_np.shape) == 2 and scores_np.shape[1] == 1):
        raise TypeError('score_np is not right')

    windows_num = scores_np.shape[0]
    indices_coordinates = np.concatenate((scores_np, coordinates), 1)      #[339,5]

    indices = np.argsort(indices_coordinates[:, 0])           #(339, )   根据元素有小到大返回他们的位置
    indices_coordinates = np.concatenate((indices_coordinates, np.arange(0,windows_num).reshape(windows_num,1)), 1)[indices]                  #[339,6]
    indices_results = []

    res = indices_coordinates

    while res.any():
        indice_coordinates = res[-1]         #[6,]
        indices_results.append(indice_coordinates[5])

        if len(indices_results) == proposalN:
            return np.array(indices_results).reshape(1,proposalN).astype(np.int)
        res = res[:-1]          #[338,6]

        # 排除与选中的anchor box iou大于阈值的anchor boxes
        start_max = np.maximum(res[:, 1:3], indice_coordinates[1:3])    #[338,2]
        end_min = np.minimum(res[:, 3:5], indice_coordinates[3:5])     #[338,2]
        lengths = end_min - start_max + 1                   #[338,2]
        intersec_map = lengths[:, 0] * lengths[:, 1]    #[338,]
        intersec_map[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
        iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1] + 1) * (res[:, 4] - res[:, 2] + 1) +
                                      (indice_coordinates[3] - indice_coordinates[1] + 1) *
                                      (indice_coordinates[4] - indice_coordinates[2] + 1) - intersec_map)   #(338,)

        # indices_res = res[:,5]
        # res = res[((indices_res < window1_num)&(iou_map_cur <= iou_thresh1))|((indices_res >= window1_num)&(iou_map_cur <= iou_thresh2))]

        res = res[iou_map_cur <= iou_threshs]

    if len(indices_results) < topN:
        print('less than topN,is %d' % len(indices_results))
    while len(indices_results) != proposalN:
        indices_results.append(indice_coordinates[5])

    return np.array(indices_results).reshape(1, -1).astype(np.int)

def KD_loss(student_outputs, teacher_outputs, alpha=2, T=10):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """

    outputs_S = F.log_softmax(student_outputs / T, dim=1)
    outputs_T = F.softmax(teacher_outputs / T, dim=1)

    KD_loss = alpha * nn.KLDivLoss()(outputs_S, outputs_T)


    # Ps = F.softmax(student_outputs/T)
    # Pt = F.softmax(teacher_outputs/T)
    # KD_loss =
    #
    # KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
    #                          F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
    #           F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss

class Score_net(nn.Module):
    def __init__(self):
        super(Score_net, self).__init__()
        self.avgpools = [nn.AvgPool2d(ratios[i], 1) for i in range(len(ratios))]

    def forward(self, proposalN, x, ratios, window_nums_sum, cat_nums):
        batch, channels, _, _ = x.size()
        # avg1 = self.avgpool_2x2(x)
        avgs = [self.avgpools[i](x) for i in range(len(ratios))]

        # feature map sum
        fm_sum = [torch.sum(avgs[i], dim=1) for i in range(len(ratios))]  # N*14*14

        all_scores = torch.cat([fm_sum[i].view(batch, -1, 1) for i in range(len(ratios))], dim=1)
        windows_scores_np = all_scores.data.cpu().numpy()  # N*339*1
        window_scores = torch.from_numpy(windows_scores_np).cuda().reshape(batch, -1)
        # nms
        proposalN_indices = []


        for i, scores in enumerate(windows_scores_np):
            indices_results = []
            for j  in range(len(window_nums_sum)-1):
                indices_results.append(nms(scores[sum(window_nums_sum[:j+1]):sum(window_nums_sum[:j+2])], proposalN=cat_nums[j], iou_threshs=iou_threshs[j],
                                           coordinates=coordinates_cat[sum(window_nums_sum[:j+1]):sum(window_nums_sum[:j+2])]) + sum(window_nums_sum[:j+1]))
            # indices_results.reverse()
            proposalN_indices.append(np.concatenate(indices_results, 1))   # reverse


        proposalN_indices = np.array(proposalN_indices).reshape(batch, proposalN)
        proposalN_indices = torch.from_numpy(proposalN_indices).cuda()       #N*proposalN
        proposalN_windows_scores = torch.cat(
            [torch.index_select(all_score, dim=0, index=proposalN_indices[i]) for i, all_score in enumerate(all_scores)], 0).reshape(
            batch, proposalN)  # N*proposalN

        return proposalN_indices, proposalN_windows_scores, window_scores

# class Score_net(nn.Module):
#     def __init__(self, proposalN):
#         super(Score_net, self).__init__()
#         self.proposalN = proposalN
#         self.avgpools = [nn.AvgPool2d(int(window_side[i]/stride), 1) for i in range(len(window_side))]
#
#         # self.avgpool_8x8 = nn.AvgPool2d(8, 1)
#         self.down1 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1)
#         self.down2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1)
#         self.down3 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1)
#         self.relu = nn.ReLU()
#         # in_channels, out_channels, kernel_size, stride, padding
#         self.down4 = nn.Conv2d(128, 1, 1, 1, 0)
#
#     def forward(self, x):
#         batch, channels, _, _ = x.size()
#         # avg1 = self.avgpool_2x2(x)
#         avgs = [self.avgpools[i](x) for i in range(len(window_side))]
#         d1 = [self.relu(self.down1(avgs[i])) for i in range(len(window_side))]  # N*128*11*11
#         d2 = [self.relu(self.down2(d1[i])) for i in range(len(window_side))]  # N*128*7*7
#         d3 = [self.relu(self.down3(d2[i])) for i in range(len(window_side))]  # N*128*7*7
#         scores = [self.down4(d3[i]).view(batch, -1) for i in range(len(window_side))]
#
#         all_scores = torch.cat([scores[i].view(batch, -1, 1) for i in range(len(window_side))], dim=1)   # N*339*1
#         windows_scores_np = all_scores.data.cpu().numpy()  # N*339*1
#         window_scores = torch.from_numpy(windows_scores_np).cuda().reshape(batch, -1)
#         # nms
#         proposalN_indices = []
#
#         for x in windows_scores_np:
#             indices_results = nms(x,proposalN=self.proposalN, iou_threshs=iou_threshs)
#             proposalN_indices.append(indices_results)
#
#         proposalN_indices = np.array(proposalN_indices).reshape(batch, self.proposalN)
#
#         proposalN_indices = torch.from_numpy(proposalN_indices).cuda()       #N*proposalN
#         proposalN_windows_scores = torch.cat(
#             [torch.index_select(all_score, dim=0, index=proposalN_indices[i]) for i, all_score in enumerate(all_scores)], 0).reshape(
#             batch, self.proposalN)  # N*proposalN
#
#         return proposalN_indices, proposalN_windows_scores, window_scores

# class Score_net(nn.Module):
#     def __init__(self, proposalN):
#         super(Score_net, self).__init__()
#         self.proposalN = proposalN
#         self.down1 = nn.Conv2d(2048, 128, 2, 1)
#         self.down2 = nn.Conv2d(128, 128, 3, 1)
#         self.down3 = nn.Conv2d(128, 128, 3, 1)
#         self.down4 = nn.Conv2d(128, 128, 3, 1)
#         self.down5 = nn.Conv2d(128, 128, 3, 1)
#         self.ReLU = nn.ReLU()
#         self.tidy1 = nn.Conv2d(128, 1, 1, 1)  # 6 = 两种尺度*三种比例
#         self.tidy2 = nn.Conv2d(128, 1, 1, 1)
#         self.tidy3 = nn.Conv2d(128, 1, 1, 1)  # 6 = 两种尺度*三种比例
#         self.tidy4 = nn.Conv2d(128, 1, 1, 1)
#         self.tidy5 = nn.Conv2d(128, 1, 1, 1)  # 6 = 两种尺度*三种比例
#
#     def forward(self, x):
#         batch, channels, _, _ = x.size()
#         d1 = self.ReLU(self.down1(x))  # N*2048*14*14-->N*128*14*14
#         d2 = self.ReLU(self.down2(d1))  # N*128*14*14-->N*128*7*7
#         d3 = self.ReLU(self.down3(d2))  # N*2048*14*14-->N*128*14*14
#         d4 = self.ReLU(self.down4(d3))  # N*128*14*14-->N*128*7*7
#         d5 = self.ReLU(self.down5(d4))  # N*2048*14*14-->N*128*14*14
#
#         t1 = self.tidy1(d1).view(batch, -1)  # N*6*14*14-->N*1176
#         t2 = self.tidy2(d2).view(batch, -1)  # N*6*7*7-->N*294
#         t3 = self.tidy3(d3).view(batch, -1)  # N*6*14*14-->N*1176
#         t4 = self.tidy4(d4).view(batch, -1)  # N*6*7*7-->N*294
#         t5 = self.tidy5(d5).view(batch, -1)  # N*6*14*14-->N*1176
#
#
#
#         all_scores = torch.cat([t1.view(batch, -1, 1), t2.view(batch, -1, 1), t3.view(batch, -1, 1), t4.view(batch, -1, 1), t5.view(batch, -1, 1)], dim=1)   # N*339*1
#         windows_scores_np = all_scores.data.cpu().numpy()  # N*339*1
#         window_scores = torch.from_numpy(windows_scores_np).cuda().reshape(batch, -1)
#         # nms
#         proposalN_indices = []
#
#         for x in windows_scores_np:
#             indices_results = nms(x,proposalN=self.proposalN, iou_threshs=iou_threshs)
#             proposalN_indices.append(indices_results)
#
#         proposalN_indices = np.array(proposalN_indices).reshape(batch, self.proposalN)
#
#         proposalN_indices = torch.from_numpy(proposalN_indices).cuda()       #N*proposalN
#         proposalN_windows_scores = torch.cat(
#             [torch.index_select(all_score, dim=0, index=proposalN_indices[i]) for i, all_score in enumerate(all_scores)], 0).reshape(
#             batch, self.proposalN)  # N*proposalN
#
#         return proposalN_indices, proposalN_windows_scores, window_scores


# class Class_net(nn.Module):
#     def __init__(self, num_classes, topN, proposalN):
#         super(Class_net, self).__init__()
#         self.num_classes = num_classes
#         self.topN = topN
#         self.proposalN = proposalN
#         self.maxpools = [nn.MaxPool2d(int(window_side[i]/stride), 1) for i in range(len(window_side))]
#         self.avgpools = [nn.AvgPool2d(ratios[i], 1) for i in range(len(ratios))]
#         self.dropout = nn.Dropout(p=0.5)
#         # self.fc0 = nn.Linear(2048, num_classes)
#         # self.fc1 = nn.Linear(2048, num_classes)
#         # self.fc2 = nn.Linear(2048, num_classes)
#         # self.fc3 = nn.Linear(2048, num_classes)
#         # self.fc4 = nn.Linear(2048, num_classes)
#         # self.fc5 = nn.Linear(2048, num_classes)
#         # self.fc6 = nn.Linear(2048, num_classes)
#         # self.fc7 = nn.Linear(2048, num_classes)
#         self.windowscls_net = nn.Linear(2048, num_classes)
#
#     def forward(self, x, proposalN_indices):
#         batch, channels, _, _ = x.size()
#         # fcs = [self.fc0,self.fc1,self.fc2,self.fc3,self.fc4,self.fc5, self.fc6,self.fc7]
#
#         features = [self.avgpools[i](x).view(batch, channels, -1) for i in range(len(ratios))]
#
#         features_cat = torch.cat(features, dim=2)
#
#         proposalN_windows_feature = torch.cat(
#             [torch.index_select(feature_cat, dim=1, index=proposalN_indices[i]).transpose(0, 1) for i, feature_cat in
#              enumerate(
#                  features_cat)], 0).view(batch * self.proposalN, 2048)  # (N*proposalN)*2048
#
#         proposalN_windows_feature = self.dropout(proposalN_windows_feature)
#         proposalN_windows_logits = self.windowscls_net(proposalN_windows_feature)
#         # fc_indexs = torch.cat([torch.tensor([i]).unsqueeze(0).repeat(window_nums[i], 1) for i in range(len(ratios))],
#         #                          dim=0).cuda()
#         # proposalN_windows_logits = []
#         # for i in range(batch):
#         #     for j in range(self.proposalN):
#         #         proposalN_windows_logits.append(fcs[fc_indexs[proposalN_indices[i, j]][0]](proposalN_windows_feature[i, j].unsqueeze(0)))
#         # proposalN_windows_logits = torch.cat(proposalN_windows_logits, dim=0) # (N*proposalN)*200
#
#         # # ROIP
#         # x_lefttop = (coordinates_cat[:, 0] + 1) // 32
#         # y_lefttop = (coordinates_cat[:, 1] + 1) // 32
#         # j = 0
#         # coordinates = []
#         # for i in range(len(coordinates_cat)):
#         #     if i >= window_milestones[0]:
#         #         j += 1
#         #         window_milestones.pop(0)
#         #     coordinates.append(
#         #         [x_lefttop[i], y_lefttop[i], x_lefttop[i] + ratios[j][0] - 1, y_lefttop[i] + ratios[j][1] - 1])
#         # coordinates = torch.from_numpy(np.array(coordinates)).cuda()
#         # batch_indexs = torch.cat([torch.tensor(range(batch))[i].unsqueeze(0).repeat(6, 1) for i in range(batch)],
#         #                          dim=0).cuda()
#         # boxes = torch.cat(
#         #     [torch.index_select(coordinates, dim=0, index=proposalN_indices[i]) for i in range(batch)],
#         #     0)  # (N*proposalN)*2048
#         # boxes = torch.cat((batch_indexs, boxes), dim=1).float()
#         # roip_features = roi_pool(input=torch.cat([x[i].unsqueeze(0).repeat(6, 1, 1, 1) for i in range(batch)], dim=0),
#         #                          boxes=boxes, output_size=[3, 3], spatial_scale=1.0)
#
#         return proposalN_windows_feature.view(batch, self.proposalN, 2048), proposalN_windows_logits
#
# class Windows_net(nn.Module):
#     def __init__(self, num_classes, topN, proposalN):
#         super(Windows_net, self).__init__()
#         self.score_net = Score_net(proposalN)
#         # self.class_net = Class_net(num_classes, topN, proposalN)
#
#     def forward(self, x):
#         proposalN_indices, proposalN_windows_scores, window_scores = self.score_net(x.detach())
#         # proposalN_windows_feature, proposalN_windows_logits = self.class_net(x, proposalN_indices)
#
#         # return  proposalN_windows_scores, proposalN_indices, proposalN_windows_feature, proposalN_windows_logits,\
#         #         window_scores
#
#         return  proposalN_windows_scores, proposalN_indices,\
#                 window_scores

# class BiLSTM(nn.Module):
#     def __init__(self, input_size=2048, hidden_size=256, num_layers=2, num_classes=200):
#         super(BiLSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection
#
#     def forward(self, x):
#         # Set initial states
#         h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).cuda()  # 2 for bidirection
#         c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).cuda()
#
#         # Forward propagate LSTM
#         # input(seq_len, batch, input_size)
#         # h0(num_layers * num_directions, batch, hidden_size)
#         # c0(num_layers * num_directions, batch, hidden_size)
#         output, (hn, cn) = self.lstm(x, (h0, c0))  # shape (batch, seq_len, num_directions * hidden_size)
#         #output: last lstm hidden state, shape (seq_len, batch, num_directions * hidden_size)
#         #hn(num_layers * num_directions, batch, hidden_size)
#         #cn(num_layers * num_directions, batch, hidden_size)
#
#         windows_logits = self.fc(output)   # (batch, seq_len, num_directions * hidden_size)
#
#         return windows_logits

# class BiLSTM(nn.Module):
#     def __init__(self, input_size=2048, hidden_size=2048, num_layers=1, num_classes=200):
#         super(BiLSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, dropout=0, batch_first=True, bidirectional=False)
#         self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=0, batch_first=True, bidirectional=False)
#         self.maxpool = nn.MaxPool1d(topN+1)
#         self.fc = nn.Linear(hidden_size*(len(window_side)*topN+1), num_classes)  # 2 for bidirection
#         self.fc1 = nn.Linear(hidden_size, num_classes)  # 2 for bidirection
#
#     def forward(self, x):
#         # Set initial states
#         h0_0 = torch.zeros(self.num_layers * 1, x.size(0), self.hidden_size).cuda()  # 2 for bidirection
#         c0_0 = torch.zeros(self.num_layers * 1, x.size(0), self.hidden_size).cuda()
#
#         h0_1 = torch.zeros(self.num_layers * 1, x.size(0), self.hidden_size).cuda()  # 2 for bidirection
#         c0_1 = torch.zeros(self.num_layers * 1, x.size(0), self.hidden_size).cuda()
#
#         # Forward propagate LSTM
#         # input(seq_len, batch, input_size)
#         # h0(num_layers * num_directions, batch, hidden_size)
#         # c0(num_layers * num_directions, batch, hidden_size)
#
#         # 反向输入x
#         # create inverted indices
#         # idx = [i for i in range(x.size(1) - 1, -1, -1)]
#         # idx = torch.LongTensor(idx).cuda()
#         # x = x.index_select(1, idx)
#
#         output1, _ = self.lstm1(x, (h0_0, c0_0))  # shape (batch, seq_len, num_directions * hidden_size)
#         # reverse output1
#         index = np.ascontiguousarray(np.arange(0, output1.size(1))[::-1])
#         index = torch.from_numpy(index).cuda()
#         ht1 = torch.index_select(output1, 1, index)
#         output2, _ = self.lstm2(ht1, (h0_1, c0_1))  # shape (batch, seq_len, num_directions * hidden_size)
#
#         lstm_window_logits = self.fc1(output2[:,-1,:])
#
#         #output: last lstm hidden state, shape (seq_len, batch, num_directions * hidden_size)
#         #hn(num_layers * num_directions, batch, hidden_size)
#         #cn(num_layers * num_directions, batch, hidden_size)
#
#         #lstm_windows_logits = torch.cat((self.fc1(output1[:, -5]).unsqueeze(1), self.fc2(output1[:, -4]).unsqueeze(1),
#           #                          self.fc3(output1[:,-3]).unsqueeze(1), self.fc4(output1[:, -2]).unsqueeze(1),
#          #                           self.fc5(output1[:, -1]).unsqueeze(1)), dim=1)
#
#         # lstm_windows_logits = self.fc1(output1)
#
#         # hn_logits = self.fc1(hn.squeeze(0).unsqueeze(1))
#
#        # maxpool = torch.max(output1, dim=1, keepdim=True).values
#        # windows_logits = self.fc1(maxpool)   # (batch, seq_len, num_directions * hidden_size)
#         #
#         # lstm_concat_logits = self.fc(output1.reshape(output1.size(0), -1)).unsqueeze(1)
#
#         # tmp = self.fc(output1.reshape(output1.size(0), -1))
#         # lstm_concat_logits = self.fc1(tmp).unsqueeze(1)
#
#         # return windows_logits, concat_logits
#
#         return lstm_window_logits

class MainNet(nn.Module):
    def __init__(self,topN, proposalN, num_classes, channels):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(MainNet, self).__init__()
        self.num_classes = num_classes
        self.proposalN = proposalN
        self.topN = topN
        self.pretrained_model = resnet.resnet50(pretrained=True, pth_path=pretrain_path)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpools = [nn.AvgPool2d(int(window_side[i] / stride), 1) for i in range(len(window_side))]
        self.rawcls_net = nn.Linear(channels, num_classes)
        # self.windows_net = Windows_net(num_classes, topN, proposalN)
        self.score_net = Score_net()
        self.windowcls_net = nn.Linear(channels, num_classes)
        # self.catcls_net = nn.Linear(2048*6, num_classes)


    def forward(self, x, labels, batch_idx, coor_1epoch, status='test'):
        ## window drop
        # if status == 'train':
        #     with torch.no_grad():
        #         fm, embedding, conv5_b = self.pretrained_model(x.detach())            # N*2048*14*14
        #         batch_size, channel_size, side_size, _ = fm.shape
        #         assert channel_size == 2048
        #         proposalN_indices, _, _ = self.score_net(4, fm.detach(), ratios1, window_nums_sum1, cat_nums1)
        #         for j in range(batch_size):
        #             [x0, y0, x1, y1] = coordinates_cat[proposalN_indices[j, randint(0, 3)]]
        #             x[j:j + 1, :, x0:x1 + 1, y0: y1 + 1] = 0

        fm, embedding, conv5_b = self.pretrained_model(x)  # N*2048*14*14
        batch_size, channel_size, side_size, _ = fm.shape
        assert channel_size == 2048

        # raw branch
        raw_logits = self.rawcls_net(embedding)

        fm_detach = fm.detach()
        fm_detach = fm_detach.reshape(batch_size, channel_size, -1)
        conv5_b_detach = conv5_b.detach()
        conv5_b_detach = conv5_b_detach.reshape(batch_size, channel_size, -1)
        preds = torch.max(raw_logits, dim=1)[1]
        if status == 'train':
            target_weight = torch.index_select(self.rawcls_net.weight.detach(), dim=0, index=labels)
        else:
            target_weight = torch.index_select(self.rawcls_net.weight.detach(), dim=0, index=preds)
        cams = torch.cat([torch.mm(target_weight[i].reshape(1, -1), fm_detach[i]).reshape(1, 1, side_size, side_size) for i in
                range(batch_size)], dim=0)
        cams1 = torch.cat([torch.mm(target_weight[i].reshape(1, -1), conv5_b_detach[i]).reshape(1, 1, side_size, side_size) for i in
                range(batch_size)], dim=0)

        # location
        # if len(coor_1epoch) == 0 or status == 'train':
        #     # SCDA
        #     coordinates = torch.tensor(select_aggregate(fm.detach(), conv5_b.detach())).cuda()
        # else:
        #     coordinates = torch.tensor(coor_1epoch[batch_idx]).cuda()
        #SCDA
        coordinates = torch.tensor(select_aggregate(cams.detach(), cams1.detach())).cuda()

        local_imgs = torch.zeros([batch_size, 3, 448, 448]).cuda()  # [N, 3, 448, 448]
        for i in range(batch_size):
            [x0, y0, x1, y1] = coordinates[i]
            # [lx0, ly0, lx1, ly1] = torch.round((x0.float()+1)/32).int(), torch.round((y0.float() + 1)/32).int(), torch.round((x1.float()+1)/32).int(), torch.round((y1.float() + 1)/32).int()
            # _, proposalN_indices, _ = self.windows_net(F.interpolate(fm[i:i+1, :, lx0:lx1, ly0:ly1], size=(14, 14), mode='bilinear',
            #                                       align_corners=True))
            # [lx0, ly0, lx1, ly1] = coordinates_cat[proposalN_indices[0, randint(0, self.proposalN - 1)]]
            # [x_0, y_0, x_1, y_1] = lx0 / 447, ly0 / 447, lx1 / 447, ly1 / 447
            # [ax0, ay0, ax1, ay1] = x_0 * (x1-x0).float(), y_0 * (y1-y0).float(), x_1 * (x1-x0).float(), y_1 * (y1-y0).float()
            # x_clone = x[i:i+1].clone().detach()
            # x_clone[0:0+1, :, (x0+torch.round(ax0).int()):(x1+torch.round(ax1).int() + 1), (y0+torch.round(ay0).int()):(y1+torch.round(ay1).int() + 1)] = 0
            local_imgs[i:i + 1] = F.interpolate(x[i:i + 1, :, x0:(x1+1), y0:(y1+1)], size=(448, 448), mode='bilinear',
                                                  align_corners=True)  # [N, 3, 224, 224]
            # if status == 'train':
            #     if randint(0,1) == 0:
            #         local_imgs[i:i + 1] = torch.flip(local_imgs[i:i + 1], dims=[3])
        # # window drop
        # if status == 'train':
        #     with torch.no_grad():
        #         self.pretrained_model.train()
        #         local_fm, _, _ = self.pretrained_model(local_imgs.detach())
        #         proposalN_indices, _, _ = self.score_net(4, local_fm.detach(), ratios1, window_nums_sum1, cat_nums1)
        #         for j in range(batch_size):
        #             [x0, y0, x1, y1] = coordinates_cat[proposalN_indices[j, randint(0,3)]]
        #             local_imgs[j:j+1,:, x0:x1+1, y0: y1+1] = 0
        local_fm, local_embeddings, _ = self.pretrained_model(local_imgs.detach())  # [N, 2048]
        local_logits = self.rawcls_net(local_embeddings)  # [N, 200]

        # windows branch
        #N*proposalN*2048                                      x[0]                      N*proposalN        N*(2048*proposalN)
        # proposalN_windows_scores, proposalN_indices, proposalN_windows_feature, proposalN_windows_logits,\
        # window_scores= self.windows_net(local_fm)
        proposalN_indices, proposalN_windows_scores, window_scores = self.score_net(self.proposalN, local_fm.detach(), ratios, window_nums_sum, cat_nums)

        cams_scores, cams, norm_fm_sum = 0, 0, 0
        if batch_idx == 0:
            # feature map sum
            fm_detach = fm.detach()
            fm_sum = torch.sum(fm_detach, dim=1)  # N*14*14
            max_values = torch.tensor([fm_sum[i].max() for i in range(batch_size)]).cuda()
            min_values = torch.tensor([fm_sum[i].min() for i in range(batch_size)]).cuda()
            norm_fm_sum = [(fm_sum[i] - min_values[i]) / (max_values[i] - min_values[i]) for i in range(batch_size)]

            # cam
            target_weight = torch.index_select(self.rawcls_net.weight.detach(), dim=0, index=labels)
            fm_detach = fm_detach.reshape(batch_size, channel_size, -1)
            cams = [torch.mm(target_weight[i].reshape(1,-1),fm_detach[i]).reshape(1, side_size, side_size) for i in range(batch_size)]
            max_values = torch.tensor([cams[i].max() for i in range(batch_size)]).cuda()
            min_values = torch.tensor([cams[i].min() for i in range(batch_size)]).cuda()
            norm_cams_scores = [(cams[i] - min_values[i]) / (max_values[i] - min_values[i]) for i in range(batch_size)]
            cams = torch.cat(norm_cams_scores, dim=0)  # batch*side_size*side_size

            # cams_scores = [self.avgpools[i](cams).reshape(batch_size, -1) for i in range(len(window_side))]
            # proposalN_cams_scores = [torch.cat([torch.index_select(cams_scores[i][j], dim=0, index=proposalN_indices[i][j]).unsqueeze(0)
            #                                     for j in range(batch_size) ], dim=0) for i in range(len(window_side))]
            # cams_scores = [self.avgpools[i](cams).reshape(batch_size, -1) for i in range(len(window_side))]

        # # raw branch with random drop
        # raw_feature = self.avgpool(fm).view(fm.size(0), -1)  # N*2048
        # assert (masks_cp != masks).any()
        # drop_window = np.random.randint(2*self.proposalN, size=batch_size)
        # raw_masks = np.reshape(masks, (batch_size, self.proposalN, side_size, side_size))      # N*proposalN*14*14
        # raw_masks = np.concatenate((raw_masks, np.reshape(masks_cp, (batch_size, self.proposalN, side_size, side_size))), axis=1)
        # drop_raw_masks = np.array([raw_mask[drop_window[i], ...] for i, raw_mask in enumerate(raw_masks)])
        #
        # drop_raw_masks = torch.from_numpy(drop_raw_masks).cuda().unsqueeze(1).repeat(1, 2048, 1, 1).type(
        # torch.float32)  # N*2048*14*14
        # drop_raw_fm = fm * drop_raw_masks.detach()  # N*2048*14*14
        # drop_raw_fm_sum = drop_raw_fm.sum(dim=[2, 3])  # N * 2048
        # drop_masks_sum = drop_raw_masks.sum(dim=[2, 3])  # N * 2048
        # drop_raw_embedding = drop_raw_fm_sum / drop_masks_sum
        # raw_logits = self.rawcls_net(drop_raw_embedding)  # (N*proposalN)*200

        # cat branch
        # cat_feature = torch.cat((raw_feature.unsqueeze(1), topN_windows_feature.reshape(batch_size,self.topN,2048)), dim=1)
        # add_cat_feature = torch.sum(cat_feature, dim=1)

        # window_imgs cls

        window_imgs = torch.zeros([batch_size, self.proposalN, 3, 224, 224]).cuda()  # [N, 4, 3, 224, 224]
        for i in range(batch_size):
            for j in range(self.proposalN):
                [x0, y0, x1, y1] = coordinates_cat[proposalN_indices[i, j]]
                window_imgs[i:i + 1, j] = F.interpolate(local_imgs[i:i + 1, :, x0:(x1+1), y0:(y1+1)], size=(224, 224), mode='bilinear',
                                                      align_corners=True)  # [N, 4, 3, 224, 224]
        window_imgs = window_imgs.reshape(batch_size * self.proposalN, 3, 224, 224)  # [N*4, 3, 224, 224]
        _, window_embeddings, _ = self.pretrained_model(window_imgs.detach())  # [N*4, 2048]
        del window_imgs
        window_embedding = window_embeddings.reshape(batch_size, self.proposalN, -1)  # [N, 6, 2048]
        window_embedding = window_embedding[:, :self.topN, ...]  # [N, 4, 2048]  contiguous返回内存分布连续的tensor,以便view操作
        # window_embedding = torch.sum(window_embedding, dim=1)
        window_embedding = window_embedding.reshape(batch_size, -1)  # [N, 8192]
        #
        # # proposalN_windows_logits have the shape: B*N*200
        proposalN_windows_logits = self.windowcls_net(window_embeddings)  # [N* 4, 200]

        # concat branch
        # cat_feature = torch.cat((window_embedding, embedding, local_embeddings), dim=1)
        # cat_feature = torch.cat((window_embedding, embedding), dim=1)
        # cat_logits = self.catcls_net(cat_feature)

        # lstm
        # cat_feature = torch.cat((proposalN_windows_feature, embedding.unsqueeze(1)), dim=1)
        # output, _ = self.lstm(cat_feature)
        # _, seq_len, hidden_size = output.shape
        # cat_logits = self.windowcls_net(output.reshape(batch_size*seq_len, hidden_size))

        # # concat_loss list
        # windows_feature_groups = [proposalN_windows_features[:, 2048*x:2048*(x+self.topN)] for x in range(1 + self.proposalN - self.topN)]  # (N*topN)*2048
        # concat_logits_list = [self.catcls_net(torch.cat((raw_feature, x), dim=1)).detach() for x in windows_feature_groups]
        # concat_logits_groups = torch.cat(concat_logits_list, dim=1).view(-1, 200) # N*(1 + self.proposalN - self.topN)*200 --> (N*(1 + self.proposalN - self.topN))*200

        # windows set zeros
        # masks = torch.from_numpy(masks).cuda().unsqueeze(1).repeat(1, 2048, 1, 1).type(
        #     torch.float32)  # (N*proposalN)*2048*14*14
        # fm_repeat = torch.repeat_interleave(fm.detach(), 6, dim=0)  # (N*proposalN)*2048*14*14
        # drop_fm = fm_repeat * masks                        # (N*proposalN)*2048*14*14
        # drop_embedding = self.avgpool(drop_fm).view(-1, 2048)           # (N*proposalN)*2048
        # drop_logits = self.rawcls_net(drop_embedding)                    # (N*proposalN)*200

        # drop raw loss list

        # # BiLSTM
        # # windows feature 按windows size由大到小，概率由大到小排序
        # topN_windows_feature = [topN_windows_feature[-(i+1)].view(batch_size, self.topN, channel_size) for i in range(len(window_side))]
        # topN_windows_feature = torch.cat(topN_windows_feature, dim=1)   # N*(topN*3)*2048
        # lstm_cat_feature = torch.cat((raw_feature.unsqueeze(1window_imgs), topN_windows_feature), dim=1)     # N*(1+topN*3)*2048


        # # windows_drop
        # drop_coor = [window_coordinates[i][proposalN_indices[i].view(-1).cpu().numpy()] for i in range(len(window_side))]  # (N*proposalN, 4)
        # masks = [np.ones([drop_coor[i].shape[0], side_size, side_size], dtype=int) for i in range(len(window_side))]  # (N*proposalN)*14*14
        # for i in range(len(window_side)):
        #     for idx, coor in enumerate(drop_coor[i]):
        #         masks[i][idx, coor[0]:coor[2], coor[1]:coor[3]] = 0
        #
        # masks = [torch.from_numpy(masks[i]).cuda().unsqueeze(1).repeat(1,2048,1,1).type(torch.float32) for i in range(len(window_side))]    # (N*proposalN)*2048*14*14
        # fm_repeat = torch.repeat_interleave(fm.detach(),6,dim=0)                        # (N*proposalN)*2048*14*14
        # drop_fm = [fm_repeat*masks[i] for i in range(len(window_side))]                         # (N*proposalN)*2048*14*14
        # drop_fm_sum = [drop_fm[i].sum(dim=[2,3]) for i in range(len(window_side))]                      # (N*proposalN) * 2048
        # masks_sum = [masks[i].sum(dim=[2,3]) for i in range(len(window_side))]                          # (N*proposalN) * 2048
        # drop_embedding = [drop_fm_sum[i] / masks_sum[i] for i in range(len(window_side))]
        # drop_logits = [self.rawcls_net(drop_embedding[i]) for i in range(len(window_side))]             # (N*proposalN)*200

        # cat_logits = self.catcls_net(torch.cat((embedding, local_embeddings, window_embedding), dim=1))

        return raw_logits, proposalN_windows_scores, proposalN_windows_logits,\
               proposalN_indices, window_scores, coordinates, local_logits, local_imgs

# def list_loss(logits, targets):
#     temp = F.log_softmax(logits, -1)
#     loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
#     return torch.stack(loss)
#
# def ranking_loss(score, targets, proposal_num):
#     loss = Variable(torch.zeros(1).cuda())
#     batch_size = score.size(0)
#     for i in range(proposal_num):
#         targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)     # [N, 6]
#         # targets_p = (targets < targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)  # [N, 6]     for drop loss
#         pivot = score[:, i].unsqueeze(1)          # [N, 1]
#         loss_p = (1 - pivot + score) * targets_p    # 根据论文的话应该是 1 - score + pivot，不过由于tragets是反过来的，所以这个也是反过来的，没毛病
#         loss_p = torch.sum(F.relu(loss_p))
#         loss += loss_p
#     return loss / batch_size
#
# def list_concat_loss(topN, proposalN, concat_logits_groups, labels):
#     batch = labels.shape[0]
#     labels = labels.unsqueeze(1).repeat(1, 1+proposalN-topN).view(-1)
#     log_logits = F.log_softmax(concat_logits_groups, -1)
#     loss = [-log_logits[i][labels[i].item()] for i in range(concat_logits_groups.size(0))]
#
#     return torch.stack(loss).view(batch,1+proposalN-topN)
#
# def ranking_concat_loss(score, targets, topN, proposal_num):
#     loss = Variable(torch.zeros(1).cuda())
#     batch_size = score.size(0)
#     for i in range(1+proposal_num-topN):
#         targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)     # [N, 6]
#         pivot = score[:, i].unsqueeze(1)          # [N, 1]
#         loss_p = (1 - pivot + score[:, :(1+proposal_num-topN)]) * targets_p    # 根据论文的话应该是 1 - score + pivot，不过由于tragets是反过来的，所以这个也是反过来的，没毛病
#         loss_p = torch.sum(F.relu(loss_p))
#         loss += loss_p
#     return loss / batch_size
#
# def score_ranking_loss(cams_scores, window_scores, proposalN_indices,margin=0.5):
#     batch, windows_num = cams_scores[0].shape
#     loss = Variable(torch.zeros(1).cuda())
#     for i in range(len(window_side)):
#         # _, cam_rank_indices = torch.sort(cams_scores[i], dim=1, descending=True)
#         for k in range(batch):
#             unique_indices = torch.unique(proposalN_indices[i][k]).shape[0]
#             cam_indices_scores = torch.gather(cams_scores[i][k], dim=0, index=proposalN_indices[i][k][:unique_indices])
#             _, cam_rank_indices = torch.sort(cam_indices_scores, descending=True)
#             window_ranked_scores = torch.gather(window_scores[i][k], dim=0, index=cam_rank_indices)
#             for j in range(unique_indices-1):
#                 window_ranked_scores_diff = window_ranked_scores[j] - window_ranked_scores
#                 # window_ranked_scores_diff_sum = torch.sum(F.relu(window_ranked_scores_diff[:j]+margin)) + \
#                 #                                 torch.sum(F.relu(-(window_ranked_scores_diff[j+1:])+margin))
#                 window_ranked_scores_diff_sum = torch.sum(F.relu(-(window_ranked_scores_diff[j + 1:]) + margin))
#                 loss += window_ranked_scores_diff_sum
#     return loss
#
#
# class LabelSmoothing(nn.Module):
#     "Implement label smoothing.  size表示类别总数 "
#     def __init__(self, size, smoothing=0.0):
#         super(LabelSmoothing, self).__init__()
#         self.Logsoftmax = nn.LogSoftmax()
#         self.criterion = nn.KLDivLoss(size_average=False)
#         #self.padding_idx = padding_idx
#         self.confidence = 1.0 - smoothing#if i=y的公式
#         self.smoothing = smoothing
#         self.size = size
#         self.true_dist = None
#
#     def forward(self, x, target):
#         """
#         x表示输入 (N，M)N个样本，M表示总类数，每一个类的概率log P
#         target表示label（M，）
#         """
#         assert x.size(1) == self.size
#         x = self.Logsoftmax(x)
#         true_dist = x.data.clone()#先深复制过来
#         #print true_dist
#         true_dist.fill_(self.smoothing / (self.size - 1))#otherwise的公式
#         #print true_dist
#         #变成one-hot编码，1表示按列填充，
#         #target.data.unsqueeze(1)表示索引,confidence表示填充的数字
#         true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#         self.true_dist = true_dist
#         return self.criterion(x, Variable(true_dist, requires_grad=False))
