import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import json
import random
import numpy as np
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VisualBertModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, features_dim, decoder_dim, attention_dim, dropout=0.5):
        """
        :param features_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.features_att = weight_norm(nn.Linear(features_dim, attention_dim))  # linear layer to transform encoded image
        self.decoder_att = weight_norm(nn.Linear(decoder_dim, attention_dim))  # linear layer to transform decoder's output
        self.full_att = weight_norm(nn.Linear(attention_dim, 1))  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, image_features, decoder_hidden):
        """
        Forward propagation.
        :param image_features: encoded images, a tensor of dimension (batch_size, 36, features_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.features_att(image_features)  # (batch_size, 7, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.dropout(self.relu(att1 + att2.unsqueeze(1)))).squeeze(2)  # (batch_size, 36)
        alpha = self.softmax(att)  # (batch_size, 7)
        attention_weighted_encoding = (image_features * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, features_dim)

        return attention_weighted_encoding


class VisionAttention(nn.Module):
    """Implements additive attention and return the attention vector used to weight the values.
        Additive attention consists in concatenating key and query and then passing them trough a linear layer."""

    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)

    def forward(self, hidden, enc_output, length=None):
        # key = [batch size, dec hid dim]
        # queries = [batch size, src sent len, enc hid dim]
        hidden = hidden.unsqueeze(1)
        bs, max_len, _ = enc_output.size()
        enc_output = self.attn(enc_output)
        attention = torch.sum(hidden * enc_output, dim=2)
        if length is not None:
            padding_mask = torch.arange(0, max_len).type_as(length).unsqueeze(0).expand(bs, max_len)
            padding_mask = ~padding_mask.lt(length)
            attention.masked_fill_(padding_mask, -math.inf)

        return F.softmax(attention, dim=1)


class ChangeDetectorDoubleAttDyn(nn.Module):

    def __init__(self, cfg, vocab_size=105):
        super().__init__()
        self.embed_dim = cfg.model.change_detector.emb_dim
        self.input_dim = cfg.model.change_detector.input_dim
        self.dim = cfg.model.change_detector.emb_dim
        self.visualBERT = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

        self.embedding = nn.Embedding(vocab_size, self.embed_dim)
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.embed_dim, batch_first=True)
        self.attention = Attention(self.dim, self.embed_dim, self.embed_dim)  # attention network

        self.embed = nn.Sequential(
            nn.Conv2d(self.input_dim, self.dim, kernel_size=1, padding=0),
            nn.GroupNorm(32, self.dim),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        self.relu = nn.ReLU()

        self.att = nn.Conv2d(self.dim, 1, kernel_size=1, padding=0)
        self.fc1 = nn.Linear(self.input_dim // 2, 6)
        self.fc2 = nn.Linear(self.input_dim, self.input_dim // 2)
        # self.fc4 = nn.Linear(self.input_dim + self.input_dim // 2, self.input_dim // 2)
        self.fc3 = nn.Linear(self.input_dim // 2, self.input_dim)
        self.fc_to_decode = nn.Linear(768, self.input_dim // 2)
        self.ac_fn = nn.Softmax(dim=1)
        self.init_weights()
        self.vision_attention = VisionAttention(768, self.embed_dim)
        self.fc_bert = nn.Linear(self.input_dim // 2, 768)


    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)


    def forward(self, inputs, visual_token_type_ids, visual_attention_mask, encoded_answers, qc, answers_ids, answers_token_type_ids, answers_mask):
        batch_size, frame_num, feat_dim, H, W = inputs.shape
        num = frame_num - 1
        # print('inputs.shape', inputs.shape)
        answers_embedding = self.embedding(encoded_answers)
        answers_lstm, (h_a, c_a) = self.lstm(answers_embedding)
        tup_res = []
        # diff_visual_feat = torch.zeros(batch_size, num, feat_dim).to(device)
        for i in range(1, frame_num):
            _, att = self.cal_diff(inputs[:, i, :, :].squeeze(), inputs[:, i - 1, :, :].squeeze())
            # diff_visual_feat[:, i, :] = att[-1]
            tup_res.append(att)
        image_features_attention = torch.zeros((batch_size, num, feat_dim)).to(device)

        w_b, w_a, attend1, attend2, input_attend = tup_res[0]
        # print(w_b.shape, attend1.shape)
        att_w_b = torch.zeros_like(w_b).to(device).expand(-1, num, -1, -1)
        att_w_a = torch.zeros_like(w_a).to(device).expand(-1, num, -1, -1)
        att1 = torch.zeros_like(attend1).unsqueeze(dim=1).expand(-1, num, -1)
        att2 = torch.zeros_like(attend2).unsqueeze(dim=1).expand(-1, num, -1)
        # print(att_w_b.shape)
        # print('tup_res_len:', len(tup_res))
        for i in range(num):
            w_b, w_a, attend1, attend2, input_attend = tup_res[i]
            image_features_attention[:, i, :] = input_attend
            att_w_a[:, i, :, :] = w_a.squeeze(dim=1)
            att_w_b[:, i, :, :] = w_b.squeeze(dim=1)
            att1[:, i, :] = attend1
            att2[:, i, :] = attend2

        visual_token_type_ids = torch.ones(image_features_attention[:, :, 0].squeeze().shape).to(device).long()
        visual_attention_mask = torch.ones(image_features_attention[:, :, 0].squeeze().shape).to(device).long()
        att1 = self.attention(att1, answers_lstm[:, -1, :])
        att2 = self.attention(att2, answers_lstm[:, -1, :])
        att1 = self.fc_bert(att1)
        att2 = self.fc_bert(att2)
        image_features_attention = self.fc3(image_features_attention)
        # att1 = self.fc3(att1)
        # att2 = self.fc3(att2)

        h1, c1 = h_a.squeeze(0), c_a.squeeze(0)

        vision_score = self.vision_attention(h1, image_features_attention)
        image_features_attention = torch.bmm(vision_score.unsqueeze(1), fusion_feature_diff).squeeze(1)

        # image_features_attention = self.ac_fn(image_features_attention)
        # att1 = self.ac_fn(att1)
        # att2 = self.ac_fn(att2)
        # att_w_a = self.ac_fn(att_w_a)
        # att_w_b = self.ac_fn(att_w_b)
        # image_features_attention = image_features_attention / num
        # att1 = att1 / num  # before
        # att2 = att2 / num  # after

        att_w_a = att_w_a.mean(dim=1)
        att_w_b = att_w_b.mean(dim=1)

        return att_w_b, att_w_a, att1, att2, image_features_attention


    def cal_diff(self, input_1, input_2):
        batch_size, _, H, W = input_1.size()
        input_diff = input_2 - input_1
        # print('input1', input_1.shape)
        # print('input_2', input_2.shape)
        # print('input_diff', input_diff.shape)
        input_before = torch.cat([input_1, input_diff], 1)
        input_after = torch.cat([input_2, input_diff], 1)
        # print('input_before.shape', input_before.shape)
        embed_before = self.embed(input_before)
        embed_after = self.embed(input_after)
        att_weight_before = F.sigmoid(self.att(embed_before))
        att_weight_after = F.sigmoid(self.att(embed_after))

        att_1_expand = att_weight_before.expand_as(input_1)
        attended_1 = (input_1 * att_1_expand).sum(2).sum(2)  # (batch, dim)
        att_2_expand = att_weight_after.expand_as(input_2)
        attended_2 = (input_2 * att_2_expand).sum(2).sum(2)  # (batch, dim)
        input_attended = attended_2 - attended_1
        pred = self.fc1(input_attended)
        # return att_weight_before, att_weight_after, attended_1, attended_2, input_attended

        return pred, (att_weight_before, att_weight_after, attended_1, attended_2, input_attended)


class AddSpatialInfo(nn.Module):

    def _create_coord(self, img_feat):
        batch_size, _, h, w = img_feat.size()
        coord_map = img_feat.new_zeros(2, h, w)
        for i in range(h):
            for j in range(w):
                coord_map[0][i][j] = (j * 2.0 / w) - 1
                coord_map[1][i][j] = (i * 2.0 / h) - 1
        sequence = [coord_map] * batch_size
        coord_map_in_batch = torch.stack(sequence)
        return coord_map_in_batch

    def forward(self, img_feat):
        coord_map = self._create_coord(img_feat)
        img_feat_aug = torch.cat([img_feat, coord_map], dim=1)
        return img_feat_aug
