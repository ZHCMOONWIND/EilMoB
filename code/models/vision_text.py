import torch
import tqdm
import os
from typing import Optional
from torch import Tensor
from PIL import Image
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch import optim, nn
from transformers import ViTForImageClassification, get_linear_schedule_with_warmup
import torch.nn.functional as F
from queue import Queue
import scipy.stats as st
import numpy as np
import math
from torch.optim.lr_scheduler import LambdaLR
import cv2
import ast
import json
def read_json(path):
    with open(path,"r",encoding = 'utf-8') as f:
        data = json.load(f)
    return data
def write_json(path,data):
    with open(path,"w",encoding = 'utf-8') as f:
        json.dump(data,f)
def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):

    def _lr_lambda(current_step):

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr
    return LambdaLR(optimizer, _lr_lambda, last_epoch)

class MultiModalDataset(Dataset):
    def __init__(self, text_tools, vision_transforms, args, mode):
        self.args = args
        self.vision_transform = vision_transforms[mode]
        self.mode = mode
        self.text_arr, self.llm_arr, self.img_path, self.label, self.idx2file = self.init_data()

    def init_data(self):
        if self.mode == 'train':
            text_path = self.args.train_text_path
            vision_path = self.args.train_image_path
            llm_path = self.args.train_llm_path
        elif self.mode == 'test':
            text_path = self.args.test_text_path
            vision_path = self.args.test_image_path
            llm_path = self.args.test_llm_path
        else :
            text_path = text_path = self.args.all_text_path
            vision_path = self.args.all_image_path
        text_arr, llm_arr, img_path, labels, idx2file = {}, {}, {}, {}, []
        skip_words = ['exgag', 'sarcasm', 'sarcastic', '<url>', 'reposting', 'joke', 'humor', 'humour', 'jokes', 'irony', 'ironic']
        llm_dic = {}

        for line in open(llm_path, 'r', encoding='utf-8').readlines():
            content = eval(line)
            file_name, llm = content[0], content[1]
            llm_dic[file_name] = llm

        for line in open(text_path, 'r', encoding='utf-8').readlines():
            content = eval(line)
            file_name, text, label = content[0], content[1], content[2]
            flag = False
            for skip_word in skip_words:
                if skip_word in content[1]: flag = True
            if flag: continue

            cur_img_path = os.path.join(vision_path, file_name+'.jpg')
            if not os.path.exists(cur_img_path):
                print(file_name)
                continue
            
            text_arr[file_name], llm_arr[file_name], labels[file_name] = text, llm_dic[file_name], label
            img_path[file_name] = os.path.join(vision_path, file_name+'.jpg')
            idx2file.append(file_name)
        return text_arr, llm_arr, img_path, labels, idx2file

    def __getitem__(self, idx):
        file_name = self.idx2file[idx]
        text = self.text_arr[file_name]
        llm = self.llm_arr[file_name]
        img_path = self.img_path[file_name]
        label = self.label[file_name]

        img = Image.open(img_path).convert("RGB")
        img = self.vision_transform(img)
        return file_name, img, text, llm, label

    def __len__(self):
        return len(self.label)

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, A, B):
        batch_size = A.size(0)
        A_norm = F.normalize(A, p=2, dim=1)  # [batchsize, feature_dim]
        B_norm = F.normalize(B, p=2, dim=1)  # [batchsize, feature_dim]
        
        sim_AB = torch.matmul(A_norm, B_norm.T)  # [batchsize, batchsize]

        sim_AB = sim_AB / self.temperature  # [batchsize, batchsize]
        
        labels = torch.arange(batch_size).to(A.device)  # [batchsize]
        
        loss_A_to_B = F.cross_entropy(sim_AB, labels)
        loss_B_to_A = F.cross_entropy(sim_AB.T, labels)
    
        loss = (loss_A_to_B + loss_B_to_A) / 2
        return loss
class CrossAttention(nn.Module):
    def __init__(self, feature_dim, dropout_prob=0.1):
        super(CrossAttention, self).__init__()
        self.text_linear = nn.Linear(feature_dim, feature_dim)
        self.extra_linear = nn.Linear(feature_dim, feature_dim)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, query, key, value):
        if query.shape[-1] != 768:
            query = self.text_linear(query)
        if key.shape[-1] != 768:
            key = self.extra_linear(key)
            value = self.extra_linear(value)
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(
            torch.tensor(key.size(-1), dtype=torch.float32)
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value)
        attended_values = self.dropout(attended_values)
        return attended_values
class Gated_Fusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gated_vt = nn.Linear(2*768, 2, bias=True)
        self.gated_tl = nn.Linear(2*768, 2, bias=True)
        self.gated_vl = nn.Linear(2*768, 2, bias=True)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, embeddings1, embeddings2, mode):
        assert mode in ['vt', 'tl', 'vl']
        embedding = torch.cat((embeddings1, embeddings2), dim=1)
        if mode == 'vt': logits = self.gated_vt(embedding)
        elif mode == 'tl': logits = self.gated_tl(embedding)
        elif mode == 'vl': logits = self.gated_vl(embedding)
        weights = self.softmax(logits)
        alpha_1, alpha_2 = weights[:, 0].unsqueeze(1), weights[:, 1].unsqueeze(1)
        F_fused = alpha_1 * embeddings1 + alpha_2 * embeddings2
        return F_fused

class CrossAttention(nn.Module):
    def __init__(self, feature_dim, dropout_prob=0.1):
        super(CrossAttention, self).__init__()
        self.text_linear = nn.Linear(feature_dim, feature_dim)
        self.extra_linear = nn.Linear(feature_dim, feature_dim)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value):
        if query.shape[-1] != 768:
            query = self.text_linear(query)
        if key.shape[-1] != 768:
            key = self.extra_linear(key)
            value = self.extra_linear(value)
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(
            torch.tensor(key.size(-1), dtype=torch.float32)
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value)
        attended_values = self.dropout(attended_values)
        return attended_values
class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,return_attn=False,kdim=None,vdim=None):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,kdim=kdim,vdim=vdim,batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before
        self.return_attn = return_attn

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        tgt2, cross_attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
    
class LimCol(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.text_fc =  nn.Sequential(
                nn.Linear(768, 768),
                nn.Dropout(0.1),
                nn.GELU()
            )
        self.vision_fc =  nn.Sequential(
                nn.Linear(768, 768),
                nn.Dropout(0.1),
                nn.GELU()
            )
        self.llm_fc = nn.Sequential(
                nn.Linear(768, 768),
                nn.Dropout(0.1),
                nn.GELU()
            )
        self.llm_fc_fg = nn.Sequential(nn.Linear(768, 768),
                                 nn.ReLU(),
                                 nn.LayerNorm(768)
                                 )
        self.text_llm_query_fc = nn.Sequential(nn.Linear(768*2, 768),
                                 nn.ReLU(),
                                 nn.Linear(768, 768),
                                 nn.LayerNorm(768)
                                 )
        self.vision_llm_query_fc = nn.Sequential(nn.Linear(768*2, 768),
                                 nn.ReLU(),
                                 nn.Linear(768, 768),
                                 nn.LayerNorm(768)
                                 )
        self.llm_text_query_fc = nn.Sequential(nn.Linear(768*2, 768),
                                 nn.ReLU(),
                                 nn.Linear(768, 768),
                                 nn.LayerNorm(768)
                                 )
        self.llm_vision_query_fc = nn.Sequential(nn.Linear(768*2, 768),
                                 nn.ReLU(),
                                 nn.Linear(768, 768),
                                 nn.LayerNorm(768)
                                 )
        self.vl_cross = TransformerLayer(768,8)
        self.tl_cross = TransformerLayer(768,8)
        self.text_llm_key_sa = TransformerEncoderLayer(768,8)
        self.vision_llm_key_sa = TransformerEncoderLayer(768,8)
        self.llm_vision_key_sa = TransformerEncoderLayer(768,8)
        self.llm_text_key_sa = TransformerEncoderLayer(768,8)
        self.text_llm_cross = TransformerLayer(768,8)
        self.vision_llm_cross = TransformerLayer(768,8)
        self.llm_text_cross = TransformerLayer(768,8)
        self.llm_vision_cross = TransformerLayer(768,8)
        self.cross_att_vt = CrossAttention(768,0.1)
        self.cross_att_lvt = CrossAttention(768,0.1)
        self.mlp_layer = nn.Sequential(nn.Linear(768 * 2, 768),
                                       nn.ReLU(),
                                       nn.Linear(768, 768),
                                       nn.LayerNorm(768))
        self.attetion_block = nn.Sequential(nn.Linear(768 * 2, 768),
                                            nn.ReLU(),
                                            nn.Linear(768, 768),
                                            nn.Sigmoid())
        self.sentiment_bias_text = nn.Parameter(torch.tensor(0.1))
        self.sentiment_bias_llm = nn.Parameter(torch.tensor(0.1))
        self.len_weight_text = nn.Parameter(torch.tensor(0.32))
        self.sentiment_weight_text = nn.Parameter(torch.tensor(0.68))
        self.len_weight_llm = nn.Parameter(torch.tensor(0.32))
        self.sentiment_weight_llm = nn.Parameter(torch.tensor(0.68))

        self.ReLu=nn.ReLU()
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.semantic_fusion = Gated_Fusion(args)
        self.softmax = nn.Softmax(dim=-1)

        self.vision_text_classifier = nn.Linear(768, 2)
        self.text_llm_classifier = nn.Linear(768, 2)
        self.vision_llm_classifier = nn.Linear(768, 2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, vision_embeddings, text_embeddings, llm_embeddings, text_lens, llm_lens, text_sentiments, llm_sentiments, mode, label=None):
        vision_feat, text_feat, llm_feat = vision_embeddings['cls'], text_embeddings['cls'], llm_embeddings['cls']

        batch_size = vision_feat.size()[0]
        device = self.args.gpu
        vision_feat = self.vision_fc(vision_feat)
        text_feat = self.text_fc(text_feat)
        llm_feat = self.llm_fc(llm_feat)
        

        #Cross-modal Interaction Module
        vision_feat_l = self.vl_cross(vision_feat, llm_feat)
        llm_feat_v = self.vl_cross(llm_feat, vision_feat)

        text_feat_l = self.tl_cross(text_feat, llm_feat)
        llm_feat_t = self.tl_cross(llm_feat, text_feat)
        

        #Incongruity Mining Module
        text_llm_query = self.text_llm_query_fc(torch.cat([text_feat, llm_feat_v], dim=-1)).unsqueeze(1)
        text_llm_key = torch.cat((text_feat.unsqueeze(1),llm_feat_v.unsqueeze(1)),dim=1)
        text_llm_key = self.text_llm_key_sa(text_llm_key)
        text_llm = self.text_llm_cross(text_llm_query, text_llm_key).squeeze(1)

        vision_llm_query = self.vision_llm_query_fc(torch.cat([vision_feat, llm_feat_t], dim=-1)).unsqueeze(1) 
        vision_llm_key = torch.cat((vision_feat.unsqueeze(1),llm_feat_t.unsqueeze(1)),dim=1)
        vision_llm_key = self.vision_llm_key_sa(vision_llm_key)
        vision_llm = self.vision_llm_cross(vision_llm_query, vision_llm_key).squeeze(1)

        llm_text_query = self.llm_text_query_fc(torch.cat([llm_feat, text_feat_l], dim=-1)).unsqueeze(1)
        llm_text_key = torch.cat((llm_feat.unsqueeze(1),text_feat_l.unsqueeze(1)),dim=1)
        llm_text_key = self.llm_text_key_sa(llm_text_key)
        llm_feat_t = self.llm_text_cross(llm_text_query, llm_text_key).squeeze(1)

        llm_vision_query = self.llm_vision_query_fc(torch.cat([llm_feat, vision_feat_l], dim=-1)).unsqueeze(1)
        llm_vision_key = torch.cat((llm_feat.unsqueeze(1),vision_feat_l.unsqueeze(1)),dim=1)
        llm_vision_key = self.llm_vision_key_sa(llm_vision_key)
        llm_feat_v = self.llm_vision_cross(llm_vision_query, llm_vision_key).squeeze(1)

        #Modality Bridging Module
        text_semantic_CLS = self.cross_att_vt(text_llm, vision_llm, vision_llm)
        vision_semantic_CLS = self.cross_att_vt(vision_llm, text_llm, text_llm)

        llm_semantic_CLS_v = self.cross_att_lvt(llm_feat_v, vision_llm, vision_llm)
        llm_semantic_CLS_t = self.cross_att_lvt(llm_feat_t, text_llm, text_llm)

        #Adaptive Weighting Module
        #len_weight
        text_len_weight = text_lens/(text_lens+llm_lens)
        llm_len_weight = llm_lens/(text_lens+llm_lens)
        #sentiment_weight
        text_compound = text_sentiments[:, 3]
        llm_compound = llm_sentiments[:, 3]
        sentiment_diff_all = torch.abs(text_compound - llm_compound)
        text_compound = text_sentiments[:, 3].abs() + self.sentiment_bias_text
        llm_compound = llm_sentiments[:, 3].abs() + self.sentiment_bias_llm
        all_compound = text_compound + llm_compound
        text_sentiment_weight = text_compound / all_compound
        llm_sentiment_weight = llm_compound / all_compound
        #len+sentiment_weight
        text_len_radio  = self.len_weight_text / (self.len_weight_text+self.sentiment_weight_text)
        text_sentiment_radio  = self.sentiment_weight_text / (self.len_weight_text+self.sentiment_weight_text)
        llm_len_radio  = self.len_weight_llm / (self.len_weight_llm+self.sentiment_weight_llm)
        llm_sentiment_radio  = self.sentiment_weight_llm / (self.len_weight_llm+self.sentiment_weight_llm)
        text_weight = text_len_weight*text_len_radio + text_sentiment_weight*text_sentiment_radio
        llm_weight = llm_len_weight*llm_len_radio + llm_sentiment_weight*llm_sentiment_radio
        text_weight_expanded = text_weight.unsqueeze(1)
        llm_weight_expanded = llm_weight.unsqueeze(1)

        #Constraint Fusion Module
        final_cls_vt = self.semantic_fusion(vision_semantic_CLS, text_semantic_CLS, 'vt')
        final_cls_tl = self.semantic_fusion(text_semantic_CLS*text_weight_expanded, llm_semantic_CLS_t*llm_weight_expanded, 'tl')
        final_cls_vl = self.semantic_fusion(vision_semantic_CLS, llm_semantic_CLS_v, 'vl')

        vision_text_logit = self.vision_text_classifier(final_cls_vt)
        text_llm_logit = self.text_llm_classifier(final_cls_tl)
        vision_llm_logit = self.vision_llm_classifier(final_cls_vl)

        logits=[]
        if label is not None:
            loss = self.criterion(vision_text_logit, label) + self.criterion(text_llm_logit, label) + self.criterion(vision_llm_logit, label)
        
        vision_text_logit = self.softmax(vision_text_logit)
        text_llm_logit = self.softmax(text_llm_logit)
        vision_llm_logit = self.softmax(vision_llm_logit)
        logits.append(vision_text_logit)
        logits.append(text_llm_logit)
        logits.append(vision_llm_logit)
        return logits, loss
        

def get_multimodal_model(args):
    return LimCol(args)


def get_multimodal_configuration(args, model):
    optimizer = optim.AdamW(model.parameters(), lr=args.multimodal_lr, weight_decay=args.multimodal_weight_decay)
    num_training_steps = int(args.train_set_len / args.batch_size * args.epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    criterion = nn.CrossEntropyLoss()
    return optimizer, scheduler, criterion