import os
import json
import numpy as np
import random
import time
import en_core_web_sm
from spacy.tokenizer import Tokenizer
from tqdm import tqdm
import re
from transformers import BertTokenizer
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class SVQADataset(Dataset):
    shapes = set(['ball', 'block', 'cube', 'cylinder', 'sphere'])
    sphere = set(['ball', 'sphere'])
    cube = set(['block', 'cube'])
    cylinder = set(['cylinder'])

    colors = set(['red', 'cyan', 'brown', 'blue', 'purple', 'green', 'gray', 'yellow'])

    materials = set(['metallic', 'matte', 'rubber', 'shiny', 'metal'])
    rubber = set(['matte', 'rubber'])
    metal = set(['metal', 'metallic', 'shiny'])

    question_cate2label = {
        'count': 0,
        'integer comparison': 1,
        'exist': 2,
        'attribute comparison': 3,
        'query': 4
    }

    question_type2label = {
        'what': 0,
        'how many': 1,
        'boolean': 2
    }

    def __init__(self, cfg, split):
        self.cfg = cfg

        print('Speaker Dataset loading vocab json file: ', cfg.data.vocab_json)
        self.vocab_json = cfg.data.vocab_json
        self.word_to_idx = json.load(open(self.vocab_json, 'r'))
        self.idx_to_word = {}
        for word, idx in self.word_to_idx.items():
            self.idx_to_word[idx] = word
        self.vocab_size = len(self.idx_to_word)
        self.json_data = None
        print('vocab size is ', self.vocab_size)

        # self.type_mapping = json.load(open(cfg.data.type_mapping_json, 'r'))
        # self.type_to_img = {}
        # for k, v in self.type_mapping.items():
        #     self.type_to_img[k] = set([int(x.split('.')[0]) for x in v])

        # self.d_feat_dir = cfg.data.default_feature_dir
        # self.s_feat_dir = cfg.data.semantic_feature_dir
        # self.n_feat_dir = cfg.data.nonsemantic_feature_dir
        #
        # self.d_feats = sorted(os.listdir(self.d_feat_dir))
        # self.s_feats = sorted(os.listdir(self.s_feat_dir))
        # self.n_feats = sorted(os.listdir(self.n_feat_dir))
        #
        # assert len(self.d_feats) == len(self.s_feats) == len(self.n_feats), \
        #     'The number of features are different from each other!'
        #
        # self.d_img_dir = cfg.data.default_img_dir
        # self.s_img_dir = cfg.data.semantic_img_dir
        # self.n_img_dir = cfg.data.nonsemantic_img_dir
        #
        # self.d_imgs = sorted(os.listdir(self.d_img_dir))
        # self.s_imgs = sorted(os.listdir(self.s_img_dir))
        # self.n_imgs = sorted(os.listdir(self.n_img_dir))

        # self.splits = json.load(open(cfg.data.splits_json, 'r'))

        # loading json data
        self.split = split

        if split == 'train':
            with open(cfg.data.train_data_path, 'r') as f:
                self.json_data = json.load(f)
            self.batch_size = cfg.data.train.batch_size
            self.seq_per_img = cfg.data.train.seq_per_img
            self.split_idxs = [int(i["video_id"]) for i in self.json_data]
            self.num_samples = len(self.split_idxs)
            if cfg.data.train.max_samples is not None:
                self.num_samples = min(cfg.data.train.max_samples, self.num_samples)
        elif split == 'val':
            with open(cfg.data.val_data_path, 'r') as fd:
                self.json_data = json.load(fd)
            self.batch_size = cfg.data.val.batch_size
            self.seq_per_img = cfg.data.val.seq_per_img
            self.split_idxs = [int(i["video_id"]) for i in self.json_data]
            self.num_samples = len(self.split_idxs)
            if cfg.data.val.max_samples is not None:
                self.num_samples = min(cfg.data.val.max_samples, self.num_samples)
        elif split == 'test':
            with open(cfg.data.test_data_path, 'r') as ft:
                self.json_data = json.load(ft)
            self.batch_size = cfg.data.test.batch_size
            self.seq_per_img = cfg.data.test.seq_per_img
            self.split_idxs = [int(i["video_id"]) for i in self.json_data]
            self.num_samples = len(self.split_idxs)
            if cfg.data.test.max_samples is not None:
                self.num_samples = min(cfg.data.test.max_samples, self.num_samples)
        else:
            raise Exception('Unknown data split %s' % split)

        print("Dataset size for %s: %d" % (split, self.num_samples))

        # load the textual information
        print(self.question_type2label)
        self.question_category = [self.question_cate2label[i["question_category"]] for i in self.json_data]
        self.question_type = [self.question_type2label[i["question_type"]] for i in self.json_data]
        self.questions, self.answers = list(prepare(self.json_data))
        print('max_question_length : {}'.format(self.max_question_length))

        answers = []
        for i in self.json_data:
            answers.append(i["answer"])
        print(self.answers[0: 10])
        self.answers_bert = tokenizer(answers, return_tensors="pt", padding=True)

        self.questions = [self._encode_question(q, self.max_question_length, if_q=True) for q in self.questions]
        self.answers = [self._encode_answers(a) for a in self.answers]

        # self.h5_label_file = h5py.File(cfg.data.h5_label_file, 'r')
        # seq_size = self.h5_label_file['labels'].shape
        # self.labels = self.h5_label_file['labels'][:]  # just gonna load...
        # self.neg_labels = self.h5_label_file['neg_labels'][:]
        # self.max_question_length = seq_size[1]
        # self.label_start_idx = self.h5_label_file['label_start_idx'][:]
        # self.label_end_idx = self.h5_label_file['label_end_idx'][:]
        # self.neg_label_start_idx = self.h5_label_file['neg_label_start_idx'][:]
        # self.neg_label_end_idx = self.h5_label_file['neg_label_end_idx'][:]
        # print('Max sequence length is %d' % self.max_question_length)
        # self.h5_label_file.close()

        # loading visual features
        # video_list = os.listdir('../VideoQG/data/SVQA/useVideo')
        # self.video_ids = [int(i.split('.')[0]) for i in video_list]
        # self.video_ids.sort()
        self.frame_features_path = cfg.data.image_feature_file
        with h5py.File(self.frame_features_path, 'r') as f:
            feat_video_ids = f['video_id'][()]
        self.video_id_to_index = {int(id): i for i, id in enumerate(feat_video_ids)}



    def _load_video(self, video_id):
        """loading frame feature"""
        if not hasattr(self, 'frame_features'):
            self.frame_features = h5py.File(self.frame_features_path, 'r')
        idx = self.video_id_to_index[video_id]
        vid = self.frame_features['video_id'][idx]
        assert int(vid) == video_id
        feats = self.frame_features["resnet_features"][idx]
        return feats


    @property
    def max_question_length(self):
        if not hasattr(self, '_max_length'):
            data_max_length = max(map(len, self.questions))
            print(f'data_max_length', data_max_length)
            # data_max_length = data_max_length2
            self._max_length = data_max_length
        return self._max_length
    
    def _encode_question(self, sentence, max_length, if_q=False):
        """ Turn a question into a vector of indices and a question length """
        if if_q:
            enc_q = [self.word_to_idx['<START>']] + [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for
                                                        word in sentence] + [
                        self.word_to_idx['<END>']] + [self.word_to_idx['<NULL>']] * (max_length - len(sentence))
            # Find questions lengths
            q_len = len(sentence) + 2
            vec = torch.LongTensor(enc_q)
        else:
            enc_q = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in sentence] + [
                self.word_to_idx['<NULL>']] * (max_length - len(sentence))
            # Find questions lengths
            q_len = len(sentence)
            vec = torch.LongTensor(enc_q)
        return vec, torch.LongTensor([q_len])


    def _encode_answers(self, answer):
        """ Turn an answer into a vector """
        # answer vec will be a vector of answer counts to determine which answers will contribute to the loss.
        # this should be multiplied with 0.1 * negative log-likelihoods that a model produces and then summed up
        # to get the loss that is weighted by how many humans gave that answer
        enc_a = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in answer] + [
            self.word_to_idx['<NULL>']] * (1 - len(answer))
        # Find questions lengths
        a_len = len(answer)
        vec = torch.LongTensor(enc_a[:1])
        return vec, torch.LongTensor([a_len])


    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        random.seed()
        video_id = self.split_idxs[index]
        q, q_length = self.questions[index]
        a, a_length = self.answers[index]
        qt = torch.LongTensor([self.question_type[index]])
        qc = torch.LongTensor([self.question_category[index]])
        # qc = a
        visual_embeds = torch.from_numpy(self._load_video(video_id)).float()
        visual_token_type_ids = torch.ones(visual_embeds.shape[:1], dtype=torch.long)
        visual_attention_mask = torch.ones(visual_embeds.shape[:1], dtype=torch.long)

        # Fetch image data
        # one easy way to augment data is to use nonsemantically changed
        # scene as the default :)
        # if self.split == 'train':
        #     if random.random() < 0.5:
        #         d_feat_path = os.path.join(self.d_feat_dir, self.d_feats[img_idx])
        #         d_img_path = os.path.join(self.d_img_dir, self.d_imgs[img_idx])
        #         n_feat_path = os.path.join(self.n_feat_dir, self.n_feats[img_idx])
        #         n_img_path = os.path.join(self.n_img_dir, self.n_imgs[img_idx])
        #     else:
        #         d_feat_path = os.path.join(self.n_feat_dir, self.n_feats[img_idx])
        #         d_img_path = os.path.join(self.n_img_dir, self.n_imgs[img_idx])
        #         n_feat_path = os.path.join(self.d_feat_dir, self.d_feats[img_idx])
        #         n_img_path = os.path.join(self.d_img_dir, self.d_imgs[img_idx])
        # else:
        #     d_feat_path = os.path.join(self.d_feat_dir, self.d_feats[img_idx])
        #     d_img_path = os.path.join(self.d_img_dir, self.d_imgs[img_idx])
        #     n_feat_path = os.path.join(self.n_feat_dir, self.n_feats[img_idx])
        #     n_img_path = os.path.join(self.n_img_dir, self.n_imgs[img_idx])
        #
        # q_feat_path = os.path.join(self.s_feat_dir, self.s_feats[img_idx])
        # q_img_path = os.path.join(self.s_img_dir, self.s_imgs[img_idx])
        #
        # d_feature = torch.FloatTensor(np.load(d_feat_path))
        # n_feature = torch.FloatTensor(np.load(n_feat_path))
        # q_feature = torch.FloatTensor(np.load(q_feat_path))
        #
        # # Fetch change type labels
        # aux_label_pos = -1
        # for type, img_set in self.type_to_img.items():
        #     if img_idx in img_set:
        #         aux_label_pos = self.type_to_label[type]
        #         break
        # aux_label_neg = self.type_to_label['no_change']
        #
        # Fetch sequence labels
        # n_cap = q_length
        #
        # seq = np.zeros([self.seq_per_img, self.max_question_length + 1], dtype=int)
        # if n_cap < self.seq_per_img:
        #     # we need to subsample (with replacement)
        #     for q in range(self.seq_per_img):
        #         ixl = random.randint(ix1, ix2)
        #         seq[q, :self.max_question_length] = \
        #             self.labels[ixl, :self.max_question_length]
        # else:
        #     ixl = random.randint(ix1, ix2 - self.seq_per_img + 1)
        #     seq[:, :self.max_question_length] = \
        #         self.labels[ixl: ixl + self.seq_per_img, :self.max_question_length]

        # # Fetch negative sequence labels
        # ix1 = self.neg_label_start_idx[img_idx]
        # ix2 = self.neg_label_end_idx[img_idx]
        # n_cap = ix2 - ix1 + 1
        #
        # neg_seq = np.zeros([self.seq_per_img, self.max_question_length + 1], dtype=int)
        # if n_cap < self.seq_per_img:
        #     # we need to subsample (with replacement)
        #     for q in range(self.seq_per_img):
        #         ixl = random.randint(ix1, ix2)
        #         neg_seq[q, :self.max_question_length] = \
        #             self.neg_labels[ixl, :self.max_question_length]
        # else:
        #     ixl = random.randint(ix1, ix2 - self.seq_per_img + 1)
        #     neg_seq[:, :self.max_question_length] = \
        #         self.neg_labels[ixl: ixl + self.seq_per_img, :self.max_question_length]
        #
        # Generate masks
        mask = np.ones_like(q)
        # for i in range(q_length + 1):
        #     mask[i] = 1
        # nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 1, seq)))
        # for ix, row in enumerate(mask):
        #     row[:nonzeros[ix]] = 1

        mask = torch.from_numpy(mask).float()
        #
        # neg_mask = np.zeros_like(neg_seq)
        # nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 1, neg_seq)))
        # for ix, row in enumerate(neg_mask):
        #     row[:nonzeros[ix]] = 1
        #
        # return (d_feature, n_feature, q_feature,
        #         seq, neg_seq, mask, neg_mask, aux_label_pos, aux_label_neg,
        #         d_img_path, n_img_path, q_img_path)
        answers_bert_ids, answers_bert_token_type_ids, answers_bert_mask = \
            self.answers_bert['input_ids'][index], self.answers_bert['token_type_ids'][index], \
            self.answers_bert['attention_mask'][index]

        return visual_embeds, visual_token_type_ids, visual_attention_mask, q, q_length, a, a_length, mask, qt, qc, answers_bert_ids, answers_bert_token_type_ids, answers_bert_mask


    def get_vocab_size(self):
        return self.vocab_size

    def get_idx_to_word(self):
        return self.idx_to_word

    def get_word_to_idx(self):
        return self.word_to_idx

    def get_max_seq_length(self):
        return self.max_question_length


def svqa_collate(batch):
    transposed = list(zip(*batch))
    d_feat_batch = transposed[0]
    n_feat_batch = transposed[1]
    q_feat_batch = transposed[2]
    seq_batch = default_collate(transposed[3])
    neg_seq_batch = default_collate(transposed[4])
    mask_batch = default_collate(transposed[5])
    neg_mask_batch = default_collate(transposed[6])
    aux_label_pos_batch = default_collate(transposed[7])
    aux_label_neg_batch = default_collate(transposed[8])
    if any(f is not None for f in d_feat_batch):
        d_feat_batch = default_collate(d_feat_batch)
    if any(f is not None for f in n_feat_batch):
        n_feat_batch = default_collate(n_feat_batch)
    if any(f is not None for f in q_feat_batch):
        q_feat_batch = default_collate(q_feat_batch)

    d_img_batch = transposed[9]
    n_img_batch = transposed[10]
    q_img_batch = transposed[11]
    return (d_feat_batch, n_feat_batch, q_feat_batch,
            seq_batch, neg_seq_batch,
            mask_batch, neg_mask_batch,
            aux_label_pos_batch, aux_label_neg_batch,
            d_img_batch, n_img_batch, q_img_batch)


class SVQADataLoader(DataLoader):

    def __init__(self, dataset, **kwargs):
        kwargs['collate_fn'] = svqa_collate
        super().__init__(dataset, **kwargs)


def prepare(data):
    print('tokenizer questions and answers ...')
    question_gts, answers = [], []
    nlp = en_core_web_sm.load()
    tokenizer = Tokenizer(nlp.vocab)
    for i, row in tqdm(enumerate(data)):
        # sentence_gt = _special_chars.sub('', sentence_gt)
        question_gt = row['question'].strip().lower()
        answer = row['answer'].strip().lower()
        question_gt = re.sub('\.|,', '', question_gt)
        question = [t.text if '?' not in t.text else t.text[:-1] for t in tokenizer(question_gt)]
        answer = [t.text for t in tokenizer(answer)]
        if i < 3:
            print(question)
            print(answer)
        question_gts.append(question)
        answers.append(answer)
    # print(f'len:', len(sentences_input), sentences_input[1], len(sentences_input[1]))
    return question_gts, answers