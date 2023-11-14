hhhimport os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from tqdm import tqdm
import sys
import json
import argparse
from torch.autograd import Variable
import time
import numpy as np
import torch
torch.backends.cudnn.enabled = True
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from optimization import BertAdam
from datasets.datasets import create_dataset
from models.modules import ChangeDetectorDoubleAttDyn, AddSpatialInfo
from models.dynamic_speaker import DynamicSpeaker
from utils.logger import Logger
from utils.utils import AverageMeter, accuracy, set_mode, save_checkpoint, \
                        LanguageModelCriterion, decode_sequence, decode_beams, \
                        build_optimizer, coco_gen_format_save, one_hot_encode, \
                        EntropyLoss
from utils.eval_utils import metric

from utils.vis_utils import visualize_att

# Load config
parser = argparse.ArgumentParser()
dataset_name = 'svqa'
if dataset_name == 'svqa':
    from configs.config_svqa import cfg, merge_cfg_from_file
elif dataset_name == 'msvd_qa':
    from configs.config_msvd_qa import cfg, merge_cfg_from_file

parser.add_argument('--cfg', default='configs/dynamic/dynamic_{}.yaml'.format(dataset_name))
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--entropy_weight', type=float, default=0.0)
parser.add_argument('--visualize_every', type=int, default=10)
args = parser.parse_args()

merge_cfg_from_file(args.cfg)
assert cfg.exp_name == os.path.basename(args.cfg).replace('.yaml', '')

# Device configuration
use_cuda = torch.cuda.is_available()
gpu_ids = cfg.gpu_id
torch.backends.cudnn.enabled = True
default_gpu_device = gpu_ids[0]
torch.cuda.set_device(default_gpu_device)
device = torch.device("cuda" if use_cuda else "cpu")

# Experiment configuration
exp_dir = cfg.exp_dir
exp_name = cfg.exp_name
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

output_dir = os.path.join(exp_dir, exp_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# cfg_file_save = os.path.join(output_dir, 'cfg.json')
# json.dump(cfg, open(cfg_file_save, 'w'))
#
# sample_dir = os.path.join(output_dir, 'eval_gen_samples')
# if not os.path.exists(sample_dir):
#     os.makedirs(sample_dir)
# sample_subdir_format = '%s_samples_%d'
#
# sent_dir = os.path.join(output_dir, 'eval_sents')
# if not os.path.exists(sent_dir):
#     os.makedirs(sent_dir)
# sent_subdir_format = '%s_sents_%d'
#
# snapshot_dir = os.path.join(output_dir, 'snapshots')
# if not os.path.exists(snapshot_dir):
#     os.makedirs(snapshot_dir)
# snapshot_file_format = '%s_checkpoint_%d.pt'

# train_logger = Logger(cfg, output_dir, is_train=True)
# val_logger = Logger(cfg, output_dir, is_train=False)

with open(cfg.data.vocab_json, 'r') as j:
    word_map = json.load(j)

vocab_size = len(word_map)

# Create model
change_detector = ChangeDetectorDoubleAttDyn(cfg, vocab_size)
change_detector.to(device)

speaker = DynamicSpeaker(cfg)
speaker.to(device)

spatial_info = AddSpatialInfo()
spatial_info.to(device)

print(change_detector)
print(speaker)
print(spatial_info)

# with open(os.path.join(output_dir, 'model_print'), 'w') as f:
#     print(change_detector, file=f)
#     print(speaker, file=f)
#     print(spatial_info, file=f)

# Data loading part
train_dataset, train_loader = create_dataset(cfg, 'train')
val_dataset, val_loader = create_dataset(cfg, 'val')
train_size = len(train_dataset)
val_size = len(val_dataset)

print(f'training dataset size', train_size)
print(f'validation dataset size', val_size)

# Define loss function and optimizer
lang_criterion = LanguageModelCriterion().to(device)
criterion_ce = nn.CrossEntropyLoss(reduction='sum').to(device)
entropy_criterion = EntropyLoss().to(device)
all_params = list(change_detector.parameters()) + list(speaker.parameters())
optimizer = build_optimizer(all_params, cfg)
# named_parameters = list(change_detector.named_parameters())
# parameters = [
#     {'params': [p for n, p in named_parameters if 'visualBERT' in n], 'lr': 1e-4},
#     {'params': [p for n, p in named_parameters if 'visualBERT' not in n], 'lr': 2e-3},
# ]
epochs = 150  # number of epochs to train for (if early stopping is not triggered)
# t_total = int(len(train_loader) * epochs)
# optimizer = BertAdam(parameters,
#                               lr=2e-3,
#                              warmup=0.1,
#                              t_total=t_total)
# lr_scheduler = optimizer
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=cfg.train.optim.step_size,
    gamma=cfg.train.optim.gamma)

# Train loop
t = 0
epoch = 0

set_mode('train', [change_detector, speaker])
ss_prob = speaker.ss_prob

# Training parameters
start_epoch = 0
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
checkpoint = None  # path to checkpoint, None if none


if checkpoint is not None:
    print('load checkpoint')
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    best_bleu4 = checkpoint['bleu-4']
    # decoder.load_state_dict(checkpoint['decoder'])


def train(train_loader, epoch):
    set_mode('train', [change_detector, speaker])
    # lr_scheduler.step()
    batch_time = AverageMeter()
    speaker_loss_avg = AverageMeter()
    total_loss_avg = AverageMeter()
    total_loss = AverageMeter()
    if epoch > cfg.train.scheduled_sampling_start and cfg.train.scheduled_sampling_start >= 0:
        frac = (epoch - cfg.train.scheduled_sampling_start) // cfg.train.scheduled_sampling_increase_every
        ss_prob_prev = ss_prob
        ss_prob = min(cfg.train.scheduled_sampling_increase_prob * frac,
                      cfg.train.scheduled_sampling_max_prob)
        speaker.ss_prob = ss_prob
        if ss_prob_prev != ss_prob:
            print('Updating scheduled sampling rate: %.4f -> %.4f' % (ss_prob_prev, ss_prob))

    for i, batch in enumerate(train_loader, epoch):
        var_params = {
            'requires_grad': False,
        }
        iter_start_time = time.time()
        visual_embs, visual_token_type_ids, visual_attention_mask, question, question_length, answer, answer_length, mask, qt, qc, answers_ids, answers_token_type_ids, answers_mask = batch
        visual_embs = Variable(visual_embs.cuda(), **var_params)
        visual_token_type_ids = Variable(visual_token_type_ids.cuda(), **var_params)
        visual_attention_mask = Variable(visual_attention_mask.cuda(), **var_params)
        question = Variable(question.cuda(), **var_params)
        question_length = Variable(question_length.cuda(), **var_params)
        answer = Variable(answer.cuda(), **var_params)
        answer_length = Variable(answer_length.cuda(), **var_params)
        mask = mask.to(device)
        qt = qt.to(device)
        qc = qc.to(device)
        answers_ids, answers_token_type_ids, answers_mask = Variable(answers_ids.cuda(), **var_params), Variable(
            answers_token_type_ids.cuda(), **var_params), Variable(answers_mask.cuda(), **var_params)

        question_length = (question_length - 1).squeeze().cpu().numpy().tolist()

        batch_size = visual_embs.shape[0]

        # d_feats, nsc_feats, sc_feats, \
        # labels, no_chg_labels, masks, no_chg_masks, aux_labels_pos, aux_labels_neg, \
        # d_img_paths, nsc_img_paths, sc_img_paths = batch
        #
        # batch_size = d_feats.size(0)
        # labels = labels.squeeze(1)
        # no_chg_labels = no_chg_labels.squeeze(1)
        # masks = masks.squeeze(1).float()
        # no_chg_masks = no_chg_masks.squeeze(1).float()
        #
        # d_feats, nsc_feats, sc_feats = d_feats.to(device), nsc_feats.to(device), sc_feats.to(device)
        # d_feats, nsc_feats, sc_feats = \
        #     spatial_info(d_feats), spatial_info(nsc_feats), spatial_info(sc_feats)
        # labels, masks = labels.to(device), masks.to(device)
        # no_chg_labels, no_chg_masks = no_chg_labels.to(device), no_chg_masks.to(device)
        # aux_labels_pos, aux_labels_neg = aux_labels_pos.to(device), aux_labels_neg.to(device)

        optimizer.zero_grad()


        # chg_pos_att_bef, chg_pos_att_aft, \
        # chg_pos_feat_bef, chg_pos_feat_aft, chg_pos_feat_diff = change_detector(d_feats, sc_feats)
        # chg_neg_att_bef, chg_neg_att_aft, \
        # chg_neg_feat_bef, chg_neg_feat_aft, chg_neg_feat_diff = change_detector(d_feats, nsc_feats)

        chg_pos_att_bef, chg_pos_att_aft, chg_pos_feat_bef, chg_pos_feat_aft, chg_pos_feat_diff = change_detector(visual_embs, visual_token_type_ids, visual_attention_mask, answer, qc, answers_ids, answers_token_type_ids, answers_mask)

        speaker_output_pos = speaker._forward(chg_pos_feat_bef,
                                              chg_pos_feat_aft,
                                              chg_pos_feat_diff,
                                              seq=question, question_types=qt, question_categories=qc, encoded_answers=answer)
        dynamic_atts = speaker.get_module_weights()  # (batch, seq_len, 3)

        # speaker_output_neg = speaker._forward(chg_neg_feat_bef,
        #                                       chg_neg_feat_aft,
        #                                       chg_neg_feat_diff,
        #                                       no_chg_labels)
        speaker_loss_1 = lang_criterion(speaker_output_pos, question[:, 1:], mask[:, 1:])
        # print(speaker_loss_1)

        # speaker_loss = 0.5 * lang_criterion(speaker_output_pos, labels[:, 1:], masks[:, 1:]) + \
        #                0.5 * lang_criterion(speaker_output_neg, no_chg_labels[:, 1:], no_chg_masks[:, 1:])
        # print(question.shape)  # (bs, seq_len)
        # print(speaker_output_pos.shape)  # (bs, seq_len, vocab_size)
        # print(question_length)
        targets = pack_padded_sequence(question[:, 1:], question_length, batch_first=True, enforce_sorted=False).data
        outputs = pack_padded_sequence(speaker_output_pos[:, 1:, :], question_length, batch_first=True, enforce_sorted=False).data
        speaker_loss = criterion_ce(outputs, targets)

        entropy_loss = -args.entropy_weight * entropy_criterion(dynamic_atts, mask)
        att_sum = (chg_pos_att_bef.sum() + chg_pos_att_aft.sum())
        loss =  2.5e-03 * att_sum + entropy_loss + speaker_loss_1
        # total_loss = speaker_loss + 2.5e-03 * att_sum + entropy_loss
        # total_loss_val = total_loss.item()
        # speaker_loss_avg.update(speaker_loss_val, 2 * batch_size)
        # total_loss_avg.update(total_loss_val, 2 * batch_size)
        # print('sum_len:', sum(question_length))
        total_loss.update(loss.item(), sum(question_length))

        stats = {}
        # stats['entropy_loss'] = entropy_loss.item()
        # stats['speaker_loss'] = speaker_loss_val
        # stats['avg_speaker_loss'] = speaker_loss_avg.avg
        # stats['total_loss'] = total_loss_val
        # stats['avg_total_loss'] = total_loss_avg.avg


        #results, sample_logprobs = model(d_feats, q_feats, labels, cfg=cfg, mode='sample')
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - iter_start_time)

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          loss=total_loss))


def validation(val_loader, t):
    epoch = t
    print('Running eval at iter %d' % t)
    set_mode('eval', [change_detector, speaker])
    losses = AverageMeter()
    batch_time = AverageMeter()
    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    with torch.no_grad():
        idx_to_word = train_dataset.get_idx_to_word()

        # if args.visualize:
        #     sample_subdir_path = sample_subdir_format % (exp_name, t)
        #     sample_save_dir = os.path.join(sample_dir, sample_subdir_path)
        #     if not os.path.exists(sample_save_dir):
        #         os.makedirs(sample_save_dir)
        # sent_subdir_path = sent_subdir_format % (exp_name, t)
        # sent_save_dir = os.path.join(sent_dir, sent_subdir_path)
        # if not os.path.exists(sent_save_dir):
        #     os.makedirs(sent_save_dir)
        # # result_sents_pos = {}
        # # result_sents_neg = {}
        for val_i, val_batch in enumerate(val_loader):
            test_iter_start_time = time.time()
            # d_feats, nsc_feats, sc_feats, \
            # labels, no_chg_labels, masks, no_chg_masks, aux_labels_pos, aux_labels_neg, \
            # d_img_paths, nsc_img_paths, sc_img_paths = val_batch
            #
            # val_batch_size = d_feats.size(0)
            var_params = {
                'requires_grad': False,
            }

            visual_embs, visual_token_type_ids, visual_attention_mask, question, question_length, answer, answer_length, _, qt, qc, answers_ids, answers_token_type_ids, answers_mask = val_batch
            visual_embs = visual_embs.to(device)
            visual_token_type_ids = Variable(visual_token_type_ids.cuda(), **var_params)
            visual_attention_mask = Variable(visual_attention_mask.cuda(), **var_params)
            question = question.to(device)
            question_length = question_length.to(device)
            answer = answer.to(device)
            answer_length = answer_length.to(device)
            qt = qt.to(device)
            qc = qc.to(device)

            answers_ids, answers_token_type_ids, answers_mask = Variable(answers_ids.cuda(), **var_params), Variable(answers_token_type_ids.cuda(), **var_params), Variable(answers_mask.cuda(), **var_params)

            batch_size = visual_embs.shape[0]
            question_length = (question_length - 1).squeeze().cpu().numpy().tolist()

            # d_feats, nsc_feats, sc_feats = d_feats.to(device), nsc_feats.to(device), sc_feats.to(device)
            # d_feats, nsc_feats, sc_feats = \
            #     spatial_info(d_feats), spatial_info(nsc_feats), spatial_info(sc_feats)
            # labels, masks = labels.to(device), masks.to(device)
            # no_chg_labels, no_chg_masks = no_chg_labels.to(device), no_chg_masks.to(device)
            # aux_labels_pos, aux_labels_neg = aux_labels_pos.to(device), aux_labels_neg.to(device)

            chg_pos_att_bef, chg_pos_att_aft, chg_pos_feat_bef, chg_pos_feat_aft, chg_pos_feat_diff = change_detector(visual_embs, visual_token_type_ids, visual_attention_mask, answer, qc, answers_ids, answers_token_type_ids, answers_mask)

            seq, seq_pro = speaker._sample(chg_pos_feat_bef,
                                                    chg_pos_feat_aft,
                                                    chg_pos_feat_diff,
                                                    seq=question, question_types=qt, question_categories=qc, encoded_answers=answer, cfg=cfg)

            # gen_sents_pos = decode_sequence(idx_to_word, seq)

            # print('gen_sents_pos.shape', gen_sents_pos.shape)



            # seq = speaker._forward(chg_pos_feat_bef,
            #                                         chg_pos_feat_aft,
            #                                         chg_pos_feat_diff,
            #                                         question)
            # print(seq_pro.shape)
            outputs_copy = seq.clone()
            # outputs = pack_padded_sequence(seq[:, 1:, :], question_length, batch_first=True,
            #                                enforce_sorted=False).data
            # targets = pack_padded_sequence(question[:, 1:], question_length, batch_first=True, enforce_sorted=False).data

            # loss = criterion_ce(outputs, targets)
            # losses.update(loss.item(), sum(question_length))
            batch_time.update(time.time() - test_iter_start_time)

            # if val_i % print_freq == 0:
            #     print('Validation: [{0}/{1}]\t'
            #           'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(val_i, len(val_loader), batch_time=batch_time, loss=losses))

            if val_i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(val_i, len(val_loader), batch_time=batch_time))

            # References
            for j in range(question.shape[0]):
                img_ques = question[j].cpu().numpy().tolist()
                img_sentence_gts = [w for w in img_ques if
                                    w not in {word_map['<START>'], word_map['<END>'], word_map['<NULL>']}]
                references.append(img_sentence_gts)

            # Hypotheses
            # _, preds = torch.max(outputs_copy, dim=-1)  # [batch_size, len]
            preds = outputs_copy.tolist()
            hypotheses.extend(preds)
            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores

        references_2 = []
        hypotheses_2 = []
        idx2word = {index: word for word, index in word_map.items()}

        # calculate bleu

        for r in tqdm(references):
            words = [idx2word[i] for i in r]
            while '<END>' in words:
                words.remove('<END>')
            while '<START>' in words:
                words.remove('<START>')
            while '<NULL>' in words:
                words.remove('<NULL>')
            references_2.append([' '.join(words)])

        for r in tqdm(hypotheses):
            words = [idx2word[i] for i in r]
            while '<END>' in words:
                words.remove('<END>')
            while '<START>' in words:
                words.remove('<START>')
            while '<NULL>' in words:
                words.remove('<NULL>')
            hypotheses_2.append([' '.join(words)])

        gen = {i: s for i, s in enumerate(hypotheses_2)}
        ref = {i: s for i, s in enumerate(references_2)}
        print(hypotheses_2[:5])
        print(references_2[:5])

        bleu = metric(ref, gen)
        # print(
        #     '\n * LOSS - {loss.avg:.3f}\n'.format(
        #         loss=losses))
        # print('bleu: %s, cider: %.6s, meteor: %.6s, rouge: %.6s' % (bleu, None, None, None))
        print('bleu: %s' % bleu)
        return bleu[-1][-1], references, hypotheses

            # pos_dynamic_atts = speaker.get_module_weights().detach().cpu().numpy()  # (batch, seq_len, 3)

            # speaker_output_neg, _ = speaker._sample(chg_neg_feat_bef,
            #                                         chg_neg_feat_aft,
            #                                         chg_neg_feat_diff,
            #                                         no_chg_labels, cfg)

            # neg_dynamic_atts = speaker.get_module_weights().detach().cpu().numpy() # (batch, seq_len, 3)


            #
            # gen_sents_pos = decode_sequence(idx_to_word, speaker_output_pos)
            # gen_sents_neg = decode_sequence(idx_to_word, speaker_output_neg)

        #     chg_pos_att_bef = chg_pos_att_bef.cpu().numpy()
        #     chg_pos_att_aft = chg_pos_att_aft.cpu().numpy()
        #
        #     chg_neg_att_bef = chg_neg_att_bef.cpu().numpy()
        #     chg_neg_att_aft = chg_neg_att_aft.cpu().numpy()
        #     dummy = np.ones_like(chg_pos_att_bef)
        #
        #     for val_j in range(speaker_output_pos.size(0)):
        #         gts = decode_sequence(idx_to_word, labels[val_j][:, 1:])
        #         gts_neg = decode_sequence(idx_to_word, no_chg_labels[val_j][:, 1:])
        #         if args.visualize and val_j % args.visualize_every == 0:
        #             visualize_att(d_img_paths[val_j], sc_img_paths[val_j],
        #                           chg_pos_att_bef[val_j], dummy[val_j], chg_pos_att_aft[val_j],
        #                           pos_dynamic_atts[val_j], gen_sents_pos[val_j], gts,
        #                           -1, -1, sample_save_dir, 'sc_')
        #             visualize_att(d_img_paths[val_j], nsc_img_paths[val_j],
        #                           chg_neg_att_bef[val_j], dummy[val_j], chg_neg_att_aft[val_j],
        #                           neg_dynamic_atts[val_j], gen_sents_neg[val_j], gts_neg,
        #                           -1, -1, sample_save_dir, 'nsc_')
        #         sent_pos = gen_sents_pos[val_j]
        #         sent_neg = gen_sents_neg[val_j]
        #         image_id = d_img_paths[val_j].split('_')[-1]
        #         result_sents_pos[image_id] = sent_pos
        #         result_sents_neg[image_id + '_n'] = sent_neg
        #         message = '%s results:\n' % d_img_paths[val_j]
        #         message += '\t' + sent_pos + '\n'
        #         message += '----------<GROUND TRUTHS>----------\n'
        #         for gt in gts:
        #             message += gt + '\n'
        #         message += '===================================\n'
        #         message += '%s results:\n' % nsc_img_paths[val_j]
        #         message += '\t' + sent_neg + '\n'
        #         message += '----------<GROUND TRUTHS>----------\n'
        #         for gt in gts_neg:
        #             message += gt + '\n'
        #         message += '===================================\n'
        #         print(message)
        #
        #

        # result_save_path_pos = os.path.join(sent_save_dir, 'sc_results.json')
        # result_save_path_neg = os.path.join(sent_save_dir, 'nsc_results.json')
        # coco_gen_format_save(result_sents_pos, result_save_path_pos)
        # coco_gen_format_save(result_sents_neg, result_save_path_neg)



while t < cfg.train.max_iter:
    epoch += 1
    print('Starting epoch %d' % epoch)
    # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
    if epochs_since_improvement == 10:
        break

    # Each batch training
    train(train_loader, epoch)
    t += 1
    speaker_state = speaker.state_dict()
    chg_det_state = change_detector.state_dict()

    # if epoch % cfg.train.snapshot_interval == 0:
    #     speaker_state = speaker.state_dict()
    #     chg_det_state = change_detector.state_dict()
    #     checkpoint = {
    #         'change_detector_state': chg_det_state,
    #         'speaker_state': speaker_state,
    #         'model_cfg': cfg
    #     }
        # save_path = os.path.join(snapshot_dir,
        #                          snapshot_file_format % (exp_name, t))
        # save_checkpoint(checkpoint, save_path)

    recent_bleu4, references, hypotheses = validation(val_loader, t)
    # Check if there was an improvement
    is_best = recent_bleu4 > best_bleu4
    best_bleu4 = max(recent_bleu4, best_bleu4)
    if not is_best:
        epochs_since_improvement += 1
        print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))

    else:
        epochs_since_improvement = 0
        print('save best checkpoint')
        save_checkpoint(None, epoch, epochs_since_improvement, chg_det_state, speaker_state, lr_scheduler, recent_bleu4,
                        is_best, cfg)
        idx2word = {index: word for word, index in word_map.items()}

        print('writing best file')
        with open('./results/{}/true_10clips.txt'.format(dataset_name), 'w') as f:
            for r in tqdm(references):
                words = [idx2word[i] for i in r]
                while '<END>' in words:
                    words.remove('<END>')
                while '<START>' in words:
                    words.remove('<START>')
                while '<NULL>' in words:
                    words.remove('<NULL>')
                f.write(' '.join(words) + '\n')

        with open('./results/{}/final_pred_10clips.txt'.format(dataset_name), 'w') as f:
            for r in tqdm(hypotheses):
                words = [idx2word[i] for i in r]
                while '<END>' in words:
                    words.remove('<END>')
                while '<START>' in words:
                    words.remove('<START>')
                while '<NULL>' in words:
                    words.remove('<NULL>')
                f.write(' '.join(words) + '\n')
