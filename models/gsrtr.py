# ----------------------------------------------------------------------------------------------
# GSRTR Official Code
# Copyright (c) Junhyeong Cho. All Rights Reserved 
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

"""
GSRTR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, accuracy_swig, accuracy_swig_bbox)
from .backbone import build_backbone
from .transformer import build_transformer


class GSRTR(nn.Module):
    """ GSRTR model for Grounded Situation Recognition"""
    def __init__(self, backbone, transformer, num_noun_classes, vidx_ridx):
        """ Initialize the model.
        Parameters:
            - backbone: torch module of the backbone to be used. See backbone.py
            - transformer: torch module of the transformer architecture. See transformer.py
            - num_noun_classes: the number of noun classes
            - vidx_ridx: verb index to role index
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.num_noun_classes = num_noun_classes
        self.vidx_ridx = vidx_ridx
        self.num_role_queries = 190
        self.num_verb_queries = 504

        # hidden dimension for queries and image features
        hidden_dim = transformer.d_model

        # query embeddings
        self.role_query_embed = nn.Embedding(self.num_role_queries, hidden_dim//2)
        self.verb_query_embed = nn.Embedding(self.num_verb_queries, hidden_dim//2)
        self.enc_verb_query_embed = nn.Embedding(1, hidden_dim)

        # 1x1 Conv
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        # classifiers & predictors (for grounded noun prediction)
        self.noun_classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.3),
                                             nn.Linear(hidden_dim*2, self.num_noun_classes))
        self.bbox_predictor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(hidden_dim*2, hidden_dim*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(hidden_dim*2, 4))
        self.bbox_conf_predictor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(hidden_dim*2, 1))
        

    def forward(self, samples, targets=None, inference=False):
        """ 
        Parameters:
               - samples: The forward expects a NestedTensor, which consists of:
                        - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - targets: This has verbs, roles and labels information
               - inference: boolean, used in inference
        Outputs:
               - out: dict of tensors. 'pred_verb', 'pred_noun', 'pred_bbox' and 'pred_bbox_conf' are keys
        """
        MAX_NUM_ROLES = 6
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        batch_size = src.shape[0]
        batch_verb, batch_noun, batch_bbox, batch_bbox_conf = [], [], [], []
        # model prediction
        for i in range(batch_size):
            if not inference:
                verb_pred, rhs, num_roles = self.transformer(self.input_proj(src[i:i+1]), 
                                                            mask[i:i+1], self.enc_verb_query_embed.weight, 
                                                            self.verb_query_embed.weight, self.role_query_embed.weight, 
                                                            pos[-1][i:i+1], self.vidx_ridx, targets=targets[i], inference=inference)
            else:
                verb_pred, rhs, num_roles = self.transformer(self.input_proj(src[i:i+1]), 
                                                            mask[i:i+1], self.enc_verb_query_embed.weight, 
                                                            self.verb_query_embed.weight, self.role_query_embed.weight, 
                                                            pos[-1][i:i+1], self.vidx_ridx, inference=inference)
            noun_pred = self.noun_classifier(rhs)
            noun_pred = F.pad(noun_pred, (0, 0, 0, MAX_NUM_ROLES - num_roles), mode='constant', value=0)[-1].view(1, MAX_NUM_ROLES, self.num_noun_classes)
            bbox_pred = self.bbox_predictor(rhs).sigmoid()
            bbox_pred = F.pad(bbox_pred, (0, 0, 0, MAX_NUM_ROLES - num_roles), mode='constant', value=0)[-1].view(1, MAX_NUM_ROLES, 4)
            bbox_conf_pred = self.bbox_conf_predictor(rhs)
            bbox_conf_pred = F.pad(bbox_conf_pred, (0, 0, 0, MAX_NUM_ROLES - num_roles), mode='constant', value=0)[-1].view(1, MAX_NUM_ROLES, 1)

            batch_verb.append(verb_pred)
            batch_noun.append(noun_pred)
            batch_bbox.append(bbox_pred)
            batch_bbox_conf.append(bbox_conf_pred)

        # outputs
        out = {}
        out['pred_verb'] = torch.cat(batch_verb, dim=0)
        out['pred_noun'] = torch.cat(batch_noun, dim=0)
        out['pred_bbox'] = torch.cat(batch_bbox, dim=0)
        out['pred_bbox_conf'] = torch.cat(batch_bbox_conf, dim=0)

        return out


class LabelSmoothing(nn.Module):
    """ NLL loss with label smoothing """
    def __init__(self, smoothing=0.0):
        """ Constructor for the LabelSmoothing module.
        Parameters:
                - smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SWiGCriterion(nn.Module):
    """ 
    Loss for GSRTR with SWiG dataset, and GSRTR evaluation.
    """
    def __init__(self, weight_dict, SWiG_json_train=None, SWiG_json_eval=None, idx_to_role=None):
        """ 
        Create the criterion.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.loss_function = LabelSmoothing(0.2)
        self.loss_function_verb = LabelSmoothing(0.3)
        self.SWiG_json_train = SWiG_json_train
        self.SWiG_json_eval = SWiG_json_eval
        self.idx_to_role = idx_to_role


    def forward(self, outputs, targets, eval=False):
        """ This performs the loss computation, and evaluation of GSRTR.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             eval: boolean, used in evlauation
        """
        NUM_ANNOTATORS = 3

        # gt verb (value & value-all) acc and calculate noun loss
        assert 'pred_noun' in outputs
        pred_noun = outputs['pred_noun']
        device = pred_noun.device
        batch_size = pred_noun.shape[0]
        batch_noun_loss, batch_noun_acc, batch_noun_correct = [], [], []
        for i in range(batch_size):
            p, t = pred_noun[i], targets[i]
            roles = t['roles']
            num_roles = len(roles)
            role_pred = p[:num_roles]
            role_targ = t['labels'][:num_roles]
            role_targ = role_targ.long()
            acc_res = accuracy_swig(role_pred, role_targ)
            batch_noun_acc += acc_res[1]
            batch_noun_correct += acc_res[0]
            role_noun_loss = []
            for n in range(NUM_ANNOTATORS):
                role_noun_loss.append(self.loss_function(role_pred, role_targ[:, n]))
            batch_noun_loss.append(sum(role_noun_loss))
        noun_loss = torch.stack(batch_noun_loss).mean()
        noun_acc = torch.stack(batch_noun_acc)
        noun_correct = torch.stack(batch_noun_correct)

        # top-1 & top 5 verb acc and calculate verb loss 
        assert 'pred_verb' in outputs
        verb_pred_logits = outputs['pred_verb'].squeeze(1)
        gt_verbs = torch.stack([t['verbs'] for t in targets])
        verb_acc_topk = accuracy(verb_pred_logits, gt_verbs, topk=(1, 5))
        verb_loss = self.loss_function_verb(verb_pred_logits, gt_verbs)
        
        # top-1 & top 5 (value & value-all) acc
        batch_noun_acc_topk, batch_noun_correct_topk = [], []
        for verbs in verb_pred_logits.topk(5)[1].transpose(0, 1):
            batch_noun_acc = []
            batch_noun_correct = []
            for i in range(batch_size):
                v, p, t = verbs[i], pred_noun[i], targets[i]
                if v == t['verbs']:
                    roles = t['roles']
                    num_roles = len(roles)
                    role_pred = p[:num_roles]
                    role_targ = t['labels'][:num_roles]
                    role_targ = role_targ.long()
                    acc_res = accuracy_swig(role_pred, role_targ)
                    batch_noun_acc += acc_res[1]
                    batch_noun_correct += acc_res[0]
                else:
                    batch_noun_acc += [torch.tensor(0., device=device)]
                    batch_noun_correct += [torch.tensor([0, 0, 0, 0, 0, 0], device=device)]
            batch_noun_acc_topk.append(torch.stack(batch_noun_acc))
            batch_noun_correct_topk.append(torch.stack(batch_noun_correct))
        noun_acc_topk = torch.stack(batch_noun_acc_topk)
        noun_correct_topk = torch.stack(batch_noun_correct_topk) # topk x batch x max roles 

        # bbox prediction
        assert 'pred_bbox' in outputs
        assert 'pred_bbox_conf' in outputs
        pred_bbox = outputs['pred_bbox']
        pred_bbox_conf = outputs['pred_bbox_conf'].squeeze(2)
        batch_bbox_acc, batch_bbox_acc_top1, batch_bbox_acc_top5 = [], [], []
        batch_bbox_loss, batch_giou_loss, batch_bbox_conf_loss = [], [], []
        for i in range(batch_size):
            pb, pbc, t = pred_bbox[i], pred_bbox_conf[i], targets[i]
            mw, mh, target_bboxes = t['max_width'], t['max_height'], t['boxes']
            cloned_pb, cloned_target_bboxes = pb.clone(), target_bboxes.clone()
            num_roles = len(t['roles'])
            bbox_exist = target_bboxes[:, 0] != -1
            num_bbox = bbox_exist.sum().item()

            # bbox conf loss
            loss_bbox_conf = F.binary_cross_entropy_with_logits(pbc[:num_roles], 
                                                                bbox_exist[:num_roles].float(), reduction='mean')
            batch_bbox_conf_loss.append(loss_bbox_conf)

            # bbox reg loss and giou loss
            if num_bbox > 0: 
                loss_bbox = F.l1_loss(pb[bbox_exist], target_bboxes[bbox_exist], reduction='none')
                loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(box_ops.swig_box_cxcywh_to_xyxy(pb[bbox_exist], mw, mh, device=device), 
                                                                       box_ops.swig_box_cxcywh_to_xyxy(target_bboxes[bbox_exist], mw, mh, device=device, gt=True)))
                batch_bbox_loss.append(loss_bbox.sum() / num_bbox)
                batch_giou_loss.append(loss_giou.sum() / num_bbox)

            # top1 correct noun & top5 correct nouns
            noun_correct_top1 = noun_correct_topk[0]
            noun_correct_top5 = noun_correct_topk.sum(dim=0)

            # convert coordinates
            pb_xyxy = box_ops.swig_box_cxcywh_to_xyxy(cloned_pb, mw, mh, device=device)
            gt_bbox_xyxy = box_ops.swig_box_cxcywh_to_xyxy(cloned_target_bboxes, mw, mh, device=device, gt=True)
            
            # accuracies
            if not eval:
                batch_bbox_acc += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles, 
                                                     noun_correct[i], bbox_exist, t, self.SWiG_json_train, 
                                                     self.idx_to_role)
                batch_bbox_acc_top1 += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                                                          noun_correct_top1[i], bbox_exist, t, self.SWiG_json_train, 
                                                          self.idx_to_role)
                batch_bbox_acc_top5 += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles, 
                                                          noun_correct_top5[i], bbox_exist, t, self.SWiG_json_train, 
                                                          self.idx_to_role)
            else:
                batch_bbox_acc += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles, 
                                                     noun_correct[i], bbox_exist, t, self.SWiG_json_eval, 
                                                     self.idx_to_role, eval)
                batch_bbox_acc_top1 += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles, 
                                                          noun_correct_top1[i], bbox_exist, t, self.SWiG_json_eval, 
                                                          self.idx_to_role, eval) 
                batch_bbox_acc_top5 += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles, 
                                                          noun_correct_top5[i], bbox_exist, t, self.SWiG_json_eval, 
                                                          self.idx_to_role, eval) 

        if len(batch_bbox_loss) > 0:
            bbox_loss = torch.stack(batch_bbox_loss).mean()
            giou_loss = torch.stack(batch_giou_loss).mean()
        else:
            bbox_loss = torch.tensor(0., device=device)
            giou_loss = torch.tensor(0., device=device)

        bbox_conf_loss = torch.stack(batch_bbox_conf_loss).mean()
        bbox_acc = torch.stack(batch_bbox_acc)
        bbox_acc_top1 = torch.stack(batch_bbox_acc_top1)
        bbox_acc_top5 = torch.stack(batch_bbox_acc_top5)

        out = {}
        # losses 
        out['loss_vce'] = verb_loss
        out['loss_nce'] = noun_loss
        out['loss_bbox'] = bbox_loss
        out['loss_giou'] = giou_loss
        out['loss_bbox_conf'] = bbox_conf_loss

        # All metrics should be calculated per verb and averaged across verbs.
        ## In the dev and test split of SWiG dataset, there are 50 images for each verb (same number of images per verb).
        ### Our implementation is correct to calculate metrics for the dev and test split of SWiG dataset. 
        ### We calculate metrics in this way for simple implementation in distributed data parallel setting. 

        # accuracies (for verb and noun)
        out['noun_acc_top1'] = noun_acc_topk[0].mean()
        out['noun_acc_all_top1'] = (noun_acc_topk[0] == 100).float().mean()*100
        out['noun_acc_top5'] = noun_acc_topk.sum(dim=0).mean()
        out['noun_acc_all_top5'] = (noun_acc_topk.sum(dim=0) == 100).float().mean()*100
        out['verb_acc_top1'] = verb_acc_topk[0]
        out['verb_acc_top5'] = verb_acc_topk[1]
        out['noun_acc_gt'] = noun_acc.mean()
        out['noun_acc_all_gt'] = (noun_acc == 100).float().mean()*100
        out['mean_acc'] = torch.stack([v for k, v in out.items() if 'noun_acc' in k or 'verb_acc' in k]).mean()
        # accuracies (for bbox)
        out['bbox_acc_gt'] = bbox_acc.mean()
        out['bbox_acc_all_gt'] = (bbox_acc == 100).float().mean()*100
        out['bbox_acc_top1'] = bbox_acc_top1.mean()
        out['bbox_acc_all_top1'] = (bbox_acc_top1 == 100).float().mean()*100
        out['bbox_acc_top5'] = bbox_acc_top5.mean()
        out['bbox_acc_all_top5'] = (bbox_acc_top5 == 100).float().mean()*100

        return out


def build(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = GSRTR(backbone,
                  transformer,
                  num_noun_classes=args.num_noun_classes,
                  vidx_ridx=args.vidx_ridx)
    criterion = None

    if not args.inference:
        weight_dict = {'loss_nce': args.noun_loss_coef, 'loss_vce': args.verb_loss_coef, 
                       'loss_bbox':args.bbox_loss_coef, 'loss_giou':args.giou_loss_coef,
                       'loss_bbox_conf':args.bbox_conf_loss_coef}
    
        if not args.test:
            criterion = SWiGCriterion(weight_dict=weight_dict, 
                                      SWiG_json_train=args.SWiG_json_train, 
                                      SWiG_json_eval=args.SWiG_json_dev, 
                                      idx_to_role=args.idx_to_role)
        else:
            criterion = SWiGCriterion(weight_dict=weight_dict, 
                                      SWiG_json_train=args.SWiG_json_train, 
                                      SWiG_json_eval=args.SWiG_json_test, 
                                      idx_to_role=args.idx_to_role)

    return model, criterion
