# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import copy
import math
import warnings
from typing import Tuple, List, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .backbone import build_backbone
from .deformable_detr import PostProcess
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class BatchedEmbeddings:
    """Set of embeddings with their respective frame number and tracking id"""

    def __init__(self, batch_size: int, num_queries: int, embed_dim: int, max_prev_frames: int = 5,
                 objectness_threshold: float = 0.8, mixed_precision: bool = False):
        self.batch_size = batch_size
        self.max_embedds_per_frame = num_queries
        self.max_prev_frames = max_prev_frames
        max_num_embeddings = num_queries * max_prev_frames

        self.embeddings = torch.zeros((batch_size, max_num_embeddings, embed_dim),
                                      dtype=torch.float if not mixed_precision else torch.half)
        self.frame_numbers = torch.full((batch_size, max_num_embeddings), fill_value=-1, dtype=torch.long)
        self.tracking_ids = torch.zeros((batch_size, max_num_embeddings), dtype=torch.int)

        self.mask = torch.zeros((batch_size, max_num_embeddings), dtype=torch.bool)
        self.nb_items = torch.zeros((batch_size,), dtype=torch.int32)

        if objectness_threshold == 0.:
            warnings.warn("An objectness threshold of 0 keeps all predictions,"
                          "objectness threshold should be above 0 to discard background predictions")

        self.objectness_threshold = objectness_threshold

    def get_embeddings(self) -> Tensor:
        r"""Returns the embeddings
        :return: embeddings of shape (batch_size, num_embeddings, embed_dim)
        """
        m = self.nb_items.max()  # crop to the size of the largest batch element
        return self.embeddings[:, :m] * self.mask[:, :m, None]

    def get_frame_numbers(self) -> Tensor:
        r"""Returns the frame numbers
        :return: frame numbers of shape (batch_size, num_embeddings)
        """
        m = self.nb_items.max()  # crop to the size of the largest batch element
        return self.frame_numbers[:, :m] * self.mask[:, :m]

    def get_tracking_ids(self) -> Tensor:
        r"""Returns the tracking ids
        :return: tracking ids of shape (batch_size, num_embeddings)
        """
        m = self.nb_items.max()  # crop to the size of the largest batch element
        return self.tracking_ids[:, :m] * self.mask[:, :m]

    def get_mask(self) -> Tensor:
        r"""Returns the mask
        :return: mask of shape (batch_size, num_embeddings)
        """
        m = self.nb_items.max()  # crop to the size of the largest batch element
        return self.mask[:, :m]

    def reset_memory(self):
        r"""Resets the memory of the embeddings"""
        if self.max_prev_frames == 0:
            return

        self.mask[:] = False
        self.nb_items[:] = 0
        self.frame_numbers[:] = -1

    def reset_memory_at(self, batch_mask: Tensor):
        r"""Resets the memory of the embeddings at the provided batch indexes
        :param batch_mask: Tensor of shape (batch_size, ) with True for the batch indexes to reset
        """
        if self.max_prev_frames == 0:
            return

        assert batch_mask.shape[0] == self.batch_size, "wrong batch size"
        self.mask[batch_mask] = False
        self.nb_items[batch_mask] = 0
        self.frame_numbers[batch_mask] = -1

    def _clear_memory(self, current_frames: Tensor):
        r"""Discards embeddings that are older than (current_frames - self.max_prev_frames)
        :param current_frames: Tensor of shape (batch_size, ) with the current frame number for every batch element
        """
        from_edit_mask = (self.frame_numbers > current_frames[:, None] - self.max_prev_frames) * self.mask
        self.mask[:] = False  # first re-initialize the mask
        for b, lim in enumerate(from_edit_mask.sum(dim=1)):
            self.mask[b, :lim] = True

        self.embeddings[self.mask] = self.embeddings[from_edit_mask]
        self.frame_numbers[self.mask] = self.frame_numbers[from_edit_mask]
        self.tracking_ids[self.mask] = self.tracking_ids[from_edit_mask]
        self.nb_items = self.mask.sum(dim=1)

        self.frame_numbers[self.mask is False] = -1

    def add_to_memory(self, pred_embedds: Tensor, pred_classes: Tensor, pred_ids: Tensor, current_frames: Tensor) -> None:
        r"""Adds the predicted embeddings to the memory
        :param pred_embedds: Predicted embeddings of shape (batch_size, num_queries, embed_dim)
        :param pred_classes: Predicted logits of shape (batch_size, num_queries, num_classes)
        :param pred_ids: Predicted tracking ids of shape (batch_size, num_queries)
        :param current_frames: Tensor of shape (batch_size, ) with the current frame number for every batch element
        """
        if self.max_prev_frames == 0:
            return

        # clear memory of embeddings that are too old
        self._clear_memory(current_frames)

        # compute the maximum score for each class and only keep predictions with score above threshold
        max_score, _ = torch.max(pred_classes.sigmoid(), dim=2)
        pred_mask = (max_score >= self.objectness_threshold)  # only keep predictions with score above threshold
        pred_nb_items = pred_mask.sum(dim=1)  # number of items to add to memory

        # update the frame numbers
        for b in range(self.frame_numbers.shape[0]):
            self.frame_numbers[b, self.nb_items[b]:(self.nb_items[b] + pred_nb_items[b])] = current_frames[b]

        # update the embeddings, tracking ids and mask
        self.embeddings[self.frame_numbers == current_frames[:, None]] = pred_embedds[pred_mask].data
        self.mask[self.frame_numbers == current_frames[:, None]] = True
        self.tracking_ids[self.frame_numbers == current_frames[:, None]] = pred_ids[pred_mask].data

        # update the number of items
        self.nb_items += pred_nb_items

    def __len__(self):
        """Returns the total number of elements"""
        return self.mask.sum()

    def __eq__(self, other):
        return torch.equal(self.embeddings, other.embeddings) \
               * torch.equal(self.mask, other.mask) \
               * torch.equal(self.frame_numbers, other.frame_numbers) \
               * torch.equal(self.tracking_ids, other.tracking_ids) \

    def to(self, devie):
        self.embeddings = self.embeddings.to(devie)
        self.frame_numbers = self.frame_numbers.to(devie)
        self.tracking_ids = self.tracking_ids.to(devie)
        self.mask = self.mask.to(devie)
        self.nb_items = self.nb_items.to(devie)


class TrackingHead(nn.Module):
    r"""Assigns tracking ids by maximizing a cost matrix
    :param tracking_threshold: similarity threshold for assigning objects to the same track
    """

    def __init__(self, tracking_threshold: float = 0.5):
        super().__init__()
        self.tracking_threshold = tracking_threshold
        self.id_count = 1  # IDs are provided using 1-indexing

    def reset_id_count(self):
        self.id_count = 1

    @torch.no_grad()
    def forward(self, tracking_costs: list, tracking_ids: list, pred_mask: Tensor) -> Tensor:
        r"""
        :param tracking_costs: list of Tensors, tracking_costs[b] of shape = (M[b], N[b])
            or (0, 0) if no previous embeddings
        :param tracking_ids: list of Tensors, tracking_ids[b] of shape = M[b]
        :param pred_mask: Tensor of shape (batch_size, num_queries)
        :return: pred_ids: Tensor of shape (batch_size, num_queries)
        """
        device = pred_mask.device
        batch_size, num_queries = pred_mask.shape

        m = max([c.shape[1] for c in tracking_costs])
        if m != 0:
            pred_ids = torch.zeros(batch_size, num_queries, dtype=torch.int, device=device)
            queries_idx = torch.arange(num_queries, dtype=torch.long, device=device)
            for b in range(batch_size):
                pred_ids[b][pred_mask[b]] = -1
                if pred_mask[b].sum() > 0 and tracking_ids[b].shape[0] > 0:
                    # pad the tracking costs and IDs with the tracking threshold
                    threshold_costs = torch.full((tracking_costs[b].shape[1], tracking_costs[b].shape[1]),
                                                 fill_value=(self.tracking_threshold + 1))
                    padded_tracking_costs = torch.vstack([tracking_costs[b].cpu(), threshold_costs])
                    threshold_ids = torch.full((tracking_costs[b].shape[1],), fill_value=-1, dtype=torch.int,
                                               device=device)
                    padded_tracking_ids = torch.hstack([tracking_ids[b], threshold_ids])

                    # compute bipartite matching with Hungarian algorithm
                    row_ind, col_ind = linear_sum_assignment(padded_tracking_costs.permute(1, 0), maximize=True)
                    row_ind = torch.as_tensor(row_ind, dtype=torch.long, device=device)
                    col_ind = torch.as_tensor(col_ind, dtype=torch.long, device=device)

                    true_row_ind = queries_idx[pred_mask[b]][row_ind]
                    pred_ids[b][true_row_ind] = padded_tracking_ids[col_ind]

            # give a new unique id to new detections (i.e. non-background unassigned predictions)
            edit_mask = (pred_ids == -1)
            new_ids_count = torch.sum(edit_mask)
            pred_ids[edit_mask] = torch.arange(self.id_count, self.id_count + new_ids_count,
                                               out=pred_ids[edit_mask])
            self.id_count += new_ids_count
        else:
            # if no previous embeddings, assign new IDs
            batch_size, num_queries = pred_mask.shape[0], pred_mask.shape[1]

            num_ids_count = pred_mask.sum()
            pred_ids = torch.zeros((batch_size, num_queries), dtype=torch.int, device=device)
            pred_ids[pred_mask] = torch.arange(self.id_count, self.id_count + num_ids_count, out=pred_ids[pred_mask])
            self.id_count += num_ids_count

        return pred_ids


class ContrasTR(nn.Module):
    """ This is the ContrasTR module that performs multiple object tracking """

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, mixed_selection=False,
                 objectness_threshold: float = 0.5, tracking_threshold: int = 0.5):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
            mixed_selection: a trick for Deformable DETR two stage
            objectness_threshold: threshold for adding new objects to the memory
            tracking_threshold: threshold for assigning objects to the same track
        """
        super().__init__()

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.track_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        elif mixed_selection:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.mixed_selection = mixed_selection

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # Tracking
        self.objectness_threshold = objectness_threshold
        self.tracking_head = TrackingHead(tracking_threshold=tracking_threshold)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            self.track_embed = _get_clones(self.track_embed, transformer.decoder.num_layers)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.track_embed = nn.ModuleList([self.track_embed for _ in range(transformer.decoder.num_layers)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    @staticmethod
    def compute_appearance_scores(pred_mask: Tensor, memory_mask: Tensor, pred_tracking_embeds: Tensor,
                                  memory_tracking_embeds: Tensor, memory_ids: Tensor) -> Tuple[List, List]:
        """
        Compute appearance scores between predicted embeddings and memory embeddings
        :param pred_mask: Tensor of shape (batch_size, num_queries) with True for valid predictions
        :param memory_mask: Tensor of shape (batch_size, num_embeddings) with True for valid memory embeddings
        :param pred_tracking_embeds: Tensor of shape (batch_size, num_queries, embed_dim) with predicted embeddings
        :param memory_tracking_embeds: Tensor of shape (batch_size, num_embeddings, embed_dim) with memory embeddings
        :param memory_ids: Tensor of shape (batch_size, num_embeddings)
        :return
            appearance_scores: List of Tensors of shape (num_unique_ids, num_queries) with appearance scores
            ids_from_appearance: List of Tensors of shape (num_unique_ids, ) with unique tracking ids
        """
        batch_size = pred_mask.shape[0]
        device = pred_mask.device

        if memory_mask.sum() == 0:
            return [torch.zeros((0, 0), device=device) for _ in range(batch_size)], \
                [torch.zeros((0, 0), device=device) for _ in range(batch_size)]

        # compute maximum cosine similarity between predicted and memory embeddings
        pred_tracking_embeds_norm = F.normalize(pred_tracking_embeds, dim=2, p=2)
        memory_tracking_embeds_norm = F.normalize(memory_tracking_embeds, dim=2, p=2)
        similarity = torch.einsum('bid,bkd->bik', memory_tracking_embeds_norm, pred_tracking_embeds_norm)

        appearance_scores = list()
        ids_from_appearance = list()
        for b in range(batch_size):
            # get unique tracking ids from memory
            masked_memory_ids = memory_ids[b][memory_mask[b]]
            unique_tracking_ids = torch.unique(masked_memory_ids)

            if pred_mask[b].sum() > 0 and masked_memory_ids.shape[0] > 0:
                # get the highest similarity score for each unique tracking id in memory
                masked_similarity = similarity[b][memory_mask[b], :][:, pred_mask[b]]
                reduced_similarity = torch.vstack(
                    [masked_similarity[masked_memory_ids == id, :].amax(dim=0) for id in unique_tracking_ids],
                )
                appearance_scores.append(reduced_similarity + 1)  # from [-1, 1] to [0, 2]
                ids_from_appearance.append(unique_tracking_ids)
            else:
                appearance_scores.append(torch.zeros((unique_tracking_ids.shape[0], pred_mask[b].sum()), device=device))
                ids_from_appearance.append(unique_tracking_ids)

        return appearance_scores, ids_from_appearance

    def forward(self, samples: NestedTensor,
                prev_embeddings: Optional[BatchedEmbeddings] = None,
                current_frames: Optional[Tensor] = None):
        """The forward expects a NestedTensor, which is a pair of tensors (images, masks).
                - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
                - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            Additionally, it expects the following inputs for association (optional):
                - prev_embeddings: the embeddings memory from the previous frame
                - current_frames: the current frame number for each batch element
            ...

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "pred_embeddings": The embeddings for all queries, before the tracking head
               - "track_embeddings": The embeddings for all queries, after the tracking head
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
               - "enc_outputs": Optional, only returned when two_stage is activated. It is a dictionnary containing
               - "pred_ids": The tracking ids for all queries (for the last layer only)
                            skipping the association if no previous embeddings are provided
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage or self.mixed_selection:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        outputs_tracking_embeddings = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            tracking_embeddings = self.track_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_tracking_embeddings.append(tracking_embeddings)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_tracking_embeddings = torch.stack(outputs_tracking_embeddings)

        if prev_embeddings is None or current_frames is None:
            # Skip association if no previous embeddings are provided
            pred_ids = torch.full(size=outputs_class[-1].shape[:-1],
                                  fill_value=-1, dtype=torch.int,
                                  device=outputs_class[-1].device)
        else:
            # recover information from previous embeddings
            memory_tracking_embedds = prev_embeddings.get_embeddings()
            memory_mask = prev_embeddings.get_mask()
            memory_tracking_ids = prev_embeddings.get_tracking_ids()

            # only keep "non-background" predictions
            pred_mask = (torch.amax(outputs_class[-1].sigmoid(), dim=2) >= self.objectness_threshold)

            # compute appearance scores
            pred_tracking_embedds = outputs_tracking_embeddings[-1].clone()
            appearance_scores, tracking_ids_from_appearance = self.compute_appearance_scores(
                pred_mask, memory_mask, pred_tracking_embedds, memory_tracking_embedds, memory_tracking_ids
            )

            # predict tracking ids (for the last layer only)
            pred_ids = self.tracking_head(appearance_scores, tracking_ids_from_appearance, pred_mask)

            # update the embeddings memory
            prev_embeddings.add_to_memory(outputs_tracking_embeddings[-1], outputs_class[-1], pred_ids, current_frames)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
               'pred_embeddings': hs[-1], 'track_embeddings': outputs_tracking_embeddings[-1],
               'reference_points': inter_references[-1], 'pred_ids': pred_ids}
        # FIXME: inter_references[-1] always ok ? (both when return_intermediate = True/False ?)
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, hs, outputs_tracking_embeddings)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, hs, outputs_tracking_embeddings):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_embeddings': c, 'track_embeddings': d}
                for a, b, c, d in
                zip(outputs_class[:-1], outputs_coord[:-1], hs[:-1], outputs_tracking_embeddings[:-1])]


class InstanceLevelContrastiveLoss(nn.Module):
    r"""Contrastive loss for instance-level embeddings
    :param tau: temperature parameter for the similarity
    """
    def __init__(self, tau: float = 0.1):
        super(InstanceLevelContrastiveLoss, self).__init__()
        assert tau > 0, "tau should be positive"
        self.tau = tau

    def forward(self, embeddings: Tensor, labels: Tensor):
        """
        :param embeddings: Tensor of shape (num_embeddings, embed_dim)
        :param labels: Tensor of shape (num_embeddings, )
        """
        assert embeddings.shape[0] == labels.shape[0], "Each embedding should have a label"
        num_embeddings = embeddings.shape[0]

        # positive pairs are the pairs of embeddings that have the same label
        positive_pairs = torch.eq(labels[None, :], labels[:, None])  # (num_embeddings, num_embeddings)
        positive_pairs[torch.arange(num_embeddings), torch.arange(num_embeddings)] = False
        pos_indices = positive_pairs.nonzero()  # get the indices of the positive pairs [[i, j], [i, k], ...]
        num_positives = positive_pairs.sum(dim=1)  # (num_embeddings, )
        if num_positives.sum() == 0:
            warnings.warn("No positive object pairs within the minibatch")

        # compute the similarity matrix
        embeddings = F.normalize(embeddings, dim=1)
        similarities = embeddings @ embeddings.t()
        similarities = (similarities / self.tau)

        # set the similarity of the diagonal to -inf
        similarities[torch.arange(num_embeddings), torch.arange(num_embeddings)] = float('-inf')

        # clone similarities and repeat num_positives times along the batch dimension
        neg_similarities = similarities.clone().repeat_interleave(num_positives, dim=0)
        neg_mask = positive_pairs.clone().repeat_interleave(num_positives, dim=0)
        neg_mask[torch.arange(neg_similarities.shape[0]), pos_indices[:, 1]] = False
        neg_similarities[neg_mask] = float('-inf')

        # compute loss using "log sum exp" formulation for stability
        logsumexp = torch.logsumexp(neg_similarities, dim=1, keepdim=False)
        logprob = - (similarities[pos_indices[:, 0], pos_indices[:, 1]] - logsumexp)

        # average the loss over the positive pairs
        loss = logprob / num_positives[pos_indices[:, 0]]

        # average the loss over the batch elements with positive pairs
        loss_value = self.tau * loss.sum() / torch.clamp((num_positives != 0).sum(), min=1.)

        return loss_value


class SetCriterion(nn.Module):
    """ This class computes the loss for ContrasTR.
    The process happens in two steps:
        1) we compute bipartite matching between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class, box, tracking embeddings, masks)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25,
                 objectness_threshold: float = 0.8, temperature=0.07):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.objectness_threshold = objectness_threshold

        # contrastive loss
        self.contrastive_loss = InstanceLevelContrastiveLoss(tau=temperature)

    def loss_labels(self, outputs, targets, indices, num_boxes, decoder_layer, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)  # B,N
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout,
                                            device=src_logits.device)  # B,N,(C+1)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]  # B,N,C

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, decoder_layer_idx):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (torch.amax(pred_logits.sigmoid(), dim=2) >= self.objectness_threshold).sum(dim=1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, decoder_layer_idx):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes, decoder_layer_idx):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_contrastive(self, outputs, targets, indices, num_boxes, decoder_layer_idx):
        """Compute the contrastive loss on tracking embeddings (synchronized across GPUs)
        """
        assert 'track_embeddings' in outputs
        device = outputs['track_embeddings'].device

        # collect the embeddings that are assigned to a target instance
        embeddings = [outputs['track_embeddings'][b, index_i, :] for b, (index_i, index_j) in enumerate(indices)]
        embeddings = torch.cat(embeddings)

        # generate unique instance ids for each embedding (video_id * 100000 + instance_id)
        embeddings_ids = torch.tensor([
            batch_targets['video_id'].item() * 100000 + instance_id.item()
            for batch_idx, (batch_targets, (_, i)) in enumerate(zip(targets, indices))
            for instance_id in batch_targets['instance_id'][i]
        ], device=device, dtype=torch.int)

        # synchronize between GPUs if necessary
        if is_dist_avail_and_initialized():
            world_size = get_world_size()
            rank = get_rank()

            embed_dim = embeddings.shape[1]

            # broadcasting the number of embeddings to all ranks (to pad the tensors)
            local_num_embed = torch.tensor([embeddings.shape[0]], device=device)
            num_embed_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
            dist.all_gather(num_embed_list, local_num_embed)
            num_embed_list = [int(size.item()) for size in num_embed_list]
            max_num_embed = max(num_embed_list)

            # receiving Tensor from all ranks
            # (we pad the tensor because torch all_gather does not support gathering tensors of different shapes)
            float_type = torch.half if torch.is_autocast_enabled() else torch.float
            embeddings_list = [torch.empty((max_num_embed, embed_dim), device="cuda", dtype=float_type) for _ in num_embed_list]
            if local_num_embed != max_num_embed:
                padding = torch.empty(size=(max_num_embed - local_num_embed, embed_dim), dtype=float_type, device="cuda")
                embeddings = torch.cat((embeddings, padding), dim=0)
            dist.all_gather(embeddings_list, embeddings)
            embeddings_list[rank] = embeddings
            embeddings_list = [embeddings_list[r][:num_embed_list[r]] for r in range(world_size)]
            all_embeddings = torch.cat(embeddings_list)

            embeddings_ids_list = [torch.empty(max_num_embed, device="cuda", dtype=torch.int) for _ in num_embed_list]
            if local_num_embed != max_num_embed:
                padding = torch.empty(size=(max_num_embed - local_num_embed,), dtype=torch.int, device="cuda")
                embeddings_ids = torch.cat((embeddings_ids, padding), dim=0)
            dist.all_gather(embeddings_ids_list, embeddings_ids)
            embeddings_ids_list = [embeddings_ids_list[r][:num_embed_list[r]] for r in range(world_size)]
            all_embeddings_ids = torch.cat(embeddings_ids_list)
        else:
            all_embeddings = embeddings
            all_embeddings_ids = embeddings_ids

        # compute the contrastive loss
        loss_cont = self.contrastive_loss(all_embeddings, all_embeddings_ids)

        losses = {'loss_contrastive': loss_cont}

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, decoder_layer_idx, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'contrastive': self.loss_contrastive
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, decoder_layer_idx, **kwargs)

    def forward(self, outputs, targets, skip_contrastive_loss: bool = False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes],
                                    dtype=torch.half if torch.is_autocast_enabled() else torch.float,
                                    device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            dist.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        if 'aux_outputs' not in outputs:
            last_decoder_layer_idx = 0
        else:
            last_decoder_layer_idx = len(outputs['aux_outputs'])

        losses = {}
        for loss in self.losses:
            kwargs = {}
            if loss == "contrastive" and skip_contrastive_loss:
                continue
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, last_decoder_layer_idx, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    if loss == "contrastive" and skip_contrastive_loss:
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, i, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                if loss == "contrastive":
                    # We don't compute tracking embeddings in the encoder
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, -1, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcessMOT(nn.Module):
    """ This module converts the model's output into the format expected by the mot api"""

    box_convertion = {'xywh': box_ops.box_cxcywh_to_xywh, 'xyxy': box_ops.box_cxcywh_to_xyxy}

    def __init__(self, out_box_type='xywh', absolute_coord=False, remapping={}):
        super().__init__()
        # remapping categories if needed.
        # mapping to -1 to remove a certain cat, re-mapping otherwise, mapping None to keep predictions out of the model
        self.remapping = remapping
        self.out_box_type = out_box_type
        self.absolute_coord = absolute_coord

    @torch.no_grad()
    def relative_to_absolute_predictions(self, boxes, targets):
        target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        # from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        return boxes

    @torch.no_grad()        # FIXME: improve the whole function forward? This function is the same as before
    def relative_to_absolute_targets(self, boxes, targets):

        # from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = targets["orig_size"][0], targets["orig_size"][1]
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
        boxes = boxes * scale_fct[None, :]

        return boxes

    @torch.no_grad()
    def forward(self, outputs, targets, is_test=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        mot_target = [{k: torch.clone(v) for k, v in targets[b].items()} for b in range(len(targets))]

        out_logits, out_bbox, out_ids = outputs['pred_logits'], outputs['pred_boxes'], outputs['pred_ids']

        prob = out_logits.sigmoid()
        scores, labels = torch.max(prob, dim=2)

        boxes = PostProcessMOT.box_convertion[self.out_box_type](out_bbox)

        if self.absolute_coord:
            boxes = self.relative_to_absolute_predictions(boxes, targets)

        # remap and/or discard predictions
        rem_labels = torch.zeros(labels.shape, dtype=labels.dtype, device=labels.device)
        for k, v in self.remapping.items():
            rem_labels[labels == k] = v

        results = []
        for i, (s, l, b, ids) in enumerate(zip(scores, rem_labels, boxes, out_ids)):
            mask = rem_labels >= 0
            results.append({'scores': s[mask[i]],
                            'labels': l[mask[i]],
                            'boxes': b[mask[i]],
                            'ids': ids[mask[i]]})

        # if in test mode, return only the predictions
        if is_test:
            return results

        # remove uninteresting targets and refactor them
        for b in range(len(targets)):
            rem_labels = torch.zeros(targets[b]['labels'].shape, dtype=targets[b]['labels'].dtype,
                                     device=targets[b]['labels'].device)
            for k, v in self.remapping.items():
                rem_labels[targets[b]['labels'] == k] = v
            mask = rem_labels >= 0
            rem_boxes = mot_target[b]["boxes"][mask]
            rem_labels = mot_target[b]["labels"][mask]
            rem_instance_id = mot_target[b]["instance_id"][mask]

            rem_boxes = PostProcessMOT.box_convertion[self.out_box_type](rem_boxes.unsqueeze(0))[0]

            if self.absolute_coord:
                rem_boxes = self.relative_to_absolute_targets(rem_boxes, targets[b])

            mot_target[b]["boxes"] = rem_boxes
            mot_target[b]["labels"] = rem_labels
            mot_target[b]["instance_id"] = rem_instance_id

        return results, mot_target


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    if args.dataset_file == "bdd100k":
        num_classes = 11    # BDD has 10 classes, class 0 will be unused

        no_remapping = {i: i for i in range(num_classes)}  # default remapping -> no remapping

        mot_our_rem = no_remapping.copy()
        mot_our_rem[0] = -1     # discard unused class 0 predictions
        mot_our_rem[9] = -1     # discard traffic light predictions
        mot_our_rem[10] = -1    # discard traffic sign predictions
        mot_our_out_box_type = 'xywh'
        mot_our_out_absolute_coord = False

        mot_off_rem = mot_our_rem.copy()  # mot official categories remapping
        mot_off_out_box_type = 'xyxy'
        mot_off_out_absolute_coord = True
    elif args.dataset_file == "mot17":
        num_classes = 2     # MOT17 has 1 classe, class 0 will be unused

        no_remapping = {i: i for i in range(num_classes)}  # default remapping -> no remapping

        mot_our_rem = no_remapping.copy()
        mot_our_out_box_type = 'xywh'
        mot_our_out_absolute_coord = False

        mot_off_rem = no_remapping.copy()  # mot official categories remapping
        mot_off_rem[0] = -1  # discard unused class 0 predictions
        mot_off_out_box_type = 'xywh'
        mot_off_out_absolute_coord = True
    else:
        raise ValueError(f'unknown dataset {args.dataset_file}')

    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)
    model = ContrasTR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        mixed_selection=args.mixed_selection,
        objectness_threshold=args.objectness_threshold,
        tracking_threshold=args.tracking_threshold,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    if args.contrastive_loss:
        weight_dict["loss_contrastive"] = args.cont_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    if args.contrastive_loss:
        losses += ["contrastive"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha,
                             objectness_threshold=args.objectness_threshold, temperature=args.cont_loss_temp)
    criterion.to(device)
    postprocessors = {
        'bbox': PostProcess(),  # postprocess predictions for COCO APIs
        'mot': PostProcessMOT(out_box_type=mot_our_out_box_type,  # postprocess pred for our MOT evaluator
                              absolute_coord=mot_our_out_absolute_coord,
                              remapping=mot_our_rem),
        'mot_official': PostProcessMOT(out_box_type=mot_off_out_box_type,  # postprocess pred for official MOT evaluator
                                       absolute_coord=mot_off_out_absolute_coord,
                                       remapping=mot_off_rem)
    }
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
