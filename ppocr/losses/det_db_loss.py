# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is refer from:
https://github.com/WenmuZhou/DBNet.pytorch/blob/master/models/losses/DB_loss.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
import paddle.nn.functional as F
from .det_basic_loss import BalanceLoss, MaskL1Loss, DiceLoss


class CrossEntropyLoss(nn.Layer):

    def __init__(self,
                 weight=None,
                 ignore_index=255,
                 top_k_percent_pixels=1.0,
                 data_format='NCHW'):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.top_k_percent_pixels = top_k_percent_pixels
        self.EPS = 1e-8
        self.data_format = data_format
        if weight is not None:
            self.weight = paddle.to_tensor(weight, dtype='float32')
        else:
            self.weight = None

    def forward(self, logit, label, semantic_weights=None):
        channel_axis = 1 if self.data_format == 'NCHW' else -1
        if self.weight is not None and logit.shape[channel_axis] != len(
                self.weight):
            raise ValueError(
                'The number of weights = {} must be the same as the number of classes = {}.'
                .format(len(self.weight), logit.shape[channel_axis]))

        if channel_axis == 1:
            logit = paddle.transpose(logit, [0, 2, 3, 1])
        label = label.astype('int64')
        # In F.cross_entropy, the ignore_index is invalid, which needs to be fixed.
        # When there is 255 in the label and paddle version <= 2.1.3, the cross_entropy OP will report an error, which is fixed in paddle develop version.
        loss = F.cross_entropy(
            logit,
            label,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.weight)

        return self._post_process_loss(logit, label, semantic_weights, loss)

    def _post_process_loss(self, logit, label, semantic_weights, loss):
        mask = label != self.ignore_index
        mask = paddle.cast(mask, 'float32')
        label.stop_gradient = True
        mask.stop_gradient = True

        if loss.ndim > mask.ndim:
            loss = paddle.squeeze(loss, axis=-1)
        loss = loss * mask
        if semantic_weights is not None:
            loss = loss * semantic_weights

        if self.weight is not None:
            _one_hot = F.one_hot(label, logit.shape[-1])
            coef = paddle.sum(_one_hot * self.weight, axis=-1)
        else:
            coef = paddle.ones_like(label)

        if self.top_k_percent_pixels == 1.0:
            avg_loss = paddle.mean(loss) / (paddle.mean(mask * coef) + self.EPS)
        else:
            loss = loss.reshape((-1,))
            top_k_pixels = int(self.top_k_percent_pixels * loss.numel())
            loss, indices = paddle.topk(loss, top_k_pixels)
            coef = coef.reshape((-1,))
            coef = paddle.gather(coef, indices)
            coef.stop_gradient = True
            coef = coef.astype('float32')
            avg_loss = loss.mean() / (paddle.mean(coef) + self.EPS)

        return avg_loss


class DBLoss(nn.Layer):
    """
    Differentiable Binarization (DB) Loss Function
    args:
        param (dict): the super paramter for DB Loss
    """

    def __init__(self,
                 balance_loss=True,
                 main_loss_type='DiceLoss',
                 alpha=5,
                 beta=10,
                 ohem_ratio=3,
                 eps=1e-6,
                 num_classes=10,
                 **kwargs):
        super(DBLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.bce_loss = BalanceLoss(
            balance_loss=balance_loss,
            main_loss_type=main_loss_type,
            negative_ratio=ohem_ratio)

        self.loss_func = CrossEntropyLoss()

    def forward(self, predicts, labels):
        predict_maps = predicts['maps']
        if self.num_classes > 1:
            predict_classes = predicts['classes']
            label_threshold_map, label_threshold_mask, label_shrink_map, label_shrink_mask, class_mask = labels[1:]
        else:
            label_threshold_map, label_threshold_mask, label_shrink_map, label_shrink_mask = labels[1:]

        shrink_maps = predict_maps[:, 0, :, :]
        threshold_maps = predict_maps[:, 1, :, :]
        binary_maps = predict_maps[:, 2, :, :]

        loss_shrink_maps = self.bce_loss(shrink_maps, label_shrink_map,
                                         label_shrink_mask)
        loss_threshold_maps = self.l1_loss(threshold_maps, label_threshold_map,
                                           label_threshold_mask)
        loss_binary_maps = self.dice_loss(binary_maps, label_shrink_map,
                                          label_shrink_mask)
        loss_shrink_maps = self.alpha * loss_shrink_maps
        loss_threshold_maps = self.beta * loss_threshold_maps

        # 处理
        if self.num_classes > 1:
            loss_classes = self.loss_func(predict_classes, class_mask)

            loss_all = loss_shrink_maps + loss_threshold_maps + loss_binary_maps + loss_classes

            losses = {'loss': loss_all,
                      "loss_shrink_maps": loss_shrink_maps,
                      "loss_threshold_maps": loss_threshold_maps,
                      "loss_binary_maps": loss_binary_maps,
                      "loss_classes": loss_classes}
        else:
            loss_all = loss_shrink_maps + loss_threshold_maps + loss_binary_maps

            losses = {'loss': loss_all,
                      "loss_shrink_maps": loss_shrink_maps,
                      "loss_threshold_maps": loss_threshold_maps,
                      "loss_binary_maps": loss_binary_maps}
        return losses
