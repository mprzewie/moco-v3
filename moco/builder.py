# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(
            self, base_encoder, dim=256, mlp_dim=4096, T=1.0,
            cassle: float = False, cassle_h: int=6, cassle_w: int=64,
            cassle_method: str="cat"
    ):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.cassle_w = cassle_w
        self.cassle_h = cassle_h
        self.cassle_method = cassle_method

        self.T = T

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self.cassle = cassle
        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        if self.cassle:
            out_size = cassle_w if cassle_method == "cat" else mlp_dim

            self.base_aug_processor = self._build_mlp(cassle_h, 15, cassle_w, out_size )
            self.momentum_aug_processor = self._build_mlp(cassle_h, 15, cassle_w, out_size)

            for param_b, param_m in zip(self.base_aug_processor.parameters(), self.momentum_aug_processor.parameters()):
                param_m.data.copy_(param_b.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        for param_b, param_m in zip(self.base_projector.parameters(), self.momentum_projector.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient



    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

        for param_b, param_m in zip(self.base_projector.parameters(), self.momentum_projector.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

        if self.cassle:
            for param_b, param_m in zip(self.base_aug_processor.parameters(), self.momentum_aug_processor.parameters()):
                param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, a1, a2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features

        e1 = self.base_encoder(x1)
        e2 = self.base_encoder(x2)

        if self.cassle:
            g1 = self.base_aug_processor(a1)
            g2 = self.base_aug_processor(a2)

            if self.cassle_method == "cat":
                e1 = torch.cat([e1, g1], dim=1)
                e2 = torch.cat([e2, g2], dim=1)
            elif self.cassle_method == "add":
                e1 = e1 + g1
                e2 = e2 + g2


        # assert False, e1.shape
        p1 = self.base_projector(e1)
        q1 = self.predictor(p1)
        q2 = self.predictor(self.base_projector(e2))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

            if self.cassle:
                g1 = self.momentum_aug_processor(a1)
                g2 = self.momentum_aug_processor(a2)
                k1 = torch.cat([k1, g1], dim=1)
                k2 = torch.cat([k2, g2], dim=1)

            k1 = self.momentum_projector(k1)
            k2 = self.momentum_projector(k2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]

        if self.cassle and self.cassle_method == "cat":
            hidden_dim += self.cassle_w

        self.base_encoder.fc, self.momentum_encoder.fc = nn.Sequential(), nn.Sequential()

        # projectors
        self.base_projector = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        self.momentum_projector = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]

        if self.cassle and self.cassle_method == "cat":
            hidden_dim += self.cassle_w

        self.base_encoder.head, self.momentum_encoder.head = nn.Sequential(), nn.Sequential()
        # del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_projector = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_projector = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
