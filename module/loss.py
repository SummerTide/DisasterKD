import torch.nn.functional as F
import torch.nn as nn
import torch

def _masked_ignore(y_pred: torch.Tensor, y_true: torch.Tensor, ignore_index: int):
    # usually used for BCE-like loss
    y_pred = y_pred.reshape((-1,))
    y_true = y_true.reshape((-1,))
    valid = y_true != ignore_index
    y_true = y_true.masked_select(valid).float()
    y_pred = y_pred.masked_select(valid).float()
    return y_pred, y_true

def tversky_loss_with_logits(y_pred: torch.Tensor, y_true: torch.Tensor, alpha: float, beta: float,
                             smooth_value: float = 1.0,
                             ignore_index: int = 255):
    y_pred, y_true = _masked_ignore(y_pred, y_true, ignore_index)

    y_pred = y_pred.sigmoid()
    tp = (y_pred * y_true).sum()
    # fp = (y_pred * (1 - y_true)).sum()
    fp = y_pred.sum() - tp
    # fn = ((1 - y_pred) * y_true).sum()
    fn = y_true.sum() - tp

    tversky_coeff = (tp + smooth_value) / (tp + alpha * fn + beta * fp + smooth_value)
    return 1. - tversky_coeff

def binary_cross_entropy_with_logits(output: torch.Tensor, target: torch.Tensor, reduction: str = 'mean',
                                     ignore_index: int = 255):
    output, target = _masked_ignore(output, target, ignore_index)
    return F.binary_cross_entropy_with_logits(output, target, reduction=reduction)



def softmax_focalloss(y_pred, y_true, ignore_index=-1, gamma=2.0, normalize=False):
    """
    Args:
        y_pred: [N, #class, H, W]
        y_true: [N, H, W] from 0 to #class
        gamma: scalar
    Returns:
    """
    losses = F.cross_entropy(y_pred, y_true, ignore_index=ignore_index, reduction='none')
    with torch.no_grad():
        p = y_pred.softmax(dim=1)
        modulating_factor = (1 - p).pow(gamma)
        valid_mask = ~ y_true.eq(ignore_index)
        masked_y_true = torch.where(valid_mask, y_true, torch.zeros_like(y_true))
        modulating_factor = torch.gather(modulating_factor, dim=1, index=masked_y_true.unsqueeze(dim=1)).squeeze_(dim=1)
        scale = 1.
        if normalize:
            scale = losses.sum() / (losses * modulating_factor).sum()
    losses = scale * (losses * modulating_factor).sum() / (valid_mask.sum() + p.size(0))

    return losses


def multi_binary_label(batched_mask: torch.Tensor, num_classes, ignore_index):
    labels = []
    for cls in range(0, num_classes):
        binary_label = torch.zeros_like(batched_mask)
        binary_label[batched_mask == cls] = 1
        binary_label[batched_mask == ignore_index] = ignore_index
        labels.append(binary_label.to(torch.long))
    return labels


def multi_binary_loss(y_pred, y_true, reduction='mean', ignore_index=-1):
    num_classes = y_pred.size(1)
    labels = multi_binary_label(y_true, num_classes, ignore_index=ignore_index)
    losses = []
    for cls in range(0, num_classes):
        logits, target = _masked_ignore(y_pred[:, cls], labels[cls], ignore_index)
        if target.numel() == 0:
            losses.append(logits.sum())
            continue
        bce_loss = F.binary_cross_entropy_with_logits(logits, torch.where(target == 0, 0.1 + target, target - 0.1))
        prob = logits.sigmoid()
        dice_loss = 1 - (2 * (prob * target).sum() + 1) / (prob.sum() + target.sum() + 1)
        losses.append(bce_loss + dice_loss)

    if 'sum' == reduction:
        tloss = sum(losses)
    elif 'mean' == reduction:
        tloss = sum(losses) / float(num_classes)
    else:
        raise ValueError()

    return tloss, losses

class SegmentationLoss(nn.Module):
    def __init__(self, loss_config):
        super(SegmentationLoss, self).__init__()
        self.loss_config = loss_config

    def forward(self, y_pred, y_true: torch.Tensor):
        loss_dict = dict()
        if 'ce' in self.loss_config:
            loss_dict['ce_loss'] = F.cross_entropy(y_pred, y_true.long(), ignore_index=-1)
        if 'fcloss' in self.loss_config:
            loss_dict['fc_loss'] = softmax_focalloss(y_pred, y_true, gamma=self.loss_config['fcloss']['gamma'], normalize=True)

        if 'bceloss' in self.loss_config:
            y_predb = y_pred[:, 0, :, :]
            invalidmask = y_true == -1
            bg_y_true = torch.where(y_true>0, torch.ones_like(y_predb), torch.zeros_like(y_predb))
            bg_y_true[invalidmask] = -1
            loss_dict['bceloss'] = binary_cross_entropy_with_logits(y_predb, bg_y_true, ignore_index=-1) * self.loss_config['bceloss']['scaler']


        if 'tverloss' in self.loss_config:
            y_predb = y_pred[:, 0, :, :]
            invalidmask = y_true == -1
            bg_y_true = torch.where(y_true>0, torch.ones_like(y_predb), torch.zeros_like(y_predb))
            bg_y_true[invalidmask] = -1
            loss_dict['tverloss'] = tversky_loss_with_logits(y_predb, bg_y_true, self.loss_config['tverloss']['alpha'],
                                                             self.loss_config['tverloss']['beta'], ignore_index=-1) * self.loss_config['tverloss']['scaler']

        if 'mbloss' in self.loss_config:
            loss_dict['mbloss'], losses = multi_binary_loss(y_pred, y_true, ignore_index=-1)
            for idx, l in enumerate(losses):
                loss_dict['loss_%d' % idx ] = l.data

        return loss_dict

__all__ = ['InfoNCE', 'info_nce']


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        loss = InfoNCE()
        batch_size, num_negative, embedding_size = 32, 48, 128
        query = torch.randn(batch_size, embedding_size)
        positive_key = torch.randn(batch_size, embedding_size)
        negative_keys = torch.randn(num_negative, embedding_size)
        output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.', query.shape, positive_key.shape)
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


if __name__ == '__main__':
    pred = torch.rand((2, 4, 10, 10))
    gt = torch.ones(2, 10, 10)
    gt[0, :, :] = -1
    loss = multi_binary_loss(pred, gt)