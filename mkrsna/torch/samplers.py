import torch
from torch.utils.data.sampler import Sampler
from typing import Iterator, List, Optional, Tuple
import numpy as np



def resolve_binary_class_upsample_weight(targets, upsample_positive_to_percent):
    "Returns weights for array of target classes so that positive examples are upsampled to given percent"
    
    weights = targets.astype(np.float64)
    
    dataset_len = len(weights)
    positives_len = len(targets[targets > 0])
    negatives_len = dataset_len-positives_len

    if positives_len == 0:
        return np.ones_like(targets, dtype=np.float64)
    
    x = negatives_len*upsample_positive_to_percent / (positives_len -  upsample_positive_to_percent*positives_len)
    weights[targets > 0] = x
    weights[targets <= 0 ] = 1
    return weights

# TODO - have exhaustive sampler with upsampling positive examples
# in contrast to random sampler which is not necessary exhaustive
# of course maybe we do not need exhaustiveness
class PositiveNegativeIndicesSampler(Sampler[int]):

    def __init__(self, positive_indices, negative_indices, num_samples,
                 generator=None) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))

        self.positive_indices = positive_indices
        self.negative_indices = negative_indices
        self.num_samples = num_samples
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples