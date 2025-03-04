import torch.nn as nn
from loss.abstract_loss_func import AbstractLossClass
from utils.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="cross_entropy")
class CrossEntropyLoss(AbstractLossClass):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        """
        Computes the cross-entropy loss.

        Args:
            inputs: A PyTorch tensor of size (batch_size, num_classes) containing the predicted scores.
            targets: A PyTorch tensor of size (batch_size) containing the ground-truth class indices.

        Returns:
            A scalar tensor representing the cross-entropy loss.
        """
        # Compute the cross-entropy loss
        # print(targets)
        loss = self.loss_fn(inputs, targets)
        # print(loss, 'cross_entropy loss')

        return loss
