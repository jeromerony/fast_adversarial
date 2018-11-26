import warnings
import tqdm
import foolbox
import torch
import torch.nn as nn


class DeepFool:
    """
    Wrapper for the DeepFool attack, using the implementation from foolbox

    Parameters
    ----------
    num_classes : int
        Number of classes of the model.
    max_iter : int, optional
        Number of steps for the attack.
    subsample : int, optional
        Limit on the number of the most likely classes that should be considered.
    device : torch.device, optional
        Device on which to perform the attack.

    """

    def __init__(self,
                 num_classes: int = 10,
                 max_iter: int = 100,
                 subsample: int = 10,
                 device=torch.device) -> None:
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.subsample = subsample
        self.device = device

    def attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
               targeted: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.

        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : torch.Tensor
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.

        Returns
        -------
        torch.Tensor
            Batch of samples modified to be adversarial to the model.

        """
        if inputs.min() < 0 or inputs.max() > 1:
            raise ValueError('Input values should be in the [0, 1] range.')
        if targeted:
            print('DeepFool is an untargeted adversarial attack. Returning clean inputs.')
            return inputs

        fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=self.num_classes, device=self.device)
        attack = foolbox.attacks.DeepFoolL2Attack(model=fmodel)

        numpy_inputs = inputs.cpu().numpy()
        numpy_labels = labels.cpu().numpy()
        batch_size = len(inputs)
        adversarials = numpy_inputs.copy()

        warnings.filterwarnings('ignore', category=UserWarning)
        for i in tqdm.tqdm(range(batch_size), ncols=80):
            adv = attack(numpy_inputs[i], numpy_labels[i], unpack=True, steps=self.max_iter, subsample=self.subsample)
            if adv is not None:
                adversarials[i] = adv
        warnings.resetwarnings()

        adversarials = torch.from_numpy(adversarials).to(self.device)

        return adversarials
