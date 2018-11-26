import warnings
import foolbox
import numpy as np
import tqdm


class DeepFoolTF:
    """
    Wrapper for the DeepFool attack on TF, using the implementation from foolbox

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
                 input,
                 logits,
                 num_classes: int = 10,
                 max_iter: int = 100,
                 subsample: int = 10) -> None:
        self.input = input
        self.logits = logits
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.subsample = subsample

    def attack(self, inputs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Performs the attack of the model for the inputs and labels.

        Parameters
        ----------
        inputs : np.ndarray
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : np.ndarray
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.

        Returns
        -------
        np.ndarray
            Batch of samples modified to be adversarial to the model.

        """
        if inputs.min() < 0 or inputs.max() > 1:
            raise ValueError('Input values should be in the [0, 1] range.')

        fmodel = foolbox.models.TensorFlowModel(self.input, self.logits, bounds=(0, 1))
        attack = foolbox.attacks.DeepFoolL2Attack(model=fmodel)

        batch_size = len(inputs)
        adversarials = inputs.copy()

        warnings.filterwarnings('ignore', category=UserWarning)
        for i in tqdm.tqdm(range(batch_size), ncols=80):
            adv = attack(inputs[i], labels[i], unpack=True, steps=self.max_iter, subsample=self.subsample)
            if adv is not None:
                adversarials[i] = adv
        warnings.resetwarnings()

        return adversarials
