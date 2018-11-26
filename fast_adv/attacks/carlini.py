from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim


class CarliniWagnerL2:
    """
    Carlini's attack (C&W): https://arxiv.org/abs/1608.04644
    Based on https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks_tf.py

    Parameters
    ----------
    image_constraints : tuple
        Bounds of the images.
    num_classes : int
        Number of classes of the model to attack.
    confidence : float, optional
        Confidence of the attack for Carlini's loss, in term of distance between logits.
    learning_rate : float
        Learning rate for the optimization.
    search_steps : int
        Number of search steps to find the best scale constant for Carlini's loss.
    max_iterations : int
        Maximum number of iterations during a single search step.
    initial_const : float
        Initial constant of the attack.
    quantize : bool, optional
        If True, the returned adversarials will have possible values (1/255, 2/255, etc.).
    device : torch.device, optional
        Device to use for the attack.
    callback : object, optional
        Callback to display losses.
    """

    def __init__(self,
                 image_constraints: Tuple[float, float],
                 num_classes: int,
                 confidence: float = 0,
                 learning_rate: float = 0.01,
                 search_steps: int = 9,
                 max_iterations: int = 10000,
                 abort_early: bool = True,
                 initial_const: float = 0.001,
                 quantize: bool = False,
                 device: torch.device = torch.device('cpu'),
                 callback: Optional = None) -> None:

        self.confidence = confidence
        self.learning_rate = learning_rate

        self.binary_search_steps = search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.num_classes = num_classes

        self.repeat = self.binary_search_steps >= 10

        self.boxmin = image_constraints[0]
        self.boxmax = image_constraints[1]
        self.boxmul = (self.boxmax - self.boxmin) / 2
        self.boxplus = (self.boxmin + self.boxmax) / 2
        self.quantize = quantize

        self.device = device
        self.callback = callback
        self.log_interval = 10

    @staticmethod
    def _arctanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x *= (1. - eps)
        return (torch.log((1 + x) / (1 - x))) * 0.5

    def _step(self, model: nn.Module, optimizer: optim.Optimizer, inputs: torch.Tensor, tinputs: torch.Tensor,
              modifier: torch.Tensor, labels: torch.Tensor, labels_infhot: torch.Tensor, targeted: bool,
              const: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = inputs.shape[0]
        adv_input = torch.tanh(tinputs + modifier) * self.boxmul + self.boxplus
        l2 = (adv_input - inputs).view(batch_size, -1).pow(2).sum(1)

        logits = model(adv_input)

        real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        other = (logits - labels_infhot).max(1)[0]
        if targeted:
            # if targeted, optimize for making the other class most likely
            logit_dists = torch.clamp(other - real + self.confidence, min=0)
        else:
            # if non-targeted, optimize for making this class least likely.
            logit_dists = torch.clamp(real - other + self.confidence, min=0)

        logit_loss = (const * logit_dists).sum()
        l2_loss = l2.sum()
        loss = logit_loss + l2_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return adv_input.detach(), logits.detach(), l2.detach(), logit_dists.detach(), loss.detach()

    def attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
               targeted: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.

        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack.
        labels : torch.Tensor
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.

        Returns
        -------
        torch.Tensor
            Batch of samples modified to be adversarial to the model

        """
        batch_size = inputs.shape[0]
        tinputs = self._arctanh((inputs - self.boxplus) / self.boxmul)

        # set the lower and upper bounds accordingly
        lower_bound = torch.zeros(batch_size, device=self.device)
        CONST = torch.full((batch_size,), self.initial_const, device=self.device)
        upper_bound = torch.full((batch_size,), 1e10, device=self.device)

        o_best_l2 = torch.full((batch_size,), 1e10, device=self.device)
        o_best_score = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)
        o_best_attack = inputs.clone()

        # setup the target variable, we need it to be in one-hot form for the loss function
        labels_onehot = torch.zeros(labels.size(0), self.num_classes, device=self.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))

        for outer_step in range(self.binary_search_steps):

            # setup the modifier variable, this is the variable we are optimizing over
            modifier = torch.zeros_like(inputs, requires_grad=True)

            # setup the optimizer
            optimizer = optim.Adam([modifier], lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)
            best_l2 = torch.full((batch_size,), 1e10, device=self.device)
            best_score = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and outer_step == (self.binary_search_steps - 1):
                CONST = upper_bound

            prev = float('inf')
            for iteration in range(self.max_iterations):
                # perform the attack
                adv, logits, l2, logit_dists, loss = self._step(model, optimizer, inputs, tinputs, modifier,
                                                                labels, labels_infhot, targeted, CONST)

                if self.callback and (iteration + 1) % self.log_interval == 0:
                    self.callback.scalar('logit_dist_{}'.format(outer_step), iteration + 1, logit_dists.mean().item())
                    self.callback.scalar('l2_norm_{}'.format(outer_step), iteration + 1, l2.sqrt().mean().item())

                # check if we should abort search if we're getting nowhere.
                if self.abort_early and iteration % (self.max_iterations // 10) == 0:
                    if loss > prev * 0.9999:
                        break
                    prev = loss

                # adjust the best result found so far
                predicted_classes = (logits - labels_onehot * self.confidence).argmax(1) if targeted else \
                    (logits + labels_onehot * self.confidence).argmax(1)

                is_adv = (predicted_classes == labels) if targeted else (predicted_classes != labels)
                is_smaller = l2 < best_l2
                o_is_smaller = l2 < o_best_l2
                is_both = is_adv * is_smaller
                o_is_both = is_adv * o_is_smaller

                best_l2[is_both] = l2[is_both]
                best_score[is_both] = predicted_classes[is_both]
                o_best_l2[o_is_both] = l2[o_is_both]
                o_best_score[o_is_both] = predicted_classes[o_is_both]
                o_best_attack[o_is_both] = adv[o_is_both]

            # adjust the constant as needed
            adv_found = (best_score == labels) if targeted else ((best_score != labels) * (best_score != -1))
            upper_bound[adv_found] = torch.min(upper_bound[adv_found], CONST[adv_found])
            adv_not_found = ~adv_found
            lower_bound[adv_not_found] = torch.max(lower_bound[adv_not_found], CONST[adv_not_found])
            is_smaller = upper_bound < 1e9
            CONST[is_smaller] = (lower_bound[is_smaller] + upper_bound[is_smaller]) / 2
            CONST[(~is_smaller) * adv_not_found] *= 10

        if self.quantize:
            adv_found = o_best_score != -1
            o_best_attack[adv_found] = self._quantize(model, inputs[adv_found], o_best_attack[adv_found],
                                                      labels[adv_found], targeted=targeted)

        # return the best solution found
        return o_best_attack

    def _quantize(self, model: nn.Module, inputs: torch.Tensor, adv: torch.Tensor, labels: torch.Tensor,
                  targeted: bool = False) -> torch.Tensor:
        """
        Quantize the continuous adversarial inputs.

        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack.
        adv : torch.Tensor
            Batch of continuous adversarial perturbations produced by the attack.
        labels : torch.Tensor
            Labels of the samples if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.

        Returns
        -------
        torch.Tensor
            Batch of samples modified to be quantized and adversarial to the model.

        """
        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.round((adv - inputs) * 255) / 255
        delta.requires_grad_(True)
        logits = model(inputs + delta)
        is_adv = (logits.argmax(1) == labels) if targeted else (logits.argmax(1) != labels)
        i = 0
        while not is_adv.all() and i < 100:
            loss = F.cross_entropy(logits, labels, reduction='sum')
            grad = autograd.grad(loss, delta)[0].view(batch_size, -1)
            order = grad.abs().max(1, keepdim=True)[0]
            direction = (grad / order).int().float()
            direction.mul_(1 - is_adv.float().unsqueeze(1))
            delta.data.view(batch_size, -1).sub_(multiplier * direction / 255)

            logits = model(inputs + delta)
            is_adv = (logits.argmax(1) == labels) if targeted else (logits.argmax(1) != labels)
            i += 1

        delta.detach_()
        if not is_adv.all():
            delta.data[~is_adv].copy_(torch.round((adv[~is_adv] - inputs[~is_adv]) * 255) / 255)

        return inputs + delta
