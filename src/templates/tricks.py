from typing import Sequence, Optional
import copy

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from torch.nn import functional as F
import torch

from avalanche.core import SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator

from avalanche.models import avalanche_forward
from avalanche.training.templates.base_sgd import BaseSGDTemplate
from avalanche.training.templates.observation_type.batch_observation import BatchObservation
from avalanche.training.templates.update_type.sgd_update import SGDUpdate


class SupervisedLBProblem:
    """Avalanche Supervised Problem modified to include the labels trick https://arxiv.org/abs/1803.10123
    """

    def labels_trick(self,outputs, labels, criterion):
        """
        Labels trick calculates the loss only on labels which appear on the current mini-batch.
        It is implemented for classification loss types (e.g. CrossEntropyLoss()).
        :param outputs: The DNN outputs of the current mini-batch (torch Tensor).
        :param labels: The ground-truth (correct tags) (torch Tensor).
        :param criterion: Criterion (loss).
        :return: Loss value, after applying the labels trick.
        """
        # Get current batch labels (and sort them for reassignment)
        unq_lbls = labels.unique().sort()[0]
        # Create a copy of the labels to avoid in-place modification
        labels_copy = labels.clone()
        # Assign new labels (0,1 ...) because we will select from the outputs only the columns of labels of the current
        #   mini-batch (outputs[:, unq_lbls]), so their "tagging" will be changed (e.g. column number 3, which corresponds
        #   to label number 3 will become column number 0 if labels 0,1,2 do not appear in the current mini-batch, so its
        #   ground-truth should be changed accordingly to label #0).
        for lbl_idx, lbl in enumerate(unq_lbls):
            labels_copy[labels_copy == lbl] = lbl_idx
        # Calcualte loss only over the heads appear in the batch:
        return criterion(outputs[:, unq_lbls], labels_copy)

    @property
    def mb_x(self):
        """Current mini-batch input."""
        return self.mbatch[0]

    @property
    def mb_y(self):
        """Current mini-batch target."""
        return self.mbatch[1]

    @property
    def mb_task_id(self):
        """Current mini-batch task labels."""
        assert len(self.mbatch) >= 3
        return self.mbatch[-1]

    def criterion(self):
        """Loss function for supervised problems."""
        #Adding Labels trick 
        
        return self.labels_trick(self.mb_output, self.mb_y,self._criterion)

    def forward(self):
        """Compute the model's output given the current mini-batch."""
        return avalanche_forward(self.model, self.mb_x, self.mb_task_id)

    def _check_minibatch(self):
        """Check if the current mini-batch has 3 components."""
        assert len(self.mbatch) >= 3

class SupervisedSSProblem:
    """Avalanche Supervised Problem modified to include the Separated Softmax trick https://arxiv.org/pdf/2003.13947.pdf
    """

    def separated_softmax(self,logits,labels):
        copy_labels= labels.clone()

        old_ss = F.log_softmax(logits[:, self.old_labels], dim=1)
        new_ss = F.log_softmax(logits[:, self.new_labels], dim=1)
        ss = torch.cat([old_ss, new_ss], dim=1)
        for i, lbl in enumerate(labels):
            copy_labels[i] = self.lbl_inv_map[lbl.item()]
        return F.nll_loss(ss, copy_labels)

    @property
    def mb_x(self):
        """Current mini-batch input."""
        return self.mbatch[0]

    @property
    def mb_y(self):
        """Current mini-batch target."""
        return self.mbatch[1]

    @property
    def mb_task_id(self):
        """Current mini-batch task labels."""
        assert len(self.mbatch) >= 3
        return self.mbatch[-1]

    def criterion(self):
        """Loss function for supervised problems."""
        #Adding Labels trick 
        
        return self.separated_softmax(self.mb_output, self.mb_y)

    def forward(self):
        """Compute the model's output given the current mini-batch."""
        return avalanche_forward(self.model, self.mb_x, self.mb_task_id)

    def _check_minibatch(self):
        """Check if the current mini-batch has 3 components."""
        assert len(self.mbatch) >= 3

class SGDUpdate__SS:
    r"""Avalanche SGD update modified with Separated Softmax trick https://arxiv.org/pdf/2003.13947.pdf

    .. attention::
        Evaluation during training has to be done on seen classes or the training will crash. Be sure to modify the training loop in `main.py` before running. 

    """
    def ss_update_1(self, y_train):
        new_labels = list(set(y_train.tolist()))
        self.new_labels += new_labels
        for i, lbl in enumerate(new_labels):
            self.lbl_inv_map[lbl] = len(self.old_labels) + i


    def training_epoch(self, **kwargs):
        """Training epoch.

        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            self.ss_update_1(self.mb_y)


            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self.criterion()

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_update(**kwargs)
            
            self._after_training_iteration(**kwargs)
            self.old_labels += self.new_labels
            self.new_labels_zombie = copy.deepcopy(self.new_labels)
            self.new_labels.clear()


class Supervised_LB_Template(BatchObservation, SupervisedLBProblem, SGDUpdate,
                         BaseSGDTemplate):
    """Base class for continual learning strategies modified with a labels trick.
    """

    PLUGIN_CLASS = SupervisedPlugin

    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            criterion=CrossEntropyLoss(),
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: Optional[int] = 1,
            device="cpu",
            plugins: Optional[Sequence["BaseSGDPlugin"]] = None,
            evaluator=default_evaluator,
            eval_every=-1,
            peval_mode="epoch",
    ):
        """Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='epoch').
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            peval_mode=peval_mode,
        )


class Supervised_SS_Template(BatchObservation, SupervisedSSProblem, SGDUpdate__SS,
                         BaseSGDTemplate):
    r"""Base class for continual learning strategies modified with the separated softmax trick.

    .. attention::
        Evaluation during training has to be done on seen classes or the training will crash. Be sure to modify the training loop in `main.py` before running. 
    """

    PLUGIN_CLASS = SupervisedPlugin

    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            criterion=CrossEntropyLoss(),
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: Optional[int] = 1,
            device="cpu",
            plugins: Optional[Sequence["BaseSGDPlugin"]] = None,
            evaluator=default_evaluator,
            eval_every=-1,
            peval_mode="epoch",
    ):
        """Init.

        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='epoch').
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            peval_mode=peval_mode,
        )

        self.lbl_inv_map = {}
        self.old_labels = []
        self.new_labels = []