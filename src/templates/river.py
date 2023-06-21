import warnings
from typing import Optional, Sequence

import os
import torch

from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.models.dynamic_modules import MultiTaskModule
from avalanche.models import FeatureExtractorBackbone

from river import metrics


class RiverTemplate(SupervisedTemplate):

    def __init__(
        self,
        deep_model,
        online_model,
        criterion,
        input_size,
        output_layer_name=None,
        train_epochs: int = 1,
        train_mb_size: int = 1,
        eval_mb_size: int = 1,
        device="cpu",
        plugins: Optional[Sequence["SupervisedPlugin"]] = None,
        evaluator=default_evaluator(),
        eval_every=-1,
    ):

        if plugins is None:
            plugins = []

        deep_model = deep_model.eval()
        if output_layer_name is not None:
            deep_model = FeatureExtractorBackbone(
                deep_model.to(device), output_layer_name
            ).eval()

        super(RiverTemplate, self).__init__(
            model=deep_model,
            optimizer=None,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )
        self.input_size=input_size

        self.online_model = online_model
        self.train_metrics_list=[metrics.ConfusionMatrix(),metrics.ClassificationReport()]
        self.test_metrics_list=[metrics.ConfusionMatrix(),metrics.ClassificationReport()]

    def tensor_to_dict(self,tensor):
        res=[]

        if len(tensor.shape)>1:
            #then we have data
            batch_size=tensor.shape[0]
            for i in range(batch_size):
                res.append(dict(zip([f'Feature {i}' for i in range(self.input_size)],tensor[i].cpu().tolist())))
        else:
            res=tensor.cpu().tolist()
        return res


    def forward(self):
        """Compute the model's output given the current mini-batch."""
        self.model.eval()
        if isinstance(self.model, MultiTaskModule):
            raise NotImplementedError
            #feat = self.model(self.mb_x, self.mb_task_id)
        else:  # no task labels
            feat = self.model(self.mb_x)
        return feat


    def training_epoch(self, **kwargs):
        """
        Training epoch.
        :param kwargs:
        :return:
        """
        for _, self.mbatch in enumerate(self.dataloader):
            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            # compute output on entire minibatch
            self.mb_output = self.forward()
            self._after_forward(**kwargs)
            
            # Optimization step
            self._before_update(**kwargs)

            # process one element at a time
            for x, y in zip(self.tensor_to_dict(self.mb_output), self.tensor_to_dict(self.mb_y)):
                #here we update the online model
                y_pred = self.online_model.predict_one(x)
                
                self.online_model.learn_one(x,y)
                if y_pred is not None:
                    for metric in self.train_metrics_list:
                        metric.update(y,y_pred)

            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)
            
    def _after_training_exp(self, **kwargs):
        for metric in self.train_metrics_list:
            print(metric)

    def eval_epoch(self, **kwargs):
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            self._before_eval_iteration(**kwargs)

            self._before_eval_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_eval_forward(**kwargs)
            
            # process one element at a time
            for x, y in zip(self.tensor_to_dict(self.mb_output), self.tensor_to_dict(self.mb_y)):
                #here we update the online model
                y_pred = self.online_model.predict_one(x)
                
                for metric in self.test_metrics_list:
                    metric.update(y,y_pred)

            self._after_eval_iteration(**kwargs)
            
    def _after_eval(self, **kwargs):
        for metric in self.test_metrics_list:
            print(metric)

    def make_optimizer(self):
        """Empty function.
        River online models do not need a Pytorch optimizer."""
        pass





__all__ = ["RiverTemplate"]
