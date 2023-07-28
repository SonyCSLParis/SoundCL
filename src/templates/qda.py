import warnings
from typing import Optional, Sequence

import os
import torch
from math import log

from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.models.dynamic_modules import MultiTaskModule
from avalanche.models import FeatureExtractorBackbone


class StreamingQDA(SupervisedTemplate):
    r"""First try at streaming QDA.

    .. attention::
        Please fix the implementation by using the final code from https://github.com/SonyCSLParis/Deep_SRDA
    
    """
    #FIXME

    def __init__(
        self,
        sqda_model,
        criterion,
        input_size,
        num_classes,
        output_layer_name=None,
        shrinkage_param=1e-4,
        streaming_update_sigma=True,
        train_epochs: int = 1,
        train_mb_size: int = 1,
        eval_mb_size: int = 1,
        device="cpu",
        plugins: Optional[Sequence["SupervisedPlugin"]] = None,
        evaluator=default_evaluator(),
        eval_every=-1,
    ):
        """Init function for the SQDA model.

        :param sqda_model: a PyTorch model
        :param criterion: loss function
        :param output_layer_name: if not None, wrap model to retrieve
            only the `output_layer_name` output. If None, the strategy
            assumes that the model already produces a valid output.
            You can use `FeatureExtractorBackbone` class to create your custom
            SQDA-compatible model.
        :param input_size: feature dimension
        :param num_classes: number of total classes in stream
        :param train_mb_size: batch size for feature extractor during
            training. Fit will be called on a single pattern at a time.
        :param eval_mb_size: batch size for inference
        :param shrinkage_param: value of the shrinkage parameter
        :param streaming_update_sigma: True if sigma is plastic else False
            feature extraction in `self.feature_extraction_wrapper`.
        :param plugins: list of StrategyPlugins
        :param evaluator: Evaluation Plugin instance
        :param eval_every: run eval every `eval_every` epochs.
            See `BaseTemplate` for details.
        """

        if plugins is None:
            plugins = []

        sqda_model = sqda_model.eval()
        if output_layer_name is not None:
            sqda_model = FeatureExtractorBackbone(
                sqda_model.to(device), output_layer_name
            ).eval()

        super(StreamingQDA, self).__init__(
            sqda_model,
            None,
            criterion,
            train_mb_size,
            train_epochs,
            eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
        )

        # SQDA parameters
        self.input_size = input_size
        self.shrinkage_param = shrinkage_param
        self.streaming_update_sigma = streaming_update_sigma

        self.num_classes=num_classes

        # setup weights for SQDA
        self.muK = torch.zeros((num_classes, input_size)).to(self.device)
        self.cK = torch.zeros(num_classes).to(self.device)
        self.SigmaK = torch.ones((num_classes,input_size, input_size)).to(self.device)
        self.Sigma = torch.ones((input_size,input_size)).to(self.device)
        self.RegSigmaK = torch.empty((num_classes,input_size, input_size)).to(self.device)
       
        self.num_updates = torch.zeros(num_classes).to(self.device)
        self.Lambda = torch.zeros_like(self.SigmaK).to(self.device)
        self.prev_num_updates = (torch.zeros(num_classes)-1).to(self.device)

        #TODO add priors
        self.priorK = torch.ones(num_classes).to(self.device)

    def forward(self, return_features=False):
        """Compute the model's output given the current mini-batch."""
        self.model.eval()
        if isinstance(self.model, MultiTaskModule):
            feat = self.model(self.mb_x, self.mb_task_id)
        else:  # no task labels
            feat = self.model(self.mb_x)
        out = self.predict(feat)
        
        if return_features:
            return out, feat
        else:
            return out

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
            self.mb_output, feats = self.forward(return_features=True)
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self.criterion()

            # Optimization step
            self._before_update(**kwargs)
            # process one element at a time
            for f, y in zip(feats, self.mb_y):
                self.fit(f.unsqueeze(0), y.unsqueeze(0))
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def make_optimizer(self):
        """Empty function.
        Deep Sqda does not need a Pytorch optimizer."""
        pass

    @torch.no_grad()
    def fit(self, x, y):
        """
        Fit the SQDA model to a new sample (x,y).
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """
        #TODO retreive the y input and only update the y value
        # covariance updates


        if self.streaming_update_sigma:
            x_minus_mu = x - self.muK[y]
            mult = torch.matmul(x_minus_mu.transpose(1, 0), x_minus_mu)
            delta = mult * self.num_updates[y] / (self.num_updates[y] + 1)
            self.SigmaK[y] = (self.num_updates[y] * self.SigmaK[y] + delta) / (
                self.num_updates[y] + 1
            )
            delta2 = mult * torch.sum(self.num_updates) / (torch.sum(self.num_updates) + 1)
            self.Sigma = (torch.sum(self.num_updates) * self.Sigma + delta2) / (
                torch.sum(self.num_updates) + 1
            )

        # update class means
        self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(1)
        self.cK[y] += 1
        self.num_updates[y] += 1

    @torch.no_grad()
    def predict(self, X,return_probas=False):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead
        of predictions returned
        :return: the test predictions or probabilities
        """

        #FIXME fix this the main problem here is that we update every iteration were in the regular setting it is updated every batchsize
        # compute/load Lambda matrix
        #Compute priors
        s=torch.sum(self.cK).item()
        for i in range(self.num_classes):
            p=self.cK[i]/s
            if p==0:
                self.priorK[i]=1
            else:
                self.priorK[i]=p

        #compute the regularized cov matrix

        for i in range(self.num_classes):
            self.RegSigmaK[i]=0.3*self.SigmaK[i]+0.7*self.Sigma
        
        # there have been updates to the model, compute Lambda
        for i in range(self.num_classes):
            if self.prev_num_updates[i] != self.num_updates[i]:
                self.Lambda[i] = torch.pinverse(
                    (1 - self.shrinkage_param) * self.RegSigmaK[i]
                    + self.shrinkage_param
                    * torch.eye(self.input_size, device=self.device)
                )
                self.prev_num_updates[i] = self.num_updates[i]

        scores=[]
        
        for i in range(X.shape[0]):
            sample_score=[]
            for k in range(self.num_classes):
                if self.cK[k]==0:
                    sample_score.append(float('-inf'))
                else:
                #log(self.priorK[k].to(self.device))
                    sample_score.append(-0.5*torch.log(torch.norm(self.RegSigmaK[k].to(self.device)))-0.5*torch.matmul((X[i].to(self.device)-self.muK[k].to(self.device)),torch.matmul(self.Lambda[k].float().to(self.device),(X[i].to(self.device)-self.muK[k].to(self.device)))))
            scores.append(torch.Tensor(sample_score))
        
        scores=torch.stack(scores)

        if not return_probas:
            return scores.to(self.device)
        else:
            return torch.softmax(scores,dim=1).cpu()

    def save_model(self, save_path, save_name):
        """
        Save the model parameters to a torch file.
        :param save_path: the path where the model will be saved
        :param save_name: the name for the saved file
        :return:
        """
        # grab parameters for saving
        d = dict()
        d["muK"] = self.muK.cpu()
        d["cK"] = self.cK.cpu()
        d["Sigma"] = self.SigmaK.cpu()
        d["num_updates"] = self.num_updates.cpu()

        # save model out
        torch.save(d, os.path.join(save_path, save_name + ".pth"))

    def load_model(self, save_path, save_name):
        """
        Load the model parameters into StreamingQDA object.
        :param save_path: the path where the model is saved
        :param save_name: the name of the saved file
        :return:
        """
        # load parameters
        d = torch.load(os.path.join(save_path, save_name + ".pth"))
        self.muK = d["muK"].to(self.device)
        self.cK = d["cK"].to(self.device)
        self.SigmaK = d["Sigma"].to(self.device)
        self.num_updates = d["num_updates"].to(self.device)

    def _check_plugin_compatibility(self):
        """Check that the list of plugins is compatible with the template.

        This means checking that each plugin impements a subset of the
        supported callbacks.
        """

        ps = self.plugins

        def get_plugins_from_object(obj):
            def is_callback(x):
                return x.startswith("before") or x.startswith("after")

            return filter(is_callback, dir(obj))

        cb_supported = set(get_plugins_from_object(self.PLUGIN_CLASS))
        cb_supported.remove("before_backward")
        cb_supported.remove("after_backward")
        for p in ps:
            cb_p = set(get_plugins_from_object(p))

            if not cb_p.issubset(cb_supported):
                warnings.warn(
                    f"Plugin {p} implements incompatible callbacks for template"
                    f" {self}. This may result in errors. Incompatible "
                    f"callbacks: {cb_p - cb_supported}",
                )
                return


__all__ = ["StreamingQDA"]
