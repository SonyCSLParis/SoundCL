
from avalanche.training.templates.base_sgd import BaseSGDPlugin
from plugins.natgrad.ekfac import EKFAC
from plugins.natgrad.kfac import KFAC

class EKFAC_Plugin(BaseSGDPlugin):
    def __init__(self,network,eps=0.1):
        """Avalanche plugin implementing EKFAC (Eigenvalue-corrected Kronecker Factorization)

        Args:
            network (nn.Module): The model to train
            eps (float): Tikhonov regularization parameter. Defaults to 0.1.
        """
        super().__init__()
        self.preconditioner = EKFAC(network,eps,ra=True)

    def before_update(self,strategy):
        self.preconditioner.step()

class KFAC_Plugin(BaseSGDPlugin):
    def __init__(self,network,eps=0.1):
        """Avalanche plugin implementing KFAC (Kronecker Factorization)

        Args:
            network (nn.Module): The model to train
            eps (float): Tikhonov regularization parameter. Defaults to 0.1.
        """
        super().__init__()
        self.preconditioner = KFAC(network,eps)

    def before_update(self,strategy):
        self.preconditioner.step()