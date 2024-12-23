from abc import ABCMeta, abstractmethod

from numpyro import infer

class AutoStep(infer.mcmc.MCMCKernel, metaclass=ABCMeta):

    def __init__(self, model=None, step_size=1.0, selector=None):
        self._model = model
        self.step_size = step_size
        self.selector = selector

    @property
    def model(self):
        return self._model

