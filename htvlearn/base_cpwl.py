from abc import ABC, abstractmethod


class BaseCPWL(ABC):
    @abstractmethod
    def evaluate(self, x):
        """ Evaluate cpwl function at x.
        """
        pass

    @abstractmethod
    def evaluate_with_grad(self, x):
        """ Evaluate cpwl function at x and compute gradient.
        """
        pass
