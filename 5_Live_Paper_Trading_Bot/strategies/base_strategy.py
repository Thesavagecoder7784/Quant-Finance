from abc import ABC, abstractmethod

class Strategy(ABC):
    """
    Abstract base class for a trading strategy.
    """
    def __init__(self, name, symbols, parameters):
        self.name = name
        self.symbols = symbols
        self.parameters = parameters

    @abstractmethod
    def generate_signals(self, data):
        """
        Generates trading signals for each symbol.
        """
        pass

    def get_state(self):
        """
        Gets the current state of the strategy.
        """
        return {}
