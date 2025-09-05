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
        Generates trading signals based on the provided data.

        :param data: A dictionary where keys are symbols and values are pandas DataFrames
                     containing historical data.
        :return: A dictionary of signals for each symbol.
        """
        pass
