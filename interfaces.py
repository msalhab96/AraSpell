from abc import ABC, abstractmethod


class IProcess(ABC):

    @abstractmethod
    def execute():
        pass


class IProcessor(ABC):

    @abstractmethod
    def run():
        pass

    @abstractmethod
    def dist_run():
        pass
