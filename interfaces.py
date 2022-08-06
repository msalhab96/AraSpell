from abc import ABC, abstractmethod, abstractproperty


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


class ITokenizer(ABC):

    @abstractmethod
    def ids2tokens(self):
        pass

    @abstractmethod
    def tokenize(self):
        pass

    @abstractmethod
    def set_tokenizer(self):
        pass

    @abstractmethod
    def save_tokenizer(self):
        pass

    @abstractmethod
    def load_tokenizer(self):
        pass

    @abstractmethod
    def add_token(self):
        pass

    @abstractmethod
    def preprocess_tokens(self):
        pass

    @abstractmethod
    def batch_tokenizer(self):
        pass

    @abstractproperty
    def vocab_size(self):
        pass

    @abstractmethod
    def get_tokens(self):
        pass


class ILogger(ABC):

    @abstractmethod
    def log_step():
        pass

    @abstractmethod
    def log():
        pass

    @abstractmethod
    def set_rank():
        pass

    @abstractmethod
    def log_img():
        pass


class IPredictor(ABC):
    
    @abstractmethod
    def predict():
        pass
