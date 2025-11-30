from datasets import load_dataset

class EsConv_Data():

    def __init__(self):
        self.dataset = load_dataset("thu-coai/esconv")
        self.train = self.dataset['train']
        self.test = self.dataset['test']
        self.validation = self.dataset['validation']

    def get_dataset(self):
        return self.dataset

    def get_data(self):
        return self.train, self.test, self.validation