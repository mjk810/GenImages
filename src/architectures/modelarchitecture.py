from abc import ABC, abstractmethod

class ModelArchitecture(ABC):

    def buildEncoder(self):
        pass

    def buildDecoder(self):
        pass