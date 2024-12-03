from abc import ABC, abstractmethod
#TODO move into modeltraining folder and rename as generative model trainer; update imports in other model train functions
class GenerativeModelTrainer(ABC):
    def __init__(self):
        pass

    def fitModel(self, filepath: str):
        self._buildArchitecture()
        self._trainModel()
        self._saveTrainedModel(filepath=filepath)

    @abstractmethod
    def _buildArchitecture(self):
        pass

    @abstractmethod
    def _trainModel(self):
        pass

    @abstractmethod
    def _saveTrainedModel(self, filepath: str):
        pass





    