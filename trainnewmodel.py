import argparse
from pathlib import Path
from src.imageprocessing.imageLoader import ImageLoader
from src.vae.vaemodeltrainer import VaeModelTrainer
from src.cvae.cvaemodeltrainer import CvaeModelTrainer
from src.diffusion.diffusionmodeltrainer import DiffusionModelTrainer
from src.gan.ganmodeltrainer import GANModelTrainer
from keras.utils import to_categorical

'''
Use to generate images from trained model; call from cmd line
Example usage:
    python3.10 trainnewmodel.py diffusiontest.keras diffusion
    python3.10 trainnewmodel.py vaetest.keras vae
    python3.10 trainnewmodel.py cvaetest.keras cvae
    python3.10 trainnewmodel.py gantest.keras gan

'''

def run(modeltype: str, modelname: str):
    #load the images
    #TODO could add the dataset name to the arg parser
    imload = ImageLoader()
    imload.loadMnistData()
    xTrain, yTrain = imload.getTrainData()
    xTest, yTest = imload.getTestData()

    #create the trainer
    mdl=None
    if modeltype.upper() == 'VAE':
        mdl = VaeModelTrainer(xTrain=xTrain, trainEpochs=2)
    elif modeltype.upper() == 'CVAE':
        yTrainEnc = to_categorical(yTrain)
        mdl = CvaeModelTrainer(xTrain=xTrain, yTrain=yTrainEnc, trainEpochs=2)
    elif modeltype.upper() == 'DIFFUSION':
        mdl = DiffusionModelTrainer(xTrain=xTrain, yTrain = None, trainEpochs=2) 
    elif modeltype.upper() == 'GAN':
        mdl = GANModelTrainer(xTrain=xTrain, yTrain = None, trainEpochs=2)
    
    if mdl:
        Path("Models").mkdir(parents=True, exist_ok=True)
        mdl.fitModel(filepath='Models/' + modelname)


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("modelname", help="Specify the filename of the saved model")
    parser.add_argument("modeltype", help="Specify the type of generative model")
    args = parser.parse_args()
    run(modeltype=args.modeltype, modelname=args.modelname)
    
    #for testing; leave commented for now
    #run(modeltype = 'cvae', modelname='deletecvaetest.keras')
