import argparse
from src.diffusion.diffusionimagegenerator import DiffusionImageGenerator
from src.vae.vaeimagegenerator import VAEImageGenerator
from src.cvae.cvaeimagegenerator import CvaeImageGenerator

'''
Use to generate images from trained model; call from cmd line
Example usage:
    python3.10 generatenewimages.py Models/diffusionMnistAllData_3epoch.keras diffusion
    python3.10 generatenewimages.py Models/decoder30.keras VAE
    python3.10 generatenewimages.py Models/testcvae30.keras cvae
'''

#TODO should pass in digit or class to generate for cvae 
#TODO should save images from here; they currently have to be manually saved
def run(modelpath: str, modeltype: str):
    #add error handling! raise exception
    if modeltype.upper() == 'VAE':
        imGen = VAEImageGenerator(modelPath=modelpath)
    elif modeltype.upper() == 'DIFFUSION':
        imGen = DiffusionImageGenerator(modelPath=modelpath, imageShape = (28, 28, 1), nSteps = 50)
    elif modeltype.upper() == 'CVAE':
        imGen = CvaeImageGenerator(modelPath = modelpath, digit = 8)

        
    imGen.generateImages(numberOfSamples=100)
    imGen.showImages(rows=10, cols=10)
    
    

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument( "modelpath", help="Specify the filepath for the trained model")
    parser.add_argument("modeltype", help="Specify the type of generative model")
    args = parser.parse_args()
    run(modelpath=args.modelpath, modeltype=args.modeltype)
    
    #modelpath = 'Models/delete.keras'
    #modeltype = 'diffusion'

    #run(modelpath=modelpath, modeltype=modeltype)
    