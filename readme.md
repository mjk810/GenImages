## Generative AI
### Getting started

##### Create a virtual environment
`python -m venv .venv`
##### Activate the venv
`PowerShell -ExecutionPolicy Bypass` <br>
`.venv/Scripts/activate` <br>
##### Install dependencies
`pip install -r requirements.txt`

### To train a new model
There are four models available: a diffusion model, a vae, a cvae, and a gan. To train one of these models, use the code below. Provide the name of the model file that will be saved and the model type. Models will be saved in the Models folder <br>
`python3.10 trainnewmodel.py diffusiontest.keras diffusion` <br>
`python3.10 trainnewmodel.py vaetest.keras vae` <br>
`python3.10 trainnewmodel.py cvaetest.keras cvae` <br>
`python3.10 trainnewmodel.py gantest.keras gan` <br>

### To generate images using a saved model
To generate images using one of the trained models, use the code below: <br>
`python3.10 generatenewimages.py Models/<ModelName> diffusion` <br>
`python3.10 generatenewimages.py Models/<ModelName> VAE` <br>
`python3.10 generatenewimages.py Models/<ModelName> cvae` <br>

### Example outputs

| Model      | Image |
| ----------- | ----------- |
| MNIST images from VAE Model | <img src="ImageOutputs/mnistGeneratedImages.png" width="200" height="200" /> |
| MNIST images from Diffusion Model | <img src="ImageOutputs/diffusionMnistGeneratedImages.png" width="200" height="200" /> |
| CVAE Model Number 3 | <img src="ImageOutputs/cvae_digit3.png" width="200" height="200" /> |
| CVAE Model Number 4 | <img src="ImageOutputs/cvae_digit4.png" width="200" height="200" /> |
| CVAE Model Number 8 | <img src="ImageOutputs/cvae_digit8.png" width="200" height="200" /> |
