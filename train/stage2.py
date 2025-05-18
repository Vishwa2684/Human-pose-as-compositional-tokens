import os
import torch
from dotenv import load_dotenv
from model.modules import BasicBlock,MixerLayer
from model.encoder import CompositionalEncoder
from process.data import MPIIDataset
from api.models import load_decoder_and_quantizer_weights
import warnings
from huggingface_hub import login
import timm
warnings.filterwarnings("ignore")
load_dotenv()

login(token=os.getenv('HUGGINGFACE_TOKEN'))
# print(*(m+"\n" for m in timm.list_models() if 'swinv2' in m))

# CONFIG
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_JOINTS = 16
IMAGE_SIZE = 256
DIMENSION = 2
HIDDEN_DIM = 256
m = 16

# loading weights
load_decoder_and_quantizer_weights()
encoder_weights = torch.load(os.path.join(os.getcwd(),'weights','50.pt'),map_location=device)

# Loading Encoder
encoder = CompositionalEncoder(k=NUM_JOINTS,d=DIMENSION,h=HIDDEN_DIM,m=m).to(device)
encoder.load_state_dict(encoder_weights['encoder'])
print('encoder weights loaded')

backbone = timm.create_model('swinv2_base_window12to16_192to256.ms_in22k_ft_in1k', pretrained=True)
backbone.to(device)
print('backbone loaded')
# backbone extracts image features X

# then use 2 basic residual convolutional blocks to modulate features of X.

# then convert the modulated features into linear projections

# Then 1-d output is reshaped to shape M x N

# M x N features are passed to 4 mixer blocks to obtain logits for token classification. The shape of logits would be M x V

###### To train the classification head we use cross entropy training to minimize the loss between logits and ground truth token classes 
###### obtained by feeding ground truth poses to encoder.

# decoder is not updated during training
