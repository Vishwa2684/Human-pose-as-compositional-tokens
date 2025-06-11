import os
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from torchvision.transforms import transforms
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from model.encoder import CompositionalEncoder,ClassificationHead,Decoder,VectorQuantizer
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
IMAGE_DIR = '../images/'
CSV_PATH = '../mpii_human_pose_v1_u12_2/mpii_human_pose.csv'
DIMENSION = 2
HIDDEN_DIM = 256
m = 16

# -----------------------------
# DATA
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor(),
])

######################################################
# PREPARING MODELS
######################################################

# loading weights
load_decoder_and_quantizer_weights()
encoder_weights = torch.load(os.path.join(os.getcwd(),'weights','50.pt'),map_location=device)

# Loading Encoder
encoder = CompositionalEncoder(k=NUM_JOINTS,d=DIMENSION,h=HIDDEN_DIM,m=m).to(device)
encoder.load_state_dict(encoder_weights['encoder'])
print('encoder weights loaded')

backbone = timm.create_model('swinv2_base_window12to16_192to256.ms_in22k_ft_in1k', pretrained=True)
backbone.to(device)
backbone.requires_grad_(False) # freeze backbone weights for now as we are training classification head
print('backbone loaded')

head = ClassificationHead(
    in_channels=1024,       # Output from Swin-V2 backbone
    num_tokens=16,          # M
    token_dim=256,          # N
    codebook_size=512       # V
).to(device)
print('classification head loaded')

decoder = Decoder(k=NUM_JOINTS,d=DIMENSION,h=HIDDEN_DIM,m=m).to(device)
decoder.load_state_dict(encoder_weights['decoder'])
decoder.requires_grad_(False) # freeze decoder weights for now, we will train it in the next vide
print('decoder weights loaded')

vq = VectorQuantizer(v=2*HIDDEN_DIM,h=HIDDEN_DIM,commitment_cost=0.25).to(device)
vq.load_state_dict(encoder_weights['quantizer'])
vq.requires_grad_(False) # freeze vq weights for now, we will train it in the next video
print('vq weights loaded')

###################
# TRAINING CONFIG
###################
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 0.05
BATCH_SIZE = 256
EPOCHS = 210

mpii =  MPIIDataset(CSV_PATH,IMAGE_DIR,transform)
data = DataLoader(mpii,batch_size=BATCH_SIZE,shuffle=True)
print('data loaded')

# backbone extracts image features X

# then use 2 basic residual convolutional blocks to modulate features of X.

# then convert the modulated features into linear projections

# Then 1-d output is reshaped to shape M x N

# M x N features are passed to 4 mixer blocks to obtain logits for token classification. The shape of logits would be M x V

###### To train the classification head we use cross entropy training to minimize the loss between logits and ground truth token classes 
###### obtained by feeding ground truth poses to encoder.

# decoder is not updated during training

optimizer = AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

for epoch in range(EPOCHS):
    total_loss = 0.0
    loop = tqdm(enumerate(data), total=len(data), desc=f"Epoch [{epoch+1}/{EPOCHS}]")

    for batch_idx, (images, poses, _) in loop:
        images, poses = images.to(device), poses.to(device)

        # Forward pass
        features = backbone.forward_features(images)                # Feature extraction
        # print(features.shape)
        with torch.no_grad():
            g_e = encoder(poses)
            _, _, encoding_indices = vq(g_e)  # [B*M]
            l = encoding_indices.view(poses.size(0), NUM_JOINTS)  # [B, M, V]                        # Ground-truth token indices (B, M)
        # print(f'l:{l.shape}')
        l_hat = head(features)                     # Predicted logits (B, M, V)
        # print(f'l_hat:{l_hat.shape}')
        
        
        # CE(l_hat,l)
        ce_loss = torch.nn.CrossEntropyLoss()(l_hat.permute(0, 2, 1), l)

        # s = l_hat * c
        s = torch.matmul(torch.softmax(l_hat, dim=-1), vq.codebook.detach())  # (B, M, H)
        g_hat = decoder(s)

        # smooth_l1(g_hat,g)
        sl1_loss = torch.nn.functional.smooth_l1_loss(g_hat, poses)
        l_all = ce_loss + sl1_loss

        optimizer.zero_grad()
        l_all.backward()
        optimizer.step()

        total_loss += l_all.item()
        loop.set_postfix(loss=l_all.item())
    if (epoch+1) % 10 == 0:
        torch.save({
            'classification_head': head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': total_loss / len(data),
            'epoch': epoch+1,
        },f'../ckpt/stage 2/classification_head_{epoch+1}.pt')
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Avg Loss: {total_loss / len(data):.4f}")