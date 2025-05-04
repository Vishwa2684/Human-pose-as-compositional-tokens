from model.encoder import CompositionalEncoder,VectorQuantizer,Decoder
from tqdm import tqdm
import torch
from torchvision.transforms import transforms
from torch.optim import Adam
from process.data import MPIIDataset,KeypointDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F 
import os

# -----------------------------
# CONFIG
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_DIR = '../images/'
CSV_PATH = '../mpii_human_pose_v1_u12_2/mpii_human_pose.csv'
IMG_SIZE = 224

# -----------------------------
# DATA
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
])

mpii =  MPIIDataset(CSV_PATH,IMAGE_DIR,transform)
keypoints = KeypointDataset(mpii)

data = DataLoader(keypoints,batch_size=16,shuffle=True)
print('data loaded')
# -----------------------------
# PARAMETERS
# -----------------------------
NUM_JOINTS = 16
DIMENSION = 2
HIDDEN_DIM = 256
m = 16

# -----------------------------
# MODELS
# -----------------------------
encoder = CompositionalEncoder(k=NUM_JOINTS,d=DIMENSION,h=HIDDEN_DIM,m=m)
vq = VectorQuantizer(v=2*HIDDEN_DIM,h=HIDDEN_DIM,commitment_cost=0.25)
decoder = Decoder(k=NUM_JOINTS,d=DIMENSION,h=HIDDEN_DIM,m=m)

# -----------------------------
# TRAINING
# -----------------------------

EPOCHS = 100
LR = 1e-4
BETA = 0.25 # loss hyperparameter

optimizer = Adam(
    list(encoder.parameters()) + list(vq.parameters()) + list(decoder.parameters()),
    lr=LR,betas=[0.9,0.999]
)

for epoch in range(EPOCHS):
    total_loss = 0.0
    encoder.train()
    vq.train()
    decoder.train()
    for batch in tqdm(data):
        g = batch.to(device)
        g_e = encoder(keypoints)
        v_q,vq_loss,encoding_indices = vq(g_e)
        g_h = decoder(v_q)

        recon_loss = F.smooth_l1_loss(g_h,g)

        l_pct = recon_loss + BETA*vq_loss

        
        if epoch % 5 == 0:
            torch.save({
                'quantizer':vq.state_dict(),
                'decoder':decoder.state_dict()
            },f'./ckpt/{epoch}.pt')

        