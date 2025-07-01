from model.encoder import CompositionalEncoder,VectorQuantizer,Decoder
from tqdm import tqdm
import torch
from torchvision.transforms import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from process.data import MPIIDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F 
import os

# -----------------------------
# CONFIG
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_DIR = '../images/'
CSV_PATH = '../mpii_human_pose_v1_u12_2/mpii_human_pose.csv'
IMG_SIZE = 256
BATCH_SIZE = 512

# -----------------------------
# DATA
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
])

mpii =  MPIIDataset(CSV_PATH,IMAGE_DIR,transform)
data = DataLoader(mpii,batch_size=BATCH_SIZE,shuffle=True)
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
encoder = CompositionalEncoder(k=NUM_JOINTS,d=DIMENSION,h=HIDDEN_DIM,m=m).to(device)
vq = VectorQuantizer(v=2*HIDDEN_DIM,h=HIDDEN_DIM,commitment_cost=0.25).to(device)
decoder = Decoder(k=NUM_JOINTS,d=DIMENSION,h=HIDDEN_DIM,m=m).to(device)

# -----------------------------
# TRAINING
# -----------------------------

EPOCHS = 50
LR = 1e-2
WEIGHT_DECAY = 0.15
BETA = 0.25 # loss hyperparameter
WARMUP_ITERS = 500
TOTAL_ITERS = len(data) * 50  # 50 epochs

optimizer = AdamW(
    list(encoder.parameters()) + list(vq.parameters()) + list(decoder.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)
scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_ITERS - WARMUP_ITERS)
global_step = 0  

CKPT_PATH = '../ckpt/6.pt'
start_epoch = 0

if os.path.exists(CKPT_PATH):
    print(f"✅ Resuming from checkpoint: {CKPT_PATH}")
    
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    vq.load_state_dict(checkpoint['quantizer'])
    decoder.load_state_dict(checkpoint['decoder'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print(f"✅ Loaded classification head and optimizer. Resuming from epoch {start_epoch}")
else:
    print(f"❌ Checkpoint not found at: {CKPT_PATH}")

for epoch in range(start_epoch,EPOCHS):
    total_loss = 0.0
    encoder.train()
    vq.train()
    decoder.train()

    loop = tqdm(data, desc=f'Epoch [{epoch+1}/{EPOCHS}]', leave=False)

    for images, keypoints, visibility in loop:
        keypoints = keypoints.to(device)
        visibility = visibility.to(device)

        # Random masking: 30% of visible joints
        with torch.no_grad():
            mask_ratio = 0.3
            random_mask = (torch.rand_like(visibility) < mask_ratio).float()
            reconstruct_mask = random_mask * visibility

        keypoints_masked = keypoints.clone()
        keypoints_masked[reconstruct_mask.bool()] = 0.0

        optimizer.zero_grad()

        g_e = encoder(keypoints_masked)
        v_q, vq_loss, encoding_indices = vq(g_e)
        g_h = decoder(v_q)

        mask = reconstruct_mask.unsqueeze(-1)
        loss_per_joint = F.smooth_l1_loss(g_h, keypoints, reduction='none')
        masked_loss = (loss_per_joint * mask).sum() / mask.sum().clamp(min=1.0)

        l_pct = masked_loss + BETA * vq_loss
        l_pct.backward()
        optimizer.step()

        # Learning rate warmup
        if global_step < WARMUP_ITERS:
            warmup_factor = float(global_step + 1) / WARMUP_ITERS
            for pg in optimizer.param_groups:
                pg['lr'] = LR * warmup_factor
        else:
            scheduler.step()

        global_step += 1

        total_loss += l_pct.item()
        loop.set_postfix(loss=l_pct.item())

    avg_loss = total_loss / len(data)
    print(f'Epoch [{epoch+1}/{EPOCHS}] - Avg Loss: {avg_loss:.4f}')

    if (epoch + 1) % 2 == 0:
        os.makedirs('../ckpt', exist_ok=True)
        torch.save({
            'encoder': encoder.state_dict(),
            'quantizer': vq.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss,
            'epoch': epoch+1
        }, f'../ckpt/{epoch+1}.pt')