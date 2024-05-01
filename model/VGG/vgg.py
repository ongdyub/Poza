import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import os
from torch import nn, optim
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import pickle
import librosa

class AudioDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.wav')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filepath = os.path.join(self.directory, self.filenames[idx])
        # waveform, sample_rate = torchaudio.load(filepath)
        y, sr = librosa.load(filepath)
        y = librosa.resample(y, orig_sr=sr, target_sr=44100)
        sr = 44100
        S = librosa.feature.melspectrogram(y=y, n_mels=512, sr=sr, n_fft=1024, hop_length=256)
        # transform = torchaudio.transforms.Spectrogram(n_fft=4096)
        # specs = transform(waveform)
        # specs = specs / torch.max(specs)
        return S

def create_dataloaders(directory, batch_size=8, split_ratio=0.9):
    dataset = AudioDataset(directory=directory)
    total_count = len(dataset)
    train_count = int(total_count * split_ratio)
    test_count = total_count - train_count
    train_dataset, test_dataset = random_split(dataset, [train_count, test_count])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class VGG11_VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super(VGG11_VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_mu = nn.Linear(512 * 16 * 26, latent_dim)
        self.fc_logvar = nn.Linear(512 * 16 * 26, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 512 * 16 * 26)

        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            
            nn.Upsample(size=(32, 53)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),
            
            nn.Upsample(size=(64, 107)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            
            nn.Upsample(size=(128, 215)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            
            
            nn.Upsample(size=(256, 431)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1),
            
            nn.Upsample(size=(512, 862)),
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder_input(z)
        x = x.view(-1, 512, 16, 26)
        x = self.decoder(x)
        return x, mu, logvar
    

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

def train_vae(model, train_loader, test_loader, epochs, device):
    model.to(device)
    model.train()

    best_test_loss = float('inf')
    patience_counter = 0
    train_loss = []
    test_loss = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for specs in train_loader:
            specs = specs.to(device)
            specs = specs.unsqueeze(1)
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(specs)
            # print(reconstructed.shape)
            # print(specs.shape)
            loss = vae_loss(reconstructed, specs, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        train_loss.append(total_train_loss / len(train_loader.dataset))
        print(f'Epoch {epoch+1}, Train Loss: {train_loss[-1]}')

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for specs in test_loader:
                specs = specs.to(device)
                specs = specs.unsqueeze(1)
                reconstructed, mu, logvar = model(specs)
                loss = vae_loss(reconstructed, specs, mu, logvar)
                total_test_loss += loss.item()

        test_loss.append(total_test_loss / len(test_loader.dataset))
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Test Loss: {test_loss[-1]}')

        # scheduler.step()

        if test_loss[-1] < best_test_loss:
            best_test_loss = test_loss[-1]
            patience_counter = 0

            torch.save(model.state_dict(), f'./out/violin/model_{epoch+1}_{test_loss[-1]:.6f}.pt')
            print(f'Model saved: Epoch {epoch+1} with Test Loss: {test_loss[-1]}')
            
            with open('./out/violin/train_loss.pkl', 'wb') as file:
                pickle.dump(train_loss, file)
            with open('./out/violin/test_loss.pkl', 'wb') as file:
                pickle.dump(test_loss, file)
            
        else:
            patience_counter += 1
            
            with open('./out/violin/train_loss.pkl', 'wb') as file:
                pickle.dump(train_loss, file)
            with open('./out/violin/test_loss.pkl', 'wb') as file:
                pickle.dump(test_loss, file)
        
        # if patience_counter >= 5:
        #     with open('./out/violin/train_loss.pkl', 'wb') as file:
        #         pickle.dump(train_loss, file)
        #     with open('./out/violin/test_loss.pkl', 'wb') as file:
        #         pickle.dump(test_loss, file)
        #     print(f'Early stopping triggered after {epoch+1} epochs.')
        #     break

directory = '../../data/violin'
train_loader, test_loader = create_dataloaders(directory)
train_loss = []
test_loss = []

model = VGG11_VAE(256)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=0.00001)

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
epochs = 10000
print(device)

train_vae(model, train_loader, test_loader, epochs, device)