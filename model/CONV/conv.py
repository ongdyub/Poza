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
import librosa.display

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

def create_dataloaders(directory, batch_size=128, split_ratio=0.9):
    dataset = AudioDataset(directory=directory)
    total_count = len(dataset)
    train_count = int(total_count * split_ratio)
    test_count = total_count - train_count
    train_dataset, test_dataset = random_split(dataset, [train_count, test_count])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class Conv1DNet(nn.Module):
    def __init__(self):
        super(Conv1DNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(512, 512, 2),
            nn.LeakyReLU(),
            nn.Conv1d(512, 512, 2),
            nn.LeakyReLU(),
            nn.Conv1d(512, 256, 2),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 2),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 2),
            nn.LeakyReLU(),
            nn.Conv1d(256, 128, 2),
            nn.LeakyReLU(),
            nn.Conv1d(128, 64, 2),
            nn.LeakyReLU(),
            nn.Conv1d(64, 32, 2),
            nn.LeakyReLU(),
            nn.Conv1d(32, 32, 2),
            nn.LeakyReLU(),
            nn.Conv1d(32, 32, 2),
            nn.LeakyReLU(),
            nn.Conv1d(32, 16, 2),
            nn.LeakyReLU(),
            nn.Conv1d(16, 16, 2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 16, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(16, 32, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(32, 32, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(32, 32, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(32, 64, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(64, 128, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(128, 256, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(256, 256, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(256, 256, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(256, 512, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(512, 512, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(512, 512, 2),
        )


    def forward(self, x):
        latent = self.encoder(x)
        reconst = self.decoder(latent)
        return reconst

def loss(recon_x, x):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    return recon_loss

def train(model, train_loader, test_loader, epochs, device):
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
            # print("Target")
            # print(specs.shape)
            # print(specs.shape)
            optimizer.zero_grad()
            reconstructed = model(specs)
            # print(reconstructed.shape)
            loss = loss(reconstructed, specs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 15)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        train_loss.append(total_train_loss / len(train_loader.dataset))
        print(f'Epoch {epoch+1}, Train Loss: {train_loss[-1]}')

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for specs in test_loader:
                specs = specs.to(device)
                reconstructed = model(specs)
                loss = loss(reconstructed, specs)
                total_test_loss += loss.item()

        test_loss.append(total_test_loss / len(test_loader.dataset))
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Test Loss: {test_loss[-1]}')

        scheduler.step()

        if test_loss[-1] < best_test_loss:
            best_test_loss = test_loss[-1]
            patience_counter = 0

            torch.save(model.state_dict(), f'./out/{inst}/model_{epoch+1}_{test_loss[-1]:.6f}.pt')
            print(f'Model saved: Epoch {epoch+1} with Test Loss: {test_loss[-1]}')
            
            with open(f'./out/{inst}/test_loss.pkl', 'wb') as file:
                pickle.dump(train_loss, file)
            with open(f'./out/{inst}/test_loss.pkl', 'wb') as file:
                pickle.dump(test_loss, file)
            
        else:
            patience_counter += 1
            
            with open(f'./out/{inst}/test_loss.pkl', 'wb') as file:
                pickle.dump(train_loss, file)
            with open(f'./out/{inst}/test_loss.pkl', 'wb') as file:
                pickle.dump(test_loss, file)
        
        # if patience_counter >= 5:
        #     with open('./out/flute/train_loss.pkl', 'wb') as file:
        #         pickle.dump(train_loss, file)
        #     with open('./out/flute/test_loss.pkl', 'wb') as file:
        #         pickle.dump(test_loss, file)
        #     print(f'Early stopping triggered after {epoch+1} epochs.')
        #     break

inst = 'flute'

directory = '../../data/' + inst
train_loader, test_loader = create_dataloaders(directory)
train_loss = []
test_loss = []

model = Conv1DNet()

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=0.000001)

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
epochs = 10000
print(device)

# Train the model
train(model, train_loader, test_loader, epochs, device)
