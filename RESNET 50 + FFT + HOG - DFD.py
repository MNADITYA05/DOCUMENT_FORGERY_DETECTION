import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import Grayscale
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import os
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from PIL import Image
import hashlib
from collections import Counter
import math
from skimage.feature import hog
from skimage import exposure

def set_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_reproducibility(seed=42)

class AdaptiveDropout(nn.Module):
    def __init__(self, base_dropout=0.1, min_dropout=0.05, max_dropout=0.3):
        super().__init__()
        self.base_dropout = base_dropout
        self.min_dropout = min_dropout
        self.max_dropout = max_dropout
        self.current_dropout = base_dropout
        
    def update_dropout(self, train_acc, val_acc):
        gap = abs(train_acc - val_acc)
        if gap > 0.1 and train_acc > val_acc:
            self.current_dropout = min(self.current_dropout + 0.02, self.max_dropout)
        elif gap < 0.02 and train_acc < 0.95:
            self.current_dropout = max(self.current_dropout - 0.01, self.min_dropout)
        
    def forward(self, x):
        return F.dropout(x, p=self.current_dropout, training=self.training)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return x * self.sigmoid(out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        att = self.sigmoid(self.conv(x_cat))
        return x * att

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class HOGFeatureExtractor(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        self.orientations = 9
        self.pixels_per_cell = (8, 8)
        self.cells_per_block = (2, 2)
        self.block_norm = 'L2-Hys'
        
        self.adaptive_dropout1 = AdaptiveDropout(0.15, 0.05, 0.25)
        self.adaptive_dropout2 = AdaptiveDropout(0.1, 0.03, 0.2)
        self.adaptive_dropout3 = AdaptiveDropout(0.05, 0.01, 0.15)
        
        self.feature_processor = nn.Sequential(
            nn.Linear(34596, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            self.adaptive_dropout1,
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            self.adaptive_dropout2,
            nn.Linear(1024, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            self.adaptive_dropout3
        )

    def update_dropout(self, train_acc, val_acc):
        self.adaptive_dropout1.update_dropout(train_acc, val_acc)
        self.adaptive_dropout2.update_dropout(train_acc, val_acc)
        self.adaptive_dropout3.update_dropout(train_acc, val_acc)

    def compute_hog_features(self, image_batch):
        hog_features = []
        for i in range(image_batch.size(0)):
            img = image_batch[i].squeeze().cpu().numpy()
            img = ((img * 0.5 + 0.5) * 255).astype(np.uint8)
            
            fd = hog(img, orientations=self.orientations, 
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                    block_norm=self.block_norm, visualize=False)
            
            hog_features.append(fd)
        
        hog_tensor = torch.FloatTensor(np.array(hog_features)).to(image_batch.device)
        return hog_tensor

    def forward(self, x):
        x_gray = torch.mean(x, dim=1, keepdim=True)
        x_gray = F.interpolate(x_gray, size=(256, 256), mode='bilinear', align_corners=False)
        
        hog_features = self.compute_hog_features(x_gray)
        
        return self.feature_processor(hog_features)

class MultiScaleFFTExtractor(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        self.scales = [64, 128, 256]
        self.extractors = nn.ModuleList()
        
        for scale in self.scales:
            extractor = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                CBAM(64),
                nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                CBAM(256),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
            )
            self.extractors.append(extractor)
        
        self.adaptive_dropout = AdaptiveDropout(0.1, 0.02, 0.2)
        
        self.fusion = nn.Sequential(
            nn.Linear(256 * len(self.scales), output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            self.adaptive_dropout
        )

    def update_dropout(self, train_acc, val_acc):
        self.adaptive_dropout.update_dropout(train_acc, val_acc)

    def forward(self, x):
        features = []
        for i, extractor in enumerate(self.extractors):
            if i > 0:
                x_resized = F.interpolate(x, size=(self.scales[i], self.scales[i]), mode='bilinear', align_corners=False)
            else:
                x_resized = x
            feat = extractor(x_resized)
            features.append(feat)
        
        combined = torch.cat(features, dim=1)
        return self.fusion(combined)

class MultiScaleResNetFeatures(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])
        self.layer2 = nn.Sequential(*list(resnet.children())[5:6])
        self.layer3 = nn.Sequential(*list(resnet.children())[6:7])
        self.layer4 = nn.Sequential(*list(resnet.children())[7:8])
        
        self.attention1 = CBAM(256)
        self.attention2 = CBAM(512)
        self.attention3 = CBAM(1024)
        self.attention4 = CBAM(2048)
        
        self.lateral1 = nn.Conv2d(256, 256, 1)
        self.lateral2 = nn.Conv2d(512, 256, 1)
        self.lateral3 = nn.Conv2d(1024, 256, 1)
        self.lateral4 = nn.Conv2d(2048, 256, 1)
        
        self.smooth = nn.Conv2d(256, 256, 3, padding=1)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.adaptive_dropout = AdaptiveDropout(0.1, 0.02, 0.2)
        
        self.projection = nn.Sequential(
            nn.Linear(256 * 4, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            self.adaptive_dropout
        )

    def update_dropout(self, train_acc, val_acc):
        self.adaptive_dropout.update_dropout(train_acc, val_acc)

    def forward(self, x):
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        
        c1 = self.attention1(c1)
        c2 = self.attention2(c2)
        c3 = self.attention3(c3)
        c4 = self.attention4(c4)
        
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False)
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        
        p1 = self.smooth(p1)
        p2 = self.smooth(p2)
        p3 = self.smooth(p3)
        p4 = self.smooth(p4)
        
        f1 = self.global_pool(p1).flatten(1)
        f2 = self.global_pool(p2).flatten(1)
        f3 = self.global_pool(p3).flatten(1)
        f4 = self.global_pool(p4).flatten(1)
        
        features = torch.cat([f1, f2, f3, f4], dim=1)
        return self.projection(features)

class TriModalAttention(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.05):
        super().__init__()
        self.cross_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        
        self.adaptive_dropout1 = AdaptiveDropout(dropout, 0.01, 0.15)
        self.adaptive_dropout2 = AdaptiveDropout(dropout, 0.01, 0.15)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            self.adaptive_dropout2,
            nn.Linear(d_model * 4, d_model)
        )

    def update_dropout(self, train_acc, val_acc):
        self.adaptive_dropout1.update_dropout(train_acc, val_acc)
        self.adaptive_dropout2.update_dropout(train_acc, val_acc)

    def forward(self, x1, x2, x3):
        x1_seq = x1.unsqueeze(1)
        x2_seq = x2.unsqueeze(1)
        x3_seq = x3.unsqueeze(1)
        
        cross_out1, _ = self.cross_attn1(x1_seq, x2_seq, x2_seq)
        x1_seq = self.norm1(x1_seq + self.adaptive_dropout1(cross_out1))
        
        cross_out2, _ = self.cross_attn2(x1_seq, x3_seq, x3_seq)
        x1_seq = self.norm2(x1_seq + self.adaptive_dropout1(cross_out2))
        
        self_out, _ = self.self_attn(x1_seq, x1_seq, x1_seq)
        x1_seq = self.norm3(x1_seq + self.adaptive_dropout1(self_out))
        
        ffn_out = self.ffn(x1_seq)
        x1_seq = self.norm4(x1_seq + ffn_out)
        
        return x1_seq.squeeze(1)

class EnhancedDocumentForgeryDetector(nn.Module):
    def __init__(self, num_classes=1, feature_dim=768, nhead=12, dropout=0.05):
        super().__init__()
        
        self.visual_features = MultiScaleResNetFeatures(feature_dim)
        self.fft_features = MultiScaleFFTExtractor(feature_dim)
        self.hog_features = HOGFeatureExtractor(feature_dim)
        self.grayscale = Grayscale(num_output_channels=1)
        
        self.visual_fusion = TriModalAttention(feature_dim, nhead, dropout)
        self.fft_fusion = TriModalAttention(feature_dim, nhead, dropout)
        self.hog_fusion = TriModalAttention(feature_dim, nhead, dropout)
        
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.Sigmoid()
        )
        
        self.adaptive_dropout1 = AdaptiveDropout(0.15, 0.05, 0.3)
        self.adaptive_dropout2 = AdaptiveDropout(0.15, 0.05, 0.25)
        self.adaptive_dropout3 = AdaptiveDropout(0.1, 0.02, 0.2)
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            self.adaptive_dropout1
        )
        
        self.ensemble_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(inplace=True),
                self.adaptive_dropout2,
                nn.Linear(256, num_classes)
            ) for _ in range(3)
        ])
        
        self.meta_classifier = nn.Sequential(
            nn.Linear(feature_dim + 3, 128),
            nn.ReLU(inplace=True),
            self.adaptive_dropout3,
            nn.Linear(128, num_classes)
        )

    def update_dropout(self, train_acc, val_acc):
        self.visual_features.update_dropout(train_acc, val_acc)
        self.fft_features.update_dropout(train_acc, val_acc)
        self.hog_features.update_dropout(train_acc, val_acc)
        self.visual_fusion.update_dropout(train_acc, val_acc)
        self.fft_fusion.update_dropout(train_acc, val_acc)
        self.hog_fusion.update_dropout(train_acc, val_acc)
        self.adaptive_dropout1.update_dropout(train_acc, val_acc)
        self.adaptive_dropout2.update_dropout(train_acc, val_acc)
        self.adaptive_dropout3.update_dropout(train_acc, val_acc)

    def forward(self, x):
        visual_feat = self.visual_features(x)
        
        x_gray = self.grayscale(x)
        fft_input = torch.log1p(torch.abs(torch.fft.fftshift(torch.fft.fft2(x_gray, norm='ortho'), dim=(-2, -1))))
        fft_feat = self.fft_features(fft_input)
        
        hog_feat = self.hog_features(x)
        
        v_fused = self.visual_fusion(visual_feat, fft_feat, hog_feat)
        f_fused = self.fft_fusion(fft_feat, visual_feat, hog_feat)
        h_fused = self.hog_fusion(hog_feat, visual_feat, fft_feat)
        
        gate_weights = self.gate(torch.cat([v_fused, f_fused, h_fused], dim=1))
        
        final_feat = self.feature_fusion(torch.cat([
            gate_weights * v_fused,
            gate_weights * f_fused, 
            gate_weights * h_fused
        ], dim=1))
        
        ensemble_outputs = []
        for head in self.ensemble_heads:
            ensemble_outputs.append(head(final_feat))
        
        ensemble_probs = [torch.sigmoid(out) for out in ensemble_outputs]
        ensemble_tensor = torch.cat(ensemble_probs, dim=1)
        
        meta_input = torch.cat([final_feat, ensemble_tensor], dim=1)
        final_output = self.meta_classifier(meta_input)
        
        return final_output, ensemble_outputs

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

CONFIG = {
    "DATA_PATH": "/kaggle/input/rtm-dataset2/RTM",
    "MODEL_SAVE_PATH": "enhanced_model_adaptive.pth",
    "PLOT_SAVE_PATH": "enhanced_metrics_adaptive.png",
    "CLASS_NAMES_PATH": "class_names.json",
    "IMAGE_SIZE": 256,
    "BATCH_SIZE": 16,
    "HEAD_ONLY_EPOCHS": 12,
    "FULL_TRAIN_EPOCHS": 70,
    "LEARNING_RATE": 3e-4,
    "WEIGHT_DECAY": 3e-5,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "NUM_WORKERS": 4,
    "PREPROC_MEAN": [0.485, 0.456, 0.406],
    "PREPROC_STD": [0.229, 0.224, 0.225],
    "RANDOM_SEED": 42,
}

def get_transforms(image_size, is_train=True):
    mean = CONFIG["PREPROC_MEAN"]
    std = CONFIG["PREPROC_STD"]
    
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            ], p=0.6),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.4),
            ], p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.GaussNoise(var_limit=(10, 20), p=0.2),
            ], p=0.3),
            A.CoarseDropout(max_holes=4, max_height=8, max_width=8, fill_value=0, p=0.2),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ])

class RTMDataset(Dataset):
    def __init__(self, data_dir, transform=None, split_ratio=(0.8, 0.1, 0.1), phase='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        real_folders = ['cover', 'good']
        fake_folders = ['cpmv', 'edit', 'inpaint', 'insert', 'splice']
        
        all_paths_labels = []
        
        for folder in real_folders:
            folder_path = os.path.join(data_dir, folder)
            if os.path.exists(folder_path):
                for img_file in os.listdir(folder_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')) and not img_file.lower().endswith('mask.png'):
                        img_path = os.path.join(folder_path, img_file)
                        all_paths_labels.append((img_path, 0))
        
        for folder in fake_folders:
            folder_path = os.path.join(data_dir, folder)
            if os.path.exists(folder_path):
                for img_file in os.listdir(folder_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')) and not img_file.lower().endswith('mask.png'):
                        img_path = os.path.join(folder_path, img_file)
                        all_paths_labels.append((img_path, 1))
        
        random.shuffle(all_paths_labels)
        
        total_samples = len(all_paths_labels)
        train_end = int(total_samples * split_ratio[0])
        val_end = train_end + int(total_samples * split_ratio[1])
        
        if phase == 'train':
            selected_data = all_paths_labels[:train_end]
        elif phase == 'val':
            selected_data = all_paths_labels[train_end:val_end]
        elif phase == 'test':
            selected_data = all_paths_labels[val_end:]
        
        self.image_paths = [item[0] for item in selected_data]
        self.labels = [item[1] for item in selected_data]
        
        self.class_to_idx = {'real': 0, 'fake': 1}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, torch.tensor(float(label)), img_path

def create_dataloaders(data_path, image_size, batch_size, num_workers):
    dataloaders = {}
    
    for phase in ['train', 'val', 'test']:
        dataset = RTMDataset(data_path, transform=get_transforms(image_size, is_train=(phase == 'train')), phase=phase)
        
        if len(dataset) == 0:
            continue
            
        sampler = None
        if phase == 'train':
            targets = dataset.labels
            class_counts = np.bincount(targets, minlength=2)
            class_weights = 1. / (class_counts + 1e-6)
            sample_weights = np.array([class_weights[t] for t in targets])
            sampler = WeightedRandomSampler(torch.from_numpy(sample_weights).double(), len(sample_weights))
        
        dataloaders[phase] = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            pin_memory=True, 
            sampler=sampler, 
            shuffle=(sampler is None)
        )
    
    if 'train' not in dataloaders:
        return None, None
    
    print(f"Data loaded - Train: {len(dataloaders['train'].dataset)}, Val: {len(dataloaders['val'].dataset)}, Test: {len(dataloaders['test'].dataset)}")
    return dataloaders, list(dataloaders['train'].dataset.class_to_idx.keys())

def train_one_epoch(model, dataloader, criterion, optimizer, device, phase):
    model.train() if phase == 'train' else model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0
    
    for batch_data in tqdm(dataloader, desc=f"{phase.capitalize()}", leave=False):
        if len(batch_data) == 3:
            images, labels, _ = batch_data
        else:
            images, labels = batch_data
            
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        total_samples += images.size(0)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.set_grad_enabled(phase == 'train'):
            main_output, ensemble_outputs = model(images)
            
            main_loss = criterion(main_output, labels)
            ensemble_loss = sum([criterion(out, labels) for out in ensemble_outputs])
            total_loss = main_loss + 0.3 * ensemble_loss
            
            if phase == 'train':
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        
        running_loss += total_loss.item() * images.size(0)
        correct_preds += torch.sum((torch.sigmoid(main_output) > 0.5) == (labels > 0.5))
    
    return running_loss / total_samples, correct_preds.double() / total_samples

def plot_and_save_metrics(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss vs. Epochs')
    ax1.set_xlabel('Epochs')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Accuracy vs. Epochs')
    ax2.set_xlabel('Epochs')
    ax2.legend()
    ax2.grid(True)
    
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to '{save_path}'")

def visualize_test_samples(model, test_dataloader, device, num_samples=16):
    model.eval()
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    
    sample_count = 0
    class_names = ['Real', 'Fake']
    
    with torch.no_grad():
        for batch_data in test_dataloader:
            if len(batch_data) == 3:
                images, labels, paths = batch_data
            else:
                images, labels = batch_data
                paths = [f"sample_{i}" for i in range(len(images))]
            
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            main_output, _ = model(images)
            predictions = torch.sigmoid(main_output) > 0.5
            probabilities = torch.sigmoid(main_output)
            
            for i in range(len(images)):
                if sample_count >= num_samples:
                    break
                
                if sample_count < len(paths) and os.path.exists(paths[i]):
                    original_img = cv2.imread(paths[i])
                    if original_img is not None:
                        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                        original_img = cv2.resize(original_img, (256, 256))
                        
                        true_label = int(labels[i].item())
                        pred_label = int(predictions[i].item())
                        prob = probabilities[i].item()
                        
                        axes[sample_count].imshow(original_img)
                        axes[sample_count].set_title(
                            f'True: {class_names[true_label]}\nPred: {class_names[pred_label]} ({prob:.3f})',
                            color='green' if true_label == pred_label else 'red'
                        )
                        axes[sample_count].axis('off')
                        
                        sample_count += 1
            
            if sample_count >= num_samples:
                break
    
    plt.tight_layout()
    plt.savefig('enhanced_test_samples_adaptive.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    print(f"Using device: {CONFIG['DEVICE']}")
    
    dataloaders, class_names = create_dataloaders(
        CONFIG["DATA_PATH"], 
        CONFIG["IMAGE_SIZE"], 
        CONFIG["BATCH_SIZE"], 
        CONFIG["NUM_WORKERS"]
    )
    
    if not dataloaders:
        print("Failed to create dataloaders")
        return
    
    with open(CONFIG['CLASS_NAMES_PATH'], 'w') as f:
        json.dump(class_names, f)
    
    model = EnhancedDocumentForgeryDetector().to(CONFIG["DEVICE"])
    criterion = nn.BCEWithLogitsLoss().to(CONFIG["DEVICE"])
    
    print("\n--- STAGE 1: Training model heads ---")
    for param in model.visual_features.parameters():
        param.requires_grad = False
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=CONFIG["LEARNING_RATE"], weight_decay=CONFIG["WEIGHT_DECAY"])
    
    for epoch in range(CONFIG["HEAD_ONLY_EPOCHS"]):
        print(f"Head-Only Epoch {epoch+1}/{CONFIG['HEAD_ONLY_EPOCHS']}")
        train_loss, train_acc = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, CONFIG['DEVICE'], 'train'
        )
        print(f'-> Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
    
    print("\n--- STAGE 2: Adaptive fine-tuning full model ---")
    for param in model.parameters():
        param.requires_grad = True
    
    visual_params = list(model.visual_features.parameters())
    other_params = [p for name, p in model.named_parameters() if 'visual_features' not in name]
    
    optimizer = optim.AdamW([
        {'params': visual_params, 'lr': CONFIG["LEARNING_RATE"] / 5},
        {'params': other_params, 'lr': CONFIG["LEARNING_RATE"]}
    ], weight_decay=CONFIG["WEIGHT_DECAY"])
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=5, verbose=True, min_lr=1e-7)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    best_combined_acc = 0.0
    
    for epoch in range(CONFIG["FULL_TRAIN_EPOCHS"]):
        print(f"\nAdaptive-Train Epoch {epoch+1}/{CONFIG['FULL_TRAIN_EPOCHS']}")
        
        train_loss, train_acc = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, CONFIG['DEVICE'], 'train'
        )
        val_loss, val_acc = train_one_epoch(
            model, dataloaders['val'], criterion, optimizer, CONFIG['DEVICE'], 'val'
        )
        
        model.update_dropout(train_acc.item(), val_acc.item())
        
        print(f'-> Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.cpu())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.cpu())
        
        combined_metric = (train_acc + val_acc) / 2 - abs(train_acc - val_acc) * 0.5
        
        scheduler.step(combined_metric)
        
        acc_gap = abs(train_acc - val_acc)
        if acc_gap > 0.15:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.95
        elif acc_gap < 0.05 and train_acc < 0.95:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 1.02
                param_group['lr'] = min(param_group['lr'], CONFIG["LEARNING_RATE"])
        
        if combined_metric > best_combined_acc:
            best_combined_acc = combined_metric
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["MODEL_SAVE_PATH"])
            print(f"🎉 New best model saved with Combined Score: {combined_metric:.4f}")
    
    plot_and_save_metrics(history, CONFIG['PLOT_SAVE_PATH'])
    
    print("\n--- Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load(CONFIG["MODEL_SAVE_PATH"]))
    test_optimizer = optim.AdamW(model.parameters())
    test_loss, test_acc = train_one_epoch(
        model, dataloaders['test'], criterion, test_optimizer, CONFIG['DEVICE'], 'test'
    )
    print(f'Final Test Loss: {test_loss:.4f} | Final Test Acc: {test_acc:.4f}')
    
    print("\n--- Visualizing Test Samples ---")
    visualize_test_samples(model, dataloaders['test'], CONFIG['DEVICE'])

if __name__ == '__main__':
    main()
