import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
import numpy as np
from PIL import Image
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models import resnet50
import json
from collections import Counter
from tqdm import tqdm
import random
import torch.jit
from torch.cuda.amp import GradScaler, autocast
import multiprocessing as mp

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

class MultiModalFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(weights='IMAGENET1K_V1')
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()
        
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        
        self.hog_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.fft_extractor = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
    def extract_fft_features(self, x):
        fft = torch.fft.fft2(x.squeeze(1))
        fft_real = fft.real.unsqueeze(1)
        fft_imag = fft.imag.unsqueeze(1)
        fft_combined = torch.cat([fft_real, fft_imag], dim=1)
        return fft_combined
    
    def forward(self, x):
        resnet_features = self.resnet(x)
        
        hog_features = self.hog_extractor(x)
        
        fft_input = self.extract_fft_features(x)
        fft_features = self.fft_extractor(fft_input)
        
        return {
            'resnet': resnet_features,
            'hog': hog_features,
            'fft': fft_features
        }

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        
    def forward(self, query, key, value):
        B = query.shape[0]
        
        Q = self.q(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.dim)
        out = self.out(out)
        
        return out.squeeze(1) if out.size(1) == 1 else out.mean(1)

class AttentionBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 512)
        self.norm = nn.BatchNorm1d(512)
        self.attention = CrossAttention(512)
        self.output_proj = nn.Linear(512, output_dim)
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x):
        identity = self.residual(x)
        
        x = self.linear(x)
        x = self.norm(x)
        x = F.relu(x)
        
        x = x.unsqueeze(1)
        x = self.attention(x, x, x)
        x = self.output_proj(x)
        
        return x + identity

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        B = x.shape[0]
        
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        
        attention = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.dim)
        out = self.out(out)
        
        return self.norm(out.squeeze(1) if out.size(1) == 1 else out.mean(1))

class FusionNetwork(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)
        
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.2)
        
        self.skip1 = nn.Linear(input_dim, 512)
        self.skip2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        identity1 = self.skip1(x)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = x + identity1
        x = self.dropout2(x)
        
        identity2 = self.skip2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        
        return x + identity2

class AdvancedDocumentForgeryDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.feature_extractor = MultiModalFeatureExtractor()
        
        self.resnet_processor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512)
        )
        
        self.hog_processor = nn.Sequential(
            nn.Linear(64, 512),
            nn.BatchNorm1d(512)
        )
        
        self.fft_processor = nn.Sequential(
            nn.Linear(64, 512),
            nn.BatchNorm1d(512)
        )
        
        self.resnet_hog_attention = AttentionBlock(512, 1024)
        self.resnet_fft_attention = AttentionBlock(512, 1024)
        self.hog_fft_attention = AttentionBlock(512, 1024)
        
        self.attended_resnet_hog = AttentionBlock(1024, 1024)
        self.attended_resnet_fft = AttentionBlock(1024, 1024)
        self.attended_hog_fft = AttentionBlock(1024, 1024)
        
        self.multi_head_attention = MultiHeadSelfAttention(3072)
        
        self.fusion_network = FusionNetwork(3072, num_classes)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        
        resnet_feat = self.resnet_processor(features['resnet'])
        hog_feat = self.hog_processor(features['hog'])
        fft_feat = self.fft_processor(features['fft'])
        
        resnet_hog_cross = self.resnet_hog_attention(resnet_feat)
        resnet_fft_cross = self.resnet_fft_attention(resnet_feat)
        hog_fft_cross = self.hog_fft_attention(hog_feat)
        
        attended_resnet_hog = self.attended_resnet_hog(resnet_hog_cross)
        attended_resnet_fft = self.attended_resnet_fft(resnet_fft_cross)
        attended_hog_fft = self.attended_hog_fft(hog_fft_cross)
        
        concatenated = torch.cat([attended_resnet_hog, attended_resnet_fft, attended_hog_fft], dim=1)
        
        attended_features = self.multi_head_attention(concatenated.unsqueeze(1))
        
        output = self.fusion_network(attended_features)
        
        return output

class FastForgeryDataset(Dataset):
    def __init__(self, annotation_file, image_folder, transform=None, patch_size=224, max_patches=15000, augment=False):
        self.image_folder = image_folder
        self.transform = transform
        self.patch_size = patch_size
        self.augment = augment
        
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            if ann['category_id'] == 1:
                img_id = ann['image_id']
                if img_id not in self.image_annotations:
                    self.image_annotations[img_id] = []
                self.image_annotations[img_id].append(ann['bbox'])
        
        self.image_files = {img['id']: img['file_name'] for img in self.coco_data['images']}
        self.samples = self._generate_fast_patches(max_patches)
        
        if self.augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.2),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
        else:
            self.base_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
    
    def _generate_fast_patches(self, max_patches):
        positive_samples = []
        negative_samples = []
        
        valid_images = [(img_id, filename) for img_id, filename in self.image_files.items() 
                       if img_id in self.image_annotations and 
                       os.path.exists(os.path.join(self.image_folder, filename))]
        
        for img_id, filename in tqdm(valid_images[:min(300, len(valid_images))], desc="Fast patch generation"):
            image_path = os.path.join(self.image_folder, filename)
            
            try:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                h, w = img.shape
                if h < self.patch_size or w < self.patch_size:
                    continue
                
                bboxes = self.image_annotations[img_id]
                
                for bbox in bboxes[:3]:
                    x, y, bw, bh = bbox
                    for _ in range(5):
                        center_x = int(x + bw // 2 + random.randint(-10, 10))
                        center_y = int(y + bh // 2 + random.randint(-10, 10))
                        
                        patch_x = max(0, min(w - self.patch_size, center_x - self.patch_size // 2))
                        patch_y = max(0, min(h - self.patch_size, center_y - self.patch_size // 2))
                        
                        positive_samples.append({
                            'image_path': image_path,
                            'patch_coords': (patch_x, patch_y),
                            'label': 1
                        })
                
                for _ in range(10):
                    patch_x = random.randint(0, w - self.patch_size)
                    patch_y = random.randint(0, h - self.patch_size)
                    
                    patch_coords = (patch_x, patch_y, patch_x + self.patch_size, patch_y + self.patch_size)
                    
                    overlaps = False
                    for bbox in bboxes:
                        if self._fast_overlap_check(patch_coords, bbox):
                            overlaps = True
                            break
                    
                    if not overlaps:
                        negative_samples.append({
                            'image_path': image_path,
                            'patch_coords': (patch_x, patch_y),
                            'label': 0
                        })
            except:
                continue
        
        min_samples = min(len(positive_samples), len(negative_samples), max_patches // 2)
        balanced_samples = (random.sample(positive_samples, min_samples) + 
                          random.sample(negative_samples, min_samples))
        random.shuffle(balanced_samples)
        
        return balanced_samples
    
    def _fast_overlap_check(self, patch_coords, bbox):
        px1, py1, px2, py2 = patch_coords
        bx, by, bw, bh = bbox
        return not (px2 <= bx or px1 >= bx + bw or py2 <= by or py1 >= by + bh)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = cv2.imread(sample['image_path'], cv2.IMREAD_GRAYSCALE)
        px, py = sample['patch_coords']
        patch = image[py:py+self.patch_size, px:px+self.patch_size]
        
        patch_pil = Image.fromarray(patch)
        
        if self.augment:
            patch_tensor = self.augment_transform(patch_pil)
        else:
            patch_tensor = self.base_transform(patch_pil)
        
        return patch_tensor, sample['label']

def create_fast_dataloader(dataset, batch_size=32, shuffle=True):
    labels = [sample['label'] for sample in dataset.samples]
    label_counts = Counter(labels)
    
    class_weights = {0: 1.0/label_counts[0], 1: 1.0/label_counts[1]}
    sample_weights = [class_weights[label] for label in labels]
    
    if shuffle:
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                         num_workers=min(4, mp.cpu_count()), pin_memory=True, 
                         persistent_workers=True, prefetch_factor=2)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=min(4, mp.cpu_count()), pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)

def prepare_fast_dataset():
    train_dataset = FastForgeryDataset(
        '/kaggle/input/iit-jammu/train/_annotations.coco.json',
        '/kaggle/input/iit-jammu/train',
        patch_size=224,
        max_patches=15000,
        augment=True
    )
    
    valid_dataset = FastForgeryDataset(
        '/kaggle/input/iit-jammu/valid/_annotations.coco.json',
        '/kaggle/input/iit-jammu/valid',
        patch_size=224,
        max_patches=3000,
        augment=False
    )
    
    test_dataset = FastForgeryDataset(
        '/kaggle/input/iit-jammu/test/_annotations.coco.json',
        '/kaggle/input/iit-jammu/test',
        patch_size=224,
        max_patches=1000,
        augment=False
    )
    
    print(f"Training patches: {len(train_dataset)}")
    print(f"Validation patches: {len(valid_dataset)}")
    print(f"Test patches: {len(test_dataset)}")
    
    train_labels = [sample['label'] for sample in train_dataset.samples]
    label_counts = Counter(train_labels)
    print(f"Training - Authentic: {label_counts[0]}, Forged: {label_counts[1]}")
    
    return train_dataset, valid_dataset, test_dataset

def visualize_samples(dataset, num_samples=8):
    plt.figure(figsize=(15, 10))
    
    for i in range(min(num_samples, len(dataset))):
        patch, label = dataset[i]
        
        plt.subplot(2, 4, i + 1)
        plt.imshow(patch.squeeze(), cmap='gray')
        plt.title(f"Label: {'Forged' if label == 1 else 'Authentic'}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_patches.png', dpi=150, bbox_inches='tight')
    plt.show()

def train_advanced_model(model, train_loader, valid_loader, num_epochs=30, learning_rate=0.0005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, 
                                                   steps_per_epoch=len(train_loader), 
                                                   epochs=num_epochs)
    
    scaler = GradScaler()
    
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    
    best_valid_acc = 0.0
    patience = 7
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        
        for images, labels in train_pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            current_acc = 100 * train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        
        valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
        
        with torch.no_grad():
            for images, labels in valid_pbar:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()
                
                current_acc = 100 * valid_correct / valid_total
                valid_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        train_acc = 100 * train_correct / train_total
        valid_acc = 100 * valid_correct / valid_total
        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}] Summary:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Valid Loss: {avg_valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%')
        print('-' * 60)
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'advanced_best_model.pth')
            print(f"New best validation accuracy: {best_valid_acc:.2f}%")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return model, train_losses, valid_losses, train_accuracies, valid_accuracies

def evaluate_advanced_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    test_pbar = tqdm(test_loader, desc="Advanced Testing")
    
    with torch.no_grad():
        for images, labels in test_pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with autocast():
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1-Score: {f1:.4f}')
    
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Authentic', 'Forged'], 
                yticklabels=['Authentic', 'Forged'])
    plt.title('Confusion Matrix - Advanced Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('advanced_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return accuracy, precision, recall, f1, all_predictions, all_probabilities

def plot_training_curves(train_losses, valid_losses, train_accs, valid_accs):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(valid_losses, label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy', linewidth=2)
    plt.plot(valid_accs, label='Validation Accuracy', linewidth=2)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

def visualize_test_predictions(test_dataset, model, num_samples=12):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    plt.figure(figsize=(20, 15))
    
    indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    
    for i, idx in enumerate(indices):
        patch, true_label = test_dataset[idx]
        
        with torch.no_grad():
            patch_batch = patch.unsqueeze(0).to(device)
            with autocast():
                output = model(patch_batch)
                probabilities = F.softmax(output, dim=1)
                predicted_label = torch.argmax(output, dim=1).item()
                confidence = probabilities[0][predicted_label].item()
        
        plt.subplot(3, 4, i + 1)
        plt.imshow(patch.squeeze(), cmap='gray')
        
        true_text = 'Forged' if true_label == 1 else 'Authentic'
        pred_text = 'Forged' if predicted_label == 1 else 'Authentic'
        
        color = 'green' if true_label == predicted_label else 'red'
        
        plt.title(f'True: {true_text}\nPred: {pred_text} ({confidence:.2f})', 
                 color=color, fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('advanced_test_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Preparing advanced dataset...")
    train_dataset, valid_dataset, test_dataset = prepare_fast_dataset()
    
    print("\nVisualizing samples...")
    visualize_samples(train_dataset, num_samples=8)
    
    train_loader = create_fast_dataloader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = create_fast_dataloader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = create_fast_dataloader(test_dataset, batch_size=32, shuffle=False)
    
    print("Initializing advanced model...")
    model = AdvancedDocumentForgeryDetector(num_classes=2)
    
    print("Starting advanced training...")
    trained_model, train_losses, valid_losses, train_accs, valid_accs = train_advanced_model(
        model, train_loader, valid_loader, num_epochs=30, learning_rate=0.0005
    )
    
    print("\nPlotting training curves...")
    plot_training_curves(train_losses, valid_losses, train_accs, valid_accs)
    
    print("Loading best model for evaluation...")
    try:
        trained_model.load_state_dict(torch.load('advanced_best_model.pth'))
    except:
        print("Using current model for evaluation...")
    
    print("Advanced evaluation...")
    test_accuracy, test_precision, test_recall, test_f1, predictions, probabilities = evaluate_advanced_model(trained_model, test_loader)
    
    print("\nVisualizing test predictions...")
    visualize_test_predictions(test_dataset, trained_model, num_samples=12)
    
    print("Advanced training completed successfully!")
