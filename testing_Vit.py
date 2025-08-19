import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from torch.amp import autocast
from tqdm import tqdm

class Config:
    """Configuration for testing"""
    def __init__(self):
        self.data_dir = "./deepfake_dataset"
        self.batch_size = 8
        self.model_path = 'deepfake_vit_model.pt'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Same splits as training
        self.train_split = 0.7
        self.val_split = 0.2
        self.test_split = 0.1

class DeepfakeDataset(Dataset):
    """Dataset for testing"""
    def __init__(self, data_dir, config, transform=None, split_type='test'):
        self.transform = transform
        
        # Load all images
        real_images = [os.path.join(data_dir, "real", f) for f in os.listdir(os.path.join(data_dir, "real")) if f.endswith(('.jpg', '.png'))]
        fake_images = [os.path.join(data_dir, "deepfake", f) for f in os.listdir(os.path.join(data_dir, "deepfake")) if f.endswith(('.jpg', '.png'))]
        
        all_images = real_images + fake_images
        all_labels = [0] * len(real_images) + [1] * len(fake_images)
        
        # Same train/val/test split as training
        train_val_indices, test_indices, train_val_labels, test_labels = train_test_split(
            range(len(all_images)), all_labels, test_size=config.test_split, 
            stratify=all_labels, random_state=42
        )
        
        train_indices, val_indices, train_labels, val_labels = train_test_split(
            train_val_indices, train_val_labels,
            test_size=config.val_split / (config.train_split + config.val_split),
            stratify=train_val_labels, random_state=42
        )
        
        if split_type == 'train':
            self.images = [all_images[i] for i in train_indices]
            self.labels = train_labels
        elif split_type == 'val':
            self.images = [all_images[i] for i in val_indices]
            self.labels = val_labels
        else:  # test
            self.images = [all_images[i] for i in test_indices]
            self.labels = test_labels
        
        print(f"Loaded {len(self.images)} {split_type} samples")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = cv2.imread(self.images[idx])
            if image is None:
                return None, None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                image = self.transform(image)
            
            return image, self.labels[idx]
        except Exception as e:
            print(f"Error loading {self.images[idx]}: {e}")
            return None, None

def test_model(model, test_loader, config):
    """Test the model and get predictions"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("Testing your trained model...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            if images is None or labels is None:
                continue
                
            images, labels = images.to(config.device), labels.to(config.device)
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs, _ = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

def plot_confusion_matrix(y_true, y_pred, classes=['Real', 'Deepfake']):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Deepfake Detection Model', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    # Add accuracy info
    accuracy = accuracy_score(y_true, y_pred)
    plt.suptitle(f'Model Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)', 
                 fontsize=18, y=0.95, fontweight='bold')
    
    # Add counts and percentages to each cell
    total = cm.sum()
    for i in range(len(classes)):
        for j in range(len(classes)):
            percentage = (cm[i, j] / total) * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=12, color='red')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_detailed_results(y_true, y_pred, y_prob):
    """Print detailed classification results"""
    print("\n" + "="*70)
    print("🎯 DEEPFAKE DETECTION MODEL TEST RESULTS 🎯")
    print("="*70)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"🚀 OVERALL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Confusion matrix values
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📊 CONFUSION MATRIX BREAKDOWN:")
    print(f"✅ True Negatives (Real → Real): {tn}")
    print(f"❌ False Positives (Real → Fake): {fp}")  
    print(f"❌ False Negatives (Fake → Real): {fn}")
    print(f"✅ True Positives (Fake → Fake): {tp}")
    
    # Calculate additional metrics
    if tp + fn > 0:
        sensitivity = tp / (tp + fn)  # Recall for deepfakes
        print(f"🔍 Deepfake Detection Rate: {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    
    if tn + fp > 0:
        specificity = tn / (tn + fp)  # True negative rate
        print(f"🛡️  Real Image Protection Rate: {specificity:.4f} ({specificity*100:.2f}%)")
    
    # Per-class metrics
    print(f"\n📈 DETAILED CLASSIFICATION REPORT:")
    print("-" * 70)
    print(classification_report(y_true, y_pred, target_names=['Real', 'Deepfake'], digits=4))
    
    # Confidence statistics
    if len(y_prob) > 0:
        real_confidences = y_prob[y_true == 0, 0]  # Real class probabilities for real images
        fake_confidences = y_prob[y_true == 1, 1]  # Fake class probabilities for fake images
        
        print(f"💪 CONFIDENCE STATISTICS:")
        print(f"Real images average confidence: {np.mean(real_confidences):.4f} ({np.mean(real_confidences)*100:.2f}%)")
        print(f"Fake images average confidence: {np.mean(fake_confidences):.4f} ({np.mean(fake_confidences)*100:.2f}%)")

def test_single_image(model, image_path, transform, config):
    """Test model on a single image"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = transform(image_rgb).unsqueeze(0).to(config.device)
        
        model.eval()
        with torch.no_grad():
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs, _ = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
        
        pred_class = predicted.item()
        confidence = probabilities[0, pred_class].item()
        real_prob = probabilities[0, 0].item()
        fake_prob = probabilities[0, 1].item()
        
        result = "🟢 REAL" if pred_class == 0 else "🔴 DEEPFAKE"
        print(f"\n📸 SINGLE IMAGE PREDICTION:")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Prediction: {result}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"Real probability: {real_prob:.4f} ({real_prob*100:.2f}%)")
        print(f"Fake probability: {fake_prob:.4f} ({fake_prob*100:.2f}%)")
        
        return pred_class, confidence
        
    except Exception as e:
        print(f"Error testing single image: {e}")
        return None

def main():
    # Initialize config
    config = Config()
    
    # Check if model file exists
    if not os.path.exists(config.model_path):
        print(f"❌ Model file not found: {config.model_path}")
        print("Make sure you have the trained model file!")
        return
    
    print(f"✅ Found trained model: {config.model_path}")
    
    # Test transforms (no augmentation for testing)
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    print("📁 Loading test dataset...")
    test_dataset = DeepfakeDataset(config.data_dir, config, test_transform, split_type='test')
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, 
                            num_workers=2, pin_memory=torch.cuda.is_available())
    
    # Load the TRAINED model - need to recreate architecture to load weights
    print("🔄 Loading your trained model...")
    
    # Import the required modules for model architecture
    from transformers import ViTModel, ViTConfig
    
    # Recreate the exact model architecture from your training
    class AdvancedViTDeepfakeDetector(nn.Module):
        def __init__(self, config):
            super().__init__()
            
            vit_config = ViTConfig(
                image_size=224,
                patch_size=16,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
            )
            
            # Load the pretrained ViT (this is needed for architecture)
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', 
                                               config=vit_config, ignore_mismatched_sizes=True)
            self.fusion_attention = nn.MultiheadAttention(768, 12, dropout=0.1, batch_first=True)
            self.scale_weights = nn.Parameter(torch.ones(3))
            self.norm = nn.LayerNorm(768)
            self.dropout = nn.Dropout(0.1)
            self.fc = nn.Linear(768, 2)

        def forward(self, x, scales=None):
            vit_outputs = self.vit(pixel_values=x)
            cls_token = vit_outputs.last_hidden_state[:, 0, :]
            normalized_features = self.norm(cls_token)
            regularized_features = self.dropout(normalized_features)
            class_logits = self.fc(regularized_features)
            return class_logits, None
    
    # Create model instance
    model = AdvancedViTDeepfakeDetector(config)
    
    # Load your trained weights
    model.load_state_dict(torch.load(config.model_path, map_location=config.device))
    model.to(config.device)
    model.eval()
    
    print(f"✅ Model loaded successfully from {config.model_path}")
    print(f"🖥️  Using device: {config.device}")
    
    # Test the model
    predictions, labels, probabilities = test_model(model, test_loader, config)
    
    # Print results
    print_detailed_results(labels, predictions, probabilities)
    
    # Plot confusion matrix
    plot_confusion_matrix(labels, predictions)
    
    # Test on single images
    print("\n" + "="*70)
    print("🔍 TESTING ON INDIVIDUAL IMAGES")
    print("="*70)
    
    # Find and test some sample images
    real_dir = "./deepfake_dataset/real"
    fake_dir = "./deepfake_dataset/deepfake"
    
    if os.path.exists(real_dir):
        real_files = [f for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png'))][:3]
        for file in real_files:
            test_path = os.path.join(real_dir, file)
            test_single_image(model, test_path, test_transform, config)
    
    if os.path.exists(fake_dir):
        fake_files = [f for f in os.listdir(fake_dir) if f.endswith(('.jpg', '.png'))][:3]
        for file in fake_files:
            test_path = os.path.join(fake_dir, file)
            test_single_image(model, test_path, test_transform, config)
    
    print(f"\n🎉 TESTING COMPLETED!")
    print(f"📊 Confusion matrix saved as 'confusion_matrix.png'")
    print(f"🎯 Final Test Accuracy: {accuracy_score(labels, predictions)*100:.2f}%")

if __name__ == "__main__":
    main()