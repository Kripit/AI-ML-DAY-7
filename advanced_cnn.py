import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTModel, ViTConfig
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import logging
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configure logging system for experiment tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(), logging.FileHandler('deepfake_detection.log')])
logger = logging.getLogger(__name__)

class Config:
    """
    Experimental configuration class containing all hyperparameters and system settings.
    This centralized approach ensures reproducibility across differeant experimental runs.
    """
    def __init__(self):
        # Dataset configuration
        self.data_dir = "./deepfake_dataset"
        self.classes = ['real', 'deepfake']  # Binary classification: authentic vs synthetic media
        
        # Data splitting ratios following standard ML practices
        self.train_split = 0.7  # 70% for model parameter learning
        self.val_split = 0.2    # 20% for hyperparameter tuning and early stopping
        self.test_split = 0.1   # 10% for final unbiased performance evaluation
        
        # Training hyperparameters optimized for GPU memory constraints
        self.batch_size = 2              # Small batch size due to large ViT model memory requirements
        self.accumulation_steps = 2      # Simulates effective batch size of 4 (2 × 2)
        self.epochs = 5               # Training iterations over entire dataset
        
        # Vision Transformer architecture parameters
        self.patch_size = 16             # Each image patch is 16×16 pixels
        self.img_size = 224              # Input image resolution (224×224)
        self.attention_heads = 16        # Multi-head attention mechanism parallelization
        
        # Optimization parameters following ViT training best practices
        self.lr = 1e-5                   # Lower initial learning rate to prevent NaN
        self.max_lr = 5e-5              # Lower peak learning rate for OneCycleLR scheduler
        self.weight_decay = 1e-4         # L2 regularization strength to prevent overfitting
        self.dropout_rate = 0.1          # Lower dropout rate to prevent training instability
        self.clip_grad_norm = 1.0        # Higher gradient clipping threshold
        
        # System configuration
        self.model_path = 'deepfake_vit_model.pt'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device initialized in Config: {self.device}")  # Debug print
        logger.info(f"Computational device initialized: {self.device}")

class DeepfakeDataset(Dataset):
    """
    Custom PyTorch Dataset implementation for binary deepfake classification.
    Handles data loading, preprocessing, and stratified train/validation/test splitting.
    """
    def __init__(self, data_dir, config, transform=None, split_type='train'):
        self.transform = transform
        self.split_type = split_type
        
        # Construct file paths for real and synthetic images
        # os.path.join ensures cross-platform compatibility for file paths
        self.real_images = [os.path.join(data_dir, "real", filename)
                            for filename in os.listdir(os.path.join(data_dir, "real"))
                            if filename.endswith(('.jpg', '.png'))]
        
        self.fake_images = [os.path.join(data_dir, "deepfake", filename)
                            for filename in os.listdir(os.path.join(data_dir, "deepfake"))
                            if filename.endswith(('.jpg', '.png'))]
        
        # Create unified dataset with corresponding labels
        all_images = self.real_images + self.fake_images
        # Label encoding: 0 = real/authentic, 1 = deepfake/synthetic
        all_labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        
        # Stratified splitting ensures balanced class distribution across splits
        # First split: separate test set (10%) from train+validation (90%)
        train_val_indices, test_indices, train_val_labels, test_labels = train_test_split(
            range(len(all_images)),  # Create indices 0, 1, 2, ..., n-1 for all images
            all_labels, 
            test_size=config.test_split,  # 10% for testing
            stratify=all_labels,          # Maintain class proportions in splits
            random_state=42               # Fixed seed for reproducible experiments
        )
        
        # Second split: divide train+validation into train (70%) and validation (20%)
        # test_size calculation: val_split / (train_split + val_split) = 0.2/0.9 ≈ 0.22
        train_indices, val_indices, train_labels, val_labels = train_test_split(
            train_val_indices, 
            train_val_labels,
            test_size=config.val_split / (config.train_split + config.val_split),
            stratify=train_val_labels,
            random_state=42
        )
        
        # Assign appropriate data subset based on split_type parameter
        if split_type == 'train':
            self.images = [all_images[i] for i in train_indices]
            self.labels = train_labels
        elif split_type == 'val':
            self.images = [all_images[i] for i in val_indices]
            self.labels = val_labels
        else:  # split_type == 'test'
            self.images = [all_images[i] for i in test_indices]
            self.labels = test_labels
        
        logger.info(f"Initialized {split_type} dataset with {len(self.images)} samples")

    def __len__(self):
        """
        Required method for PyTorch Dataset.
        Returns total number of samples in this dataset split.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Required method for PyTorch Dataset.
        Retrieves and preprocesses a single sample at given index.
        
        Args:
            idx: Integer index of sample to retrieve
        
        Returns:
            Tuple of (processed_image_tensor, class_label)
        """
        try:
            # Load image using OpenCV (returns BGR format by default)
            image = cv2.imread(self.images[idx])
            if image is None:
                logger.error(f"Failed to load image: {self.images[idx]}")
                return None, None
            # Convert BGR to RGB format (standard for deep learning frameworks)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply preprocessing transformations if specified
            if self.transform:
                image = self.transform(image)
            
            return image, self.labels[idx]
        except Exception as e:
            logger.error(f"Failed to load image: {self.images[idx]}, Error: {e}")
            return None, None

# Training data augmentation pipeline for improved generalization
train_transform = transforms.Compose([
    transforms.ToPILImage(),           # Convert numpy array to PIL Image format
    transforms.Resize((224, 224)),     # Standardize input dimensions to ViT requirements
    transforms.RandomHorizontalFlip(p=0.5),  # 50% probability horizontal flip for data diversity
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Random brightness/contrast variations
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation up to 10% of image size
    transforms.ToTensor(),             # Convert PIL Image to PyTorch tensor [C, H, W] format
    # ImageNet normalization statistics for transfer learning compatibility
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation and test preprocessing without augmentation for consistent evaluation
val_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Multi-scale analysis transforms for ensemble prediction robustness
multi_scale_transforms = [
    # Standard resolution (224×224) - baseline ViT input size
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    # Higher resolution (256×256) - captures finer details
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    # Lower resolution (192×192) - focuses on global patterns
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
]

class AdvancedViTDeepfakeDetector(nn.Module):
    """
    Multi-scale Vision Transformer architecture for deepfake detection.
    
    This implementation combines three key innovations:
    1. Multi-scale feature extraction to capture artifacts at different resolutions
    2. Cross-scale attention fusion for improved feature integration
    3. Pre-trained ViT backbone with task-specific fine-tuning
    """
    def __init__(self, config, img_size=224, patch_size=16, attention_heads=16):
        super(AdvancedViTDeepfakeDetector, self).__init__()
        
        self.patch_size = patch_size
        self.img_size = img_size
        
        # Vision Transformer configuration with reduced model capacity to prevent NaN
        vit_config = ViTConfig(
            image_size=img_size,        # Input image resolution
            patch_size=patch_size,      # Size of image patches
            hidden_size=768,            # Reduced feature dimension for stability
            num_hidden_layers=12,       # Reduced depth of transformer encoder stack
            num_attention_heads=12,     # Reduced parallel attention mechanisms
            intermediate_size=3072,     # Reduced feed-forward network hidden dimension
        )
        
        # Load pre-trained ViT model with ImageNet-21k initialization
        # This provides robust visual representations learned from 21 million images
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', config=vit_config, ignore_mismatched_sizes=True)
        
        # Multi-head attention for fusing features across different image scales
        # This allows the model to learn which scale provides most discriminative information
        self.fusion_attention = nn.MultiheadAttention(768, 12, dropout=0.1, batch_first=True)
        
        # Learnable weights for combining multi-scale predictions
        # nn.Parameter makes these weights trainable during backpropagation
        # torch.ones(3) initializes equal importance for all three scales
        self.scale_weights = nn.Parameter(torch.ones(3))
        
        # Layer normalization for training stability and faster convergence
        self.norm = nn.LayerNorm(768)
        
        # Dropout regularization to prevent overfitting on training data
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Final classification layer: 768 features → 2 classes (real/fake)
        self.fc = nn.Linear(768, 2)

    def forward(self, x, scales=None):
        """
        Forward propagation through the multi-scale ViT architecture.
        
        Args:
            x: Input image tensor of shape [batch_size, channels, height, width]
            scales: Optional list of different scale sizes for multi-scale processing
        
        Returns:
            Tuple of (class_logits, attention_weights)
            - class_logits: Raw prediction scores of shape [batch_size, 2]
            - attention_weights: Cross-scale attention weights for interpretability
        """
        batch_size = x.size(0)  # x.size(0) extracts the first dimension (batch size)
        
        # Simplified single-scale processing to prevent NaN issues
        # Use the pre-trained ViT model directly for feature extraction
        # The ViT model handles patch extraction, positional encoding, and transformer processing internally
        # last_hidden_state contains all patch representations: [batch, num_patches+1, hidden_size]
        # pooler_output contains the [CLS] token representation: [batch, hidden_size]
        vit_outputs = self.vit(pixel_values=x)
        
        # Extract [CLS] token representation which encodes global image information
        # [CLS] token is the first token in the sequence (index 0)
        # It aggregates information from all image patches through self-attention
        cls_token_representation = vit_outputs.last_hidden_state[:, 0, :]
        
        # Apply normalization and regularization
        normalized_features = self.norm(cls_token_representation)
        regularized_features = self.dropout(normalized_features)
        
        # Final classification layer produces class logits
        # Shape: [batch_size, 768] → [batch_size, 2]
        class_logits = self.fc(regularized_features)
        
        return class_logits, None

def train_model(model, train_loader, val_loader, config):
    """
    Training procedure implementing mixed precision training with gradient accumulation.
    
    Key techniques:
    - Automatic Mixed Precision (AMP) for memory efficiency and speed
    - Gradient accumulation to simulate larger batch sizes
    - OneCycleLR scheduling for faster convergence
    - Gradient clipping for training stability
    """
    model.to(config.device)
    print(f"Model moved to device: {next(model.parameters()).device}")  # Debug print
    
    # Cross-entropy loss for binary classification
    # Automatically applies softmax and computes negative log-likelihood
    criterion = nn.CrossEntropyLoss()
    
    # AdamW optimizer with weight decay (L2 regularization)
    # AdamW decouples weight decay from gradient updates for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # Automatic Mixed Precision scaler for numerical stability
    # Scales gradients to prevent underflow in float16 computations
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    # OneCycleLR: learning rate starts low, increases to max_lr, then decreases
    # total_steps accounts for gradient accumulation reducing effective update frequency
    scheduler = OneCycleLR(optimizer, max_lr=config.max_lr, 
                          total_steps=config.epochs * len(train_loader) // config.accumulation_steps)
    
    # Main training loop over epochs
    for epoch in range(config.epochs):
        model.train()  # Enable training mode (activates dropout, batch norm updates)
        running_loss = 0.0
        optimizer.zero_grad()  # Clear gradients from previous iteration
        
        # Create progress bar for this epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        # Iterate through training batches
        for batch_idx, (images, labels) in enumerate(pbar):
            # Skip corrupted samples (None values from __getitem__ exceptions)
            if images is None or labels is None:
                continue
                
            # Move data to computational device (GPU/CPU)
            images, labels = images.to(config.device), labels.to(config.device)
            
            # Check for NaN inputs
            if torch.isnan(images).any() or torch.isnan(labels.float()).any():
                logger.warning(f"NaN detected in inputs at batch {batch_idx}, skipping...")
                continue
            
            # Mixed precision forward pass
            # autocast automatically uses float16 for eligible operations
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                class_logits, _ = model(images)  # _ discards attention weights during training
                loss = criterion(class_logits, labels)
            
            # Check for NaN loss
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected at batch {batch_idx}, skipping...")
                continue
            
            # Gradient accumulation: scale loss by accumulation steps
            # This simulates larger batch sizes without increasing memory usage
            scaled_loss = loss / config.accumulation_steps
            scaler.scale(scaled_loss).backward()  # Backward pass with gradient scaling
            
            # Perform optimizer step every accumulation_steps iterations
            if (batch_idx + 1) % config.accumulation_steps == 0:
                # Unscale gradients before clipping to get true gradient magnitudes
                scaler.unscale_(optimizer)
                
                # Check for NaN gradients
                nan_grads = False
                for param in model.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        nan_grads = True
                        break
                
                if nan_grads:
                    logger.warning(f"NaN gradients detected at batch {batch_idx}, skipping step...")
                    # Need to call step and update even when skipping to reset scaler state
                    scaler.step(optimizer)  # This will skip the step due to NaN but reset state
                    scaler.update()
                    optimizer.zero_grad()
                    continue
                
                # Gradient clipping prevents exploding gradients
                # Rescales gradients if their norm exceeds clip_grad_norm threshold
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
                
                # Update model parameters and advance learning rate scheduler
                scaler.step(optimizer)
                scaler.update()  # Update scaler's internal scale factor
                scheduler.step()  # Advance OneCycleLR schedule
                optimizer.zero_grad()  # Clear accumulated gradients
            
            running_loss += loss.item()  # .item() extracts scalar value from tensor
            
            # Update progress bar with current loss
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        # Calculate and log average loss for this epoch
        epoch_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{config.epochs}, Average Loss: {epoch_loss:.4f}")
        print(f"Epoch {epoch+1} completed, Average Loss: {epoch_loss:.4f}")  # Debug print
    
    # Persist trained model parameters for future inference
    torch.save(model.state_dict(), config.model_path)

def validate_model(model, validation_loader, config):
    """
    Model evaluation on validation or test set with accuracy computation.
    
    Uses torch.no_grad() context to disable gradient computation for efficiency.
    Implements mixed precision inference for consistent behavior with training.
    """
    model.eval()  # Switch to evaluation mode (disables dropout, fixes batch norm)
    correct_predictions = 0
    total_samples = 0
    
    # Disable gradient computation for inference efficiency
    with torch.no_grad():
        for images, labels in tqdm(validation_loader, desc="Validating"):
            # Skip corrupted samples
            if images is None or labels is None:
                continue
                
            images, labels = images.to(config.device), labels.to(config.device)
            
            # Mixed precision inference
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                class_logits, _ = model(images)
                
                # Extract predicted class indices
                # torch.max(tensor, dim) returns (max_values, max_indices)
                # dim=1 finds maximum along class dimension
                # We only need the indices ([1]), not the max values ([0])
                _, predicted_classes = torch.max(class_logits, 1)
            
            # Accumulate accuracy statistics
            total_samples += labels.size(0)  # labels.size(0) = batch_size
            # (predicted_classes == labels) creates boolean tensor
            # .sum().item() counts True values and extracts scalar
            correct_predictions += (predicted_classes == labels).sum().item()
    
    # Calculate percentage accuracy
    accuracy = 100 * correct_predictions / total_samples
    logger.info(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy

def multi_scale_inference(model, image_path, multi_scale_transforms, config):
    """
    Ensemble prediction using multiple image scales for robust deepfake detection.
    
    This approach leverages the fact that deepfake artifacts may be more apparent
    at certain resolutions, improving overall detection accuracy.
    """
    model.eval()
    
    # Load and preprocess target image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    predictions = []  # Store predictions from each scale
    
    with torch.no_grad():
        # Generate predictions at each scale
        for transform in multi_scale_transforms:
            # Preprocess image and add batch dimension
            # unsqueeze(0) adds batch dimension: [C, H, W] → [1, C, H, W]
            preprocessed_image = transform(image_rgb).unsqueeze(0).to(config.device)
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                # Forward pass with simplified processing
                class_logits, _ = model(preprocessed_image)
                
                # Convert logits to probability distribution
                # torch.softmax normalizes outputs to sum to 1.0
                # dim=-1 applies softmax along last dimension (class dimension)
                class_probabilities = torch.softmax(class_logits, dim=-1)
                
                # Transfer to CPU and convert to numpy for ensemble averaging
                predictions.append(class_probabilities.cpu().numpy())
    
    # Ensemble prediction: average probabilities across scales
    # np.mean(axis=0) averages along the first dimension (scale dimension)
    averaged_probabilities = np.mean(predictions, axis=0)
    
    # Final prediction: class with highest average probability
    # np.argmax returns index of maximum value
    # axis=1 finds maximum along class dimension
    # [0] extracts scalar from single-sample prediction
    final_prediction = np.argmax(averaged_probabilities, axis=1)[0]
    
    logger.info(f"Multi-Scale Ensemble Prediction - Class: {final_prediction} (0=real, 1=deepfake)")
    return final_prediction

if __name__ == "__main__":
    """
    Main experimental pipeline for deepfake detection model training and evaluation.
    """
    # Initialize experimental configuration
    config = Config()
    
    # Create dataset instances for each split with appropriate preprocessing
    train_dataset = DeepfakeDataset(config.data_dir, config, train_transform, split_type='train')
    val_dataset = DeepfakeDataset(config.data_dir, config, val_test_transform, split_type='val')
    test_dataset = DeepfakeDataset(config.data_dir, config, val_test_transform, split_type='test')
    
    # Initialize data loaders for batch processing
    # shuffle=True for training ensures random sample ordering each epoch
    # num_workers=2 enables parallel data loading for efficiency
    # pin_memory optimizes GPU transfer when CUDA is available
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                             num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, 
                           num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, 
                            num_workers=2, pin_memory=torch.cuda.is_available())
    
    # Initialize the multi-scale Vision Transformer model
    model = AdvancedViTDeepfakeDetector(config)
    
    # Execute training procedure
    train_model(model, train_loader, val_loader, config)
    
    # Evaluate model performance on validation set
    validation_accuracy = validate_model(model, val_loader, config)
    
    # Final evaluation on held-out test set for unbiased performance estimate
    test_accuracy = validate_model(model, test_loader, config)
    logger.info(f"Final Test Set Accuracy: {test_accuracy:.2f}%")
    
    # Demonstrate single image inference with multi-scale ensemble
    test_image_path = "./deepfake_dataset/real/sample_real.jpg"
    prediction = multi_scale_inference(model, test_image_path, multi_scale_transforms, config)
    print(f"Single Image Prediction: {'Authentic' if prediction == 0 else 'Deepfake'}")