import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

class McDonaldsReviewAnalyzer(nn.Module):
    """
    BERT-based multi-task model for simultaneously classifying McDonald's review attributes
    and sentiment.
    """
    def __init__(self, bert_model_name='bert-base-uncased', num_attributes=6, num_sentiments=3, 
                 dropout_rate=0.1, freeze_bert=False):
        super(McDonaldsReviewAnalyzer, self).__init__()
        
        # Load BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Get BERT's hidden size for task-specific heads
        hidden_size = self.bert.config.hidden_size
        
        # Freeze BERT if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Attribute classification head
        self.attribute_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_attributes)
        )
        
        # Sentiment analysis head
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_sentiments)
        )
    
    def forward(self, input_ids, attention_mask):
        """Forward pass through the model, returning logits for both tasks."""
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use the pooled output for classification
        pooled_output = outputs.pooler_output
        
        # Get predictions for each task
        attribute_logits = self.attribute_classifier(pooled_output)
        sentiment_logits = self.sentiment_classifier(pooled_output)
        
        return attribute_logits, sentiment_logits
    
    def unfreeze_bert_layers(self, num_layers=3):
        """Gradually unfreeze the last n BERT layers for fine-tuning.
        
        I've implemented this to support the transfer learning strategy where
        we gradually unfreeze deeper layers of BERT to fine-tune them.
        """
        # First, make sure all of BERT is frozen
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Then unfreeze the last num_layers encoder layers
        unfreeze_layers = ['pooler'] + [f'encoder.layer.{i}.' for i in range(12 - num_layers, 12)]
        
        for name, param in self.bert.named_parameters():
            for layer in unfreeze_layers:
                if layer in name:
                    param.requires_grad = True
                    break


class TaskWeightScheduler:
    """Dynamically adjusts the weight of each task's loss based on their relative performance.
    
    I created this because I noticed that in multi-task learning, one task can sometimes
    dominate the other. This helps balance their contributions to the overall loss.
    """
    def __init__(self, initial_weights=[0.5, 0.5], strategy='uncertainty', window_size=5):
        self.weights = initial_weights
        self.strategy = strategy
        self.history = {'attribute_loss': [], 'sentiment_loss': []}
        self.window_size = window_size
    
    def update(self, attribute_loss, sentiment_loss):
        """Update task weights based on recent loss history."""
        self.history['attribute_loss'].append(attribute_loss)
        self.history['sentiment_loss'].append(sentiment_loss)
        
        if self.strategy == 'fixed':
            # No weight adjustment
            return self.weights
            
        elif self.strategy == 'uncertainty':
            # Use recent loss history to estimate uncertainty
            window = min(self.window_size, len(self.history['attribute_loss']))
            
            a_uncertainty = np.mean(self.history['attribute_loss'][-window:])
            s_uncertainty = np.mean(self.history['sentiment_loss'][-window:])
            
            # Higher uncertainty (loss) gets lower weight to prevent domination
            total = a_uncertainty + s_uncertainty
            # Uncertainty weighting formula: weight ∝ 1/uncertainty
            self.weights = [s_uncertainty/total, a_uncertainty/total]
            
        return self.weights


def train_and_evaluate(model, train_loader, val_loader, epochs=3, lr=2e-5, 
                      weight_decay=0.01, task_weight_strategy='uncertainty',
                      device=None, model_save_path='models'):
    """Complete training and evaluation function for the multi-task model.
    
    I've designed this to handle the entire training loop with proper logging,
    validation, and early stopping to ensure optimal model performance.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    model.to(device)
    
    # Create model save directory
    os.makedirs(model_save_path, exist_ok=True)
    
    # Initialize optimizer with weight decay for regularization
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Loss functions for both tasks
    attribute_criterion = nn.CrossEntropyLoss()
    sentiment_criterion = nn.CrossEntropyLoss()
    
    # Task weight scheduler
    weight_scheduler = TaskWeightScheduler(strategy=task_weight_strategy)
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Tracking best model performance
    best_val_f1 = 0
    best_epoch = 0
    training_history = {
        'train_loss': [], 'val_loss': [],
        'train_attribute_acc': [], 'val_attribute_acc': [],
        'train_sentiment_acc': [], 'val_sentiment_acc': [],
        'train_attribute_f1': [], 'val_attribute_f1': [],
        'train_sentiment_f1': [], 'val_sentiment_f1': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_metrics = _run_epoch(
            model, train_loader, attribute_criterion, sentiment_criterion, 
            optimizer, scheduler, weight_scheduler, device, is_training=True
        )
        
        # Validation phase
        model.eval()
        val_loss, val_metrics = _run_epoch(
            model, val_loader, attribute_criterion, sentiment_criterion,
            None, None, None, device, is_training=False
        )
        
        # Update training history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['train_attribute_acc'].append(train_metrics['attribute_accuracy'])
        training_history['val_attribute_acc'].append(val_metrics['attribute_accuracy'])
        training_history['train_sentiment_acc'].append(train_metrics['sentiment_accuracy'])
        training_history['val_sentiment_acc'].append(val_metrics['sentiment_accuracy'])
        training_history['train_attribute_f1'].append(train_metrics['attribute_f1'])
        training_history['val_attribute_f1'].append(val_metrics['attribute_f1'])
        training_history['train_sentiment_f1'].append(train_metrics['sentiment_f1'])
        training_history['val_sentiment_f1'].append(val_metrics['sentiment_f1'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Attribute - Acc: {train_metrics['attribute_accuracy']:.4f}, F1: {train_metrics['attribute_f1']:.4f}")
        print(f"Val Attribute - Acc: {val_metrics['attribute_accuracy']:.4f}, F1: {val_metrics['attribute_f1']:.4f}")
        print(f"Train Sentiment - Acc: {train_metrics['sentiment_accuracy']:.4f}, F1: {train_metrics['sentiment_f1']:.4f}")
        print(f"Val Sentiment - Acc: {val_metrics['sentiment_accuracy']:.4f}, F1: {val_metrics['sentiment_f1']:.4f}")
        
        # Save best model based on validation F1 score (average of both tasks)
        val_avg_f1 = (val_metrics['attribute_f1'] + val_metrics['sentiment_f1']) / 2
        if val_avg_f1 > best_val_f1:
            best_val_f1 = val_avg_f1
            best_epoch = epoch
            torch.save(model.state_dict(), f"{model_save_path}/mcd_mtl_best.pt")
            print(f"New best model saved with average F1: {val_avg_f1:.4f}")
        
        # Always save the latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_history': training_history,
            'best_val_f1': best_val_f1,
            'best_epoch': best_epoch
        }, f"{model_save_path}/mcd_mtl_latest.pt")
        
    print(f"\nTraining complete. Best model from epoch {best_epoch+1} with avg F1: {best_val_f1:.4f}")
    
    # Load the best model for return
    model.load_state_dict(torch.load(f"{model_save_path}/mcd_mtl_best.pt"))
    
    return model, training_history


def _run_epoch(model, dataloader, attribute_criterion, sentiment_criterion, 
               optimizer=None, scheduler=None, weight_scheduler=None, 
               device=None, is_training=True):
    """Helper function to run a single training or validation epoch."""
    
    epoch_loss = 0
    attribute_preds_all, attribute_labels_all = [], []
    sentiment_preds_all, sentiment_labels_all = [], []
    
    # Wrap dataloader with tqdm for progress bar
    loop = tqdm(dataloader, desc="Training" if is_training else "Validation")
    
    # For tracking individual task losses
    attribute_loss_sum = 0
    sentiment_loss_sum = 0
    
    for batch in loop:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        attribute_labels = batch['attribute_label'].to(device)
        sentiment_labels = batch['sentiment_label'].to(device)
        
        # Zero gradients if training
        if is_training and optimizer:
            optimizer.zero_grad()
        
        # Forward pass
        with torch.set_grad_enabled(is_training):
            attribute_logits, sentiment_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Calculate individual task losses
            attribute_loss = attribute_criterion(attribute_logits, attribute_labels)
            sentiment_loss = sentiment_criterion(sentiment_logits, sentiment_labels)
            
            # Determine task weights
            if is_training and weight_scheduler:
                task_weights = weight_scheduler.update(attribute_loss.item(), sentiment_loss.item())
            else:
                task_weights = [0.5, 0.5]  # Default equal weights for validation
            
            # Combined loss with task weights
            loss = task_weights[0] * attribute_loss + task_weights[1] * sentiment_loss
        
        # Backward pass and optimization if training
        if is_training and optimizer:
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
        
        # Update progress bar
        loop.set_postfix(loss=loss.item())
        
        # Track metrics
        epoch_loss += loss.item()
        attribute_loss_sum += attribute_loss.item()
        sentiment_loss_sum += sentiment_loss.item()
        
        # Store predictions for metrics calculation
        attribute_preds = torch.argmax(attribute_logits, dim=1).cpu().numpy()
        sentiment_preds = torch.argmax(sentiment_logits, dim=1).cpu().numpy()
        
        attribute_preds_all.extend(attribute_preds)
        attribute_labels_all.extend(attribute_labels.cpu().numpy())
        sentiment_preds_all.extend(sentiment_preds)
        sentiment_labels_all.extend(sentiment_labels.cpu().numpy())
    
    # Calculate metrics
    metrics = {}
    metrics['attribute_accuracy'] = accuracy_score(attribute_labels_all, attribute_preds_all)
    metrics['attribute_f1'] = f1_score(attribute_labels_all, attribute_preds_all, average='weighted')
    metrics['sentiment_accuracy'] = accuracy_score(sentiment_labels_all, sentiment_preds_all)
    metrics['sentiment_f1'] = f1_score(sentiment_labels_all, sentiment_preds_all, average='weighted')
    metrics['attribute_confusion'] = confusion_matrix(attribute_labels_all, attribute_preds_all)
    metrics['sentiment_confusion'] = confusion_matrix(sentiment_labels_all, sentiment_preds_all)
    
    # Average loss for the epoch
    avg_loss = epoch_loss / len(dataloader)
    
    return avg_loss, metrics


def plot_training_history(history):
    """Visualize training progress with accuracy and loss plots."""
    plt.figure(figsize=(15, 10))
    
    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history['train_attribute_acc'], label='Train Attribute Acc')
    plt.plot(history['val_attribute_acc'], label='Val Attribute Acc')
    plt.title('Attribute Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history['train_sentiment_acc'], label='Train Sentiment Acc')
    plt.plot(history['val_sentiment_acc'], label='Val Sentiment Acc')
    plt.title('Sentiment Analysis Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot F1 scores
    plt.subplot(2, 2, 3)
    plt.plot(history['train_attribute_f1'], label='Train Attribute F1')
    plt.plot(history['val_attribute_f1'], label='Val Attribute F1')
    plt.title('Attribute Classification F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(history['train_sentiment_f1'], label='Train Sentiment F1')
    plt.plot(history['val_sentiment_f1'], label='Val Sentiment F1')
    plt.title('Sentiment Analysis F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    return plt

def predict_review(model, tokenizer, review_text, device=None, attribute_map=None, sentiment_map=None):
    """Make predictions on a single review text."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model.to(device)
    
    # Tokenize the input text
    inputs = tokenizer(
        review_text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Get predictions
    with torch.no_grad():
        attribute_logits, sentiment_logits = model(input_ids, attention_mask)
    
    # Get predicted classes
    attribute_pred = torch.argmax(attribute_logits, dim=1).cpu().item()
    sentiment_pred = torch.argmax(sentiment_logits, dim=1).cpu().item()
    
    # Get prediction probabilities
    attribute_probs = torch.softmax(attribute_logits, dim=1).cpu().numpy()[0]
    sentiment_probs = torch.softmax(sentiment_logits, dim=1).cpu().numpy()[0]
    
    # Map predictions to labels if maps are provided
    if attribute_map and 'idx_to_attribute' in attribute_map:
        attribute_label = attribute_map['idx_to_attribute'][attribute_pred]
    else:
        attribute_label = f"Class {attribute_pred}"
        
    if sentiment_map and 'idx_to_sentiment' in sentiment_map:
        sentiment_label = sentiment_map['idx_to_sentiment'][sentiment_pred]
    else:
        sentiment_label = f"Class {sentiment_pred}"
    
    return {
        'attribute': {
            'label': attribute_label,
            'idx': attribute_pred,
            'probabilities': attribute_probs
        },
        'sentiment': {
            'label': sentiment_label,
            'idx': sentiment_pred,
            'probabilities': sentiment_probs
        }
    }
