import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import numpy as np
import re

class StarbucksReviewsDataset(Dataset):
    """Dataset class for Starbucks reviews with both attribute and sentiment labels."""
    def __init__(self, reviews, attribute_labels, sentiment_labels, tokenizer, max_length=128):
        self.reviews = reviews
        self.attribute_labels = attribute_labels
        self.sentiment_labels = sentiment_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        """Process a single review into model-ready format with both task labels."""
        review = str(self.reviews[idx])
        
        # Tokenize the review text
        encoding = self.tokenizer(
            review,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Return a dictionary with all required inputs and labels
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'attribute_label': torch.tensor(self.attribute_labels[idx], dtype=torch.long),
            'sentiment_label': torch.tensor(self.sentiment_labels[idx], dtype=torch.long)
        }

def preprocess_text(text):
    """Clean review text by removing special characters and normalizing whitespace."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters except spaces and basic punctuation
    text = re.sub(r'[^\w\s.,!?]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_prepare_data(csv_path, tokenizer_name='bert-base-uncased', test_size=0.2, val_size=0.1, 
                          batch_size=32, random_state=42):
    """Load, preprocess, and prepare Starbucks reviews data for multi-task learning."""
    # Load data
    df = pd.read_csv(csv_path)
    
    # Basic preprocessing
    df['review_clean'] = df['sentence'].apply(preprocess_text)
    
    # Using provided attribute and sentiment classes
    # Attribute mapping (already numeric in dataset)
    # Using attr_class instead of attributes because half the attributes row were NaN. Found the class labels by visualizing the data separately.
    attribute_categories = sorted(df['attr_class'].unique())
    idx_to_attribute = {
        0: 'ambiance',
        1: 'food',
        2: 'location',
        3: 'service',
        4: 'price',
        5: 'general'
    }
    attribute_to_idx = {v: k for k, v in idx_to_attribute.items()}
    
    # Sentiment mapping (already numeric in dataset)
    sentiment_categories = sorted(df['sent_class'].unique())
    # Based on examples, assuming:
    # 0: negative, 1: neutral, 2: positive
    idx_to_sentiment = {
        0: 'negative',
        1: 'neutral',
        2: 'positive'
    }
    sentiment_to_idx = {v: k for k, v in idx_to_sentiment.items()}
    
    # Split data: first into train+val and test, then train and val
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[['attr_class', 'sent_class']]
    )
    
    # Calculate validation size relative to train+val
    relative_val_size = val_size / (1 - test_size)
    
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_size, 
        random_state=random_state, stratify=train_val_df[['attr_class', 'sent_class']]
    )
    
    print(f"Data split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    
    # Create datasets
    train_dataset = StarbucksReviewsDataset(
        train_df['review_clean'].values,
        train_df['attr_class'].values,
        train_df['sent_class'].values,
        tokenizer
    )
    
    val_dataset = StarbucksReviewsDataset(
        val_df['review_clean'].values,
        val_df['attr_class'].values,
        val_df['sent_class'].values,
        tokenizer
    )
    
    test_dataset = StarbucksReviewsDataset(
        test_df['review_clean'].values,
        test_df['attr_class'].values,
        test_df['sent_class'].values,
        tokenizer
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Return everything needed for training and evaluation
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'attribute_map': {'idx_to_attribute': idx_to_attribute, 'attribute_to_idx': attribute_to_idx},
        'sentiment_map': {'idx_to_sentiment': idx_to_sentiment, 'sentiment_to_idx': sentiment_to_idx},
        'num_attributes': len(attribute_categories),
        'num_sentiments': len(sentiment_categories),
        'datasets': {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        },
        'dataframes': {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    }