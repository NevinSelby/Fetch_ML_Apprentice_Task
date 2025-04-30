import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import os
import json

# Import our modules
from model import McDonaldsReviewAnalyzer, train_and_evaluate, predict_review, plot_training_history
from data import load_and_prepare_data

def main():
    """Main function that handles the training and evaluation workflow."""
    parser = argparse.ArgumentParser(description='Train a multi-task model for McDonald\'s reviews analysis')
    
    # Dataset parameters
    parser.add_argument('--data_path', type=str, default='mcdonalds_reviews.csv', 
                        help='Path to the McDonald\'s reviews CSV file')
    parser.add_argument('--test_size', type=float, default=0.2, 
                        help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.1, 
                        help='Proportion of data to use for validation')
    
    # Model parameters
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased', 
                        help='BERT model to use')
    parser.add_argument('--freeze_bert', action='store_true', 
                        help='Whether to freeze BERT parameters initially')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='Dropout rate for classification heads')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, 
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, 
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--task_weight_strategy', type=str, default='uncertainty', 
                        choices=['fixed', 'uncertainty'], 
                        help='Strategy for weighting task losses')
    parser.add_argument('--unfreeze_after', type=int, default=1, 
                        help='Epoch after which to unfreeze BERT layers')
    parser.add_argument('--unfreeze_layers', type=int, default=3, 
                        help='Number of BERT layers to unfreeze')
    
    # Output parameters
    parser.add_argument('--model_dir', type=str, default='models', 
                        help='Directory to save models')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    
    # Mode parameters
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'predict'],
                        help='Mode to run the script in')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a saved model, required for evaluate and predict modes')
    parser.add_argument('--review_text', type=str, default=None,
                        help='Text of a review to predict, required for predict mode')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare data
    print("Loading and preparing data...")
    data_bundle = load_and_prepare_data(
        args.data_path,
        tokenizer_name=args.bert_model,
        test_size=args.test_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        random_state=args.seed
    )
    
    # Extract components from data bundle
    train_loader = data_bundle['train_loader']
    val_loader = data_bundle['val_loader']
    test_loader = data_bundle['test_loader']
    attribute_map = data_bundle['attribute_map']
    sentiment_map = data_bundle['sentiment_map']
    num_attributes = data_bundle['num_attributes']
    num_sentiments = data_bundle['num_sentiments']
    
    # Save attribute and sentiment mappings
    with open(f"{args.output_dir}/attribute_map.json", 'w') as f:
        json.dump({str(k): v for k, v in attribute_map['idx_to_attribute'].items()}, f)
    
    with open(f"{args.output_dir}/sentiment_map.json", 'w') as f:
        json.dump({str(k): v for k, v in sentiment_map['idx_to_sentiment'].items()}, f)
    
    print(f"Number of attributes: {num_attributes}")
    print(f"Number of sentiments: {num_sentiments}")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.mode == 'train':
        # Initialize model
        print("Initializing model...")
        model = McDonaldsReviewAnalyzer(
            bert_model_name=args.bert_model,
            num_attributes=num_attributes,
            num_sentiments=num_sentiments,
            dropout_rate=args.dropout,
            freeze_bert=args.freeze_bert
        )
        
        # Train model
        print("Starting training...")
        model, history = train_and_evaluate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            task_weight_strategy=args.task_weight_strategy,
            device=device,
            model_save_path=args.model_dir
        )
        
        # Save training history
        history_df = pd.DataFrame(history)
        history_df.to_csv(f"{args.output_dir}/training_history.csv", index=False)
        
        # Plot and save training curves
        plt_figure = plot_training_history(history)
        plt_figure.savefig(f"{args.output_dir}/training_curves.png")
        plt.close()
        
        print("Training complete!")
        
    elif args.mode == 'evaluate':
        # Load pre-trained model
        if not args.model_path:
            args.model_path = f"{args.model_dir}/mcd_mtl_best.pt"
        
        print(f"Loading model from {args.model_path}...")
        model = McDonaldsReviewAnalyzer(
            bert_model_name=args.bert_model,
            num_attributes=num_attributes,
            num_sentiments=num_sentiments,
            dropout_rate=args.dropout
        )
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        # Evaluate on test set
        print("Evaluating model on test set...")
        model.eval()
        _, test_metrics = _run_epoch(
            model, test_loader, nn.CrossEntropyLoss(), nn.CrossEntropyLoss(),
            None, None, None, device, is_training=False
        )
        
        print("\nTest Set Results:")
        print(f"Attribute Accuracy: {test_metrics['attribute_accuracy']:.4f}")
        print(f"Attribute F1 Score: {test_metrics['attribute_f1']:.4f}")
        print(f"Sentiment Accuracy: {test_metrics['sentiment_accuracy']:.4f}")
        print(f"Sentiment F1 Score: {test_metrics['sentiment_f1']:.4f}")
        
        # Save confusion matrices
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(test_metrics['attribute_confusion'], annot=True, fmt='d', cmap='Blues')
        plt.title('Attribute Classification Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.subplot(1, 2, 2)
        sns.heatmap(test_metrics['sentiment_confusion'], annot=True, fmt='d', cmap='Blues')
        plt.title('Sentiment Analysis Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/confusion_matrices.png")
        plt.close()
        
    elif args.mode == 'predict':
        # Load pre-trained model
        if not args.model_path:
            args.model_path = f"{args.model_dir}/mcd_mtl_best.pt"
        
        print(f"Loading model from {args.model_path}...")
        model = McDonaldsReviewAnalyzer(
            bert_model_name=args.bert_model,
            num_attributes=num_attributes,
            num_sentiments=num_sentiments,
            dropout_rate=args.dropout
        )
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        
        # Make prediction on input text
        if args.review_text:
            print(f"Predicting for review: '{args.review_text}'")
            prediction = predict_review(
                model, tokenizer, args.review_text, device, attribute_map, sentiment_map
            )
            
            print("\nPrediction Results:")
            print(f"Attribute: {prediction['attribute']['label']} "
                  f"(Confidence: {np.max(prediction['attribute']['probabilities']):.2f})")
            print(f"Sentiment: {prediction['sentiment']['label']} "
                  f"(Confidence: {np.max(prediction['sentiment']['probabilities']):.2f})")
        else:
            print("No review text provided. Use --review_text to specify a review to predict.")


if __name__ == "__main__":
    # This block will be executed when the script is run directly
    main()
