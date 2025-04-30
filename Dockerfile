FROM python:3.9

WORKDIR ./

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download BERT model and tokenizer to avoid downloading at runtime
RUN python -c "from transformers import BertModel, BertTokenizer; \
    BertModel.from_pretrained('bert-base-uncased'); \
    BertTokenizer.from_pretrained('bert-base-uncased')"

# Copy application code
COPY *.py .
COPY starbucks_reviews.csv .

# Create directories for outputs
RUN mkdir -p models results

# Default command to run training
# Can be overridden with docker run commands
ENTRYPOINT ["python", "main.py"]
CMD ["--mode", "train", "--data_path", "starbucks_reviews.csv", "--epochs", "3"]
