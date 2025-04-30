FROM python:3.9

WORKDIR ./

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py .
COPY starbucks_reviews.csv .

# Create directories for model artifacts
RUN mkdir -p models results

# Download BERT model and tokenizer in advance to avoid downloading at runtime
RUN python -c "from transformers import BertModel, BertTokenizer; BertModel.from_pretrained('bert-base-uncased'); BertTokenizer.from_pretrained('bert-base-uncased')"

# Set up environment variables
ENV PYTHONUNBUFFERED=1

# Expose port for Streamlit
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
