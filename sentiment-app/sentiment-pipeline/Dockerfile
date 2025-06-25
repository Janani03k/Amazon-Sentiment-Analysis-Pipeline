FROM bitnami/spark:3.4.1

# Switch to root to install Python packages
USER root

# Install Python + ML + Hugging Face dependencies
RUN pip install --no-cache-dir \
    boto3 \
    pandas \
    scikit-learn \
    joblib \
    python-dotenv \
    pyspark \
    transformers \
    mlflow[extras] \
    torch \
    accelerate

# Set working directory
WORKDIR /app

# Copy pipeline scripts
COPY run_clean.py .
COPY run_infer.py .
COPY run_all.py .

# Default: run clean + inference together
ENTRYPOINT ["python", "run_all.py"]
