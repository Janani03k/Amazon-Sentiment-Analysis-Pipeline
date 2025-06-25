from sagemaker.huggingface import HuggingFace, TrainingCompilerConfig
import sagemaker

# Get the SageMaker execution role
role = sagemaker.get_execution_role()

# HuggingFace Estimator with Training Compiler
huggingface_estimator = HuggingFace(
    entry_point="train.py",
    source_dir=".",
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    role=role,
    transformers_version="4.17.0",
    pytorch_version="1.10.2",
    py_version="py38",
    hyperparameters={
        "epochs": 2,
        "train_batch_size": 16
    },
    compiler_config=TrainingCompilerConfig(enabled=True),
    base_job_name="distilbert-sentiment-optimized",
    output_path="s3://mlops-sentiment-analysis-data/models/",
)

# Launch training job
huggingface_estimator.fit({
    "train": "s3://mlops-sentiment-analysis-data/Silver"
})
