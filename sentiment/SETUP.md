# Setup Instructions for Sentiment Analysis

This guide helps you set up the sentiment analysis system with TensorFlow 2.x and GPU support.

## Important Note about TensorFlow Version

⚠️ **TensorFlow 2.16+ removed the Estimator API** which this BERT implementation relies on. We recommend one of the following approaches:

### Option 1: Use TensorFlow 2.15 (Recommended for Quick Setup)
TensorFlow 2.15 is the last version that includes the Estimator API. This is the easiest path to get the code running without major changes.

### Option 2: Migrate to Keras (Recommended for Long-term)
For long-term projects, users are encouraged to migrate from the Estimator API to `tf.keras`. This would involve:
- Converting the BERT model to use Keras layers and models
- Replacing the Estimator training loop with `model.fit()`
- Updating checkpoints to Keras format

**This repository currently uses the Estimator API for compatibility with the original BERT implementation.**

## Prerequisites

- Python 3.8 - 3.11 (for TensorFlow 2.15 compatibility)
- NVIDIA GPU with CUDA support (for GPU acceleration)
- CUDA 11.8+ and cuDNN 8.6+ (for TensorFlow 2.15)

## Installation Steps

### 1. Install Dependencies

For TensorFlow 2.15 (recommended):
```bash
cd sentiment
pip install tensorflow==2.15.1
pip install pandas absl-py
```

For CPU-only installations:
```bash
pip install tensorflow==2.15.1
pip install pandas absl-py
```

If you have a NVIDIA GPU, ensure you have the appropriate CUDA and cuDNN versions installed.

### 2. Download BERT Pretrained Model

The Chinese BERT model is required but not included in this repository due to its size. Download it from Google:

#### Option 1: Direct Download

1. Download the Chinese BERT-Base model from: https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip

2. Extract the zip file:
```bash
cd sentiment
wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
unzip chinese_L-12_H-768_A-12.zip
```

3. Verify the extracted directory structure:
```
sentiment/
  └── chinese_L-12_H-768_A-12/
      ├── bert_config.json
      ├── bert_model.ckpt.data-00000-of-00001
      ├── bert_model.ckpt.index
      ├── bert_model.ckpt.meta
      └── vocab.txt
```

#### Option 2: Using curl

```bash
cd sentiment
curl -O https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
unzip chinese_L-12_H-768_A-12.zip
```

### 3. Verify Setup

Check that all required files are in place:

```bash
ls sentiment/chinese_L-12_H-768_A-12/
# Should show: bert_config.json, bert_model.ckpt.*, vocab.txt
```

### 4. Train the Model

Run the training script:

```bash
cd sentiment
bash train.sh
```

Or run with custom parameters:

```bash
python3 run_classifier.py \
  --data_dir=data \
  --task_name=sim \
  --vocab_file=chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \
  --output_dir=tmp/sim_model \
  --do_train=true \
  --do_eval=true \
  --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=300 \
  --train_batch_size=16 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0
```

## GPU Configuration

TensorFlow 2.15 will automatically use your GPU if it's properly configured. To verify GPU availability:

```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

## Troubleshooting

### TensorFlow Version Issues

If you encounter `AttributeError: module 'tensorflow' has no attribute 'estimator'`:
- You are likely using TensorFlow 2.16+, which removed the Estimator API
- Downgrade to TensorFlow 2.15.1: `pip install tensorflow==2.15.1`
- Or consider migrating the code to use `tf.keras` (requires significant code changes)

### Python Version Issues

TensorFlow 2.15 requires Python 3.8-3.11. If you're using Python 3.12+:
- Use a Python 3.11 virtual environment
- Or use Docker with Python 3.11

Example with pyenv:
```bash
pyenv install 3.11.5
pyenv virtualenv 3.11.5 bert-env
pyenv activate bert-env
pip install tensorflow==2.15.1 pandas absl-py
```

### CUDA/cuDNN Issues

If you encounter GPU-related errors:
1. Verify CUDA installation: `nvcc --version`
2. Ensure cuDNN is installed correctly
3. Check TensorFlow GPU support: `python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

For TensorFlow 2.15, you need:
- CUDA 11.8 or CUDA 12.2
- cuDNN 8.6 or later

### Memory Issues

If you run out of GPU memory:
- Reduce `train_batch_size` in train.sh (e.g., from 16 to 8 or 4)
- Reduce `max_seq_length` (e.g., from 300 to 128)

### Download Issues

If the BERT model download fails:
- Try using a VPN or mirror site
- Download manually from Google's BERT repository: https://github.com/google-research/bert

## Changes from TensorFlow 1.x to 2.x

This codebase has been updated from TensorFlow 1.x to 2.x compatible mode with the following changes:

- Updated to use `tensorflow.compat.v1` API for compatibility
- Replaced `tf.flags` with `absl.flags`
- Replaced `tf.gfile` with `tf.io.gfile`  
- Replaced `tf.layers.dense` with `tf.keras.layers.Dense`
- Replaced `tf.contrib.layers.layer_norm` with `tf.keras.layers.LayerNormalization`
- Removed TPU-specific code (TPUEstimator, CrossShardOptimizer)
- Updated to use `tf.compat.v1.estimator` API for GPU/CPU training
- Enabled `tf.disable_v2_behavior()` for TF1-style execution

**Note:** The code uses TF1 compatibility mode (`tf.compat.v1`) with TensorFlow 2.15. This allows the original BERT Estimator-based code to run on TensorFlow 2.x without a complete rewrite.

## Future Migration to Keras

For users interested in migrating to pure TensorFlow 2.x with Keras:

1. **Model Definition**: Convert `BertModel` class to use `tf.keras.Model`
2. **Training Loop**: Replace Estimator with `model.compile()` and `model.fit()`
3. **Checkpoints**: Convert TF1 checkpoints to Keras SavedModel format
4. **Data Pipeline**: Update `tf.data` pipeline to work with Keras
5. **Metrics**: Replace `tf.metrics` with `tf.keras.metrics`

Reference implementations:
- Hugging Face Transformers: https://github.com/huggingface/transformers
- TensorFlow Official Models: https://github.com/tensorflow/models/tree/master/official

## Additional Resources

- TensorFlow 2.x Migration Guide: https://www.tensorflow.org/guide/migrate
- Estimator to Keras Migration: https://www.tensorflow.org/guide/migrate/migrating_estimator
- BERT GitHub: https://github.com/google-research/bert
- Chinese BERT Models: https://github.com/google-research/bert#pre-trained-models
- TensorFlow 2.15 Release Notes: https://github.com/tensorflow/tensorflow/releases/tag/v2.15.0
