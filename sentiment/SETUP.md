# Setup Instructions for Sentiment Analysis

This guide helps you set up the sentiment analysis system with TensorFlow 2.16.1 and GPU support.

## Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (for GPU acceleration)
- CUDA 12.x and cuDNN 8.9+ (for TensorFlow 2.16.1)

## Installation Steps

### 1. Install Dependencies

```bash
cd sentiment
pip install -r requirements.txt
```

If you have a NVIDIA GPU, ensure you have the appropriate CUDA and cuDNN versions installed. TensorFlow 2.16.1 supports CUDA 12.x.

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

TensorFlow 2.16.1 will automatically use your GPU if it's properly configured. To verify GPU availability:

```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

## Troubleshooting

### CUDA/cuDNN Issues

If you encounter GPU-related errors:
1. Verify CUDA installation: `nvcc --version`
2. Ensure cuDNN is installed correctly
3. Check TensorFlow GPU support: `python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

### Memory Issues

If you run out of GPU memory:
- Reduce `train_batch_size` in train.sh (e.g., from 16 to 8 or 4)
- Reduce `max_seq_length` (e.g., from 300 to 128)

### Download Issues

If the BERT model download fails:
- Try using a VPN or mirror site
- Download manually from Google's BERT repository: https://github.com/google-research/bert

## Changes from TensorFlow 1.x

This codebase has been updated from TensorFlow 1.x to 2.16.1 with the following changes:

- Replaced `tf.flags` with `absl.flags`
- Replaced `tf.gfile` with `tf.io.gfile`
- Replaced `tf.layers` with `tf.keras.layers`
- Replaced `tf.contrib` APIs with TF 2.x compatible alternatives
- Removed TPU-specific code (TPUEstimator)
- Updated to use standard `tf.estimator.Estimator` for GPU/CPU
- Updated all deprecated TF 1.x APIs to TF 2.x equivalents

## Additional Resources

- TensorFlow 2.x Migration Guide: https://www.tensorflow.org/guide/migrate
- BERT GitHub: https://github.com/google-research/bert
- Chinese BERT Models: https://github.com/google-research/bert#pre-trained-models
