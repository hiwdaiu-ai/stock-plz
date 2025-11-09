## Sentiment Analysis with BERT

使用 BERT Pretrain Model 進行訓練與分類

### Files
- `train.sh` : Training script
- `predict.sh` : Testing script
- `intent.py` : Using model for single sentence inference 
- `chinese_L-12_H-768_A-12/` : BERT pretrained model from Google (must be downloaded separately)
- `data/` : Training and testing data

### Important: TensorFlow Version

⚠️ **This code requires TensorFlow 2.15.x** because it uses the Estimator API which was removed in TensorFlow 2.16+.

Please see [SETUP.md](SETUP.md) for detailed installation instructions and information about:
- Downloading the BERT pretrained model
- Setting up TensorFlow 2.15.1
- GPU configuration
- Migrating to Keras (for future-proofing)

### Quick Start

1. Install dependencies:
```bash
pip install tensorflow==2.15.1 pandas absl-py
```

2. Download BERT model (see SETUP.md for details)

3. Run training:
```bash
bash train.sh
```

For more details, see [SETUP.md](SETUP.md).
