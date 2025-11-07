# GitHub Copilot Instructions for stock-plz

## Project Overview

This repository contains two main machine learning projects for financial analysis:

1. **Sentiment Analysis** (`sentiment/` directory): Chinese text sentiment analysis using BERT
2. **Stock Price Prediction** (`stockPrice/` directory): Taiwan stock price prediction using LSTM/Keras

## Project Structure

### Sentiment Analysis (`sentiment/`)
- Uses Google's BERT pre-trained model for Chinese language processing
- Main files:
  - `run_classifier.py`: BERT fine-tuning runner for classification
  - `intent.py`: Single sentence inference using the trained model
  - `train.sh`: Training script
  - `predict.sh`: Testing/prediction script
  - `data/`: Training and testing datasets
- BERT pre-trained model should be placed in `chinese_L-12_H-768_A-12/` directory

### Stock Price Prediction (`stockPrice/`)
- Uses LSTM neural networks with Keras/TensorFlow
- Main files:
  - `prediction.py`: Main prediction script using LSTM
  - `data/`: Stock data from Taiwan Stock Exchange (TWSE) open data
- Default stock: 2330 (Taiwan Semiconductor Manufacturing Company)

## Technology Stack

### Core Dependencies
- **TensorFlow** >= 1.11.0 (CPU or GPU version)
- **Keras**: For building LSTM models
- **Pandas**: Data manipulation and CSV processing
- **NumPy**: Numerical computations
- **scikit-learn**: Feature scaling (MinMaxScaler)
- **matplotlib**: Data visualization

### Python Version
- Compatible with Python 2.x and 3.x (uses `from __future__ import` statements)

## Coding Guidelines

### Language-Specific Considerations
- **Chinese Language Processing**: The sentiment analysis component works with Traditional Chinese text
- Use appropriate tokenization for Chinese characters (BERT tokenizer included)
- Comments may be in Chinese or English

### Data Handling
- Stock data format: CSV files with OHLCV (Open, High, Low, Close, Volume) columns
- File naming convention for stock data: `{stockID}_YYYY_YYYY_ochlv.csv` (lowercase in filenames)
- Time series data uses 20 timesteps by default for LSTM training

### Model Training
- BERT fine-tuning parameters defined via TensorFlow flags
- LSTM configuration:
  - Default: 32 units (first layer) -> 16 units (second layer) -> Dense output
  - Input shape based on timesteps and feature count
  - Optimizer: Adam
  - Loss: Mean Squared Error

### Code Style
- Follow TensorFlow 1.x API patterns (legacy codebase)
- Use flags for configuration parameters (TensorFlow flags pattern)
- Include Apache 2.0 license headers where applicable

## Development Workflow

### For Sentiment Analysis
1. Ensure BERT pre-trained model is available
2. Place training data in `sentiment/data/`
3. Run `train.sh` for training
4. Run `predict.sh` for testing
5. Use `intent.py` for single sentence inference

### For Stock Price Prediction
1. Place stock CSV data in `stockPrice/data/`
2. Adjust `stockID`, `timesteps`, and `epochNum` variables as needed
3. Run `prediction.py` to train and predict

## Important Notes

- **Pre-trained Models**: BERT pre-trained model is not included in the repository
- **Data Sources**: Stock data from TWSE (Taiwan Stock Exchange) open data platform
- **Legacy Code**: Uses TensorFlow 1.x; consider migration path to TF 2.x for new features
- **Data Privacy**: Ensure no sensitive financial data is committed to the repository

## When Assisting with This Repository

1. Respect the existing TensorFlow 1.x patterns unless explicitly migrating
2. Maintain compatibility with Chinese language processing
3. Preserve data pipeline structures for both projects
4. Consider computational efficiency for LSTM training loops
5. Validate data shapes and preprocessing steps carefully
6. Keep training and prediction scripts separate as per existing pattern
