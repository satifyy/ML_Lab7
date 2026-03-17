# ML Lab 7 - News Category Classification

This is our ML Lab 7 project where we classify news articles into 5 categories using deep learning models.

Categories:

- Business
- Education
- Entertainment
- Sports
- Technology

Everything is done inside `ML_Lab7.ipynb`.

## Project Files

- `ML_Lab7.ipynb` - main notebook with preprocessing, modeling, training, and plots
- `business_data.csv`
- `education_data.csv`
- `entertainment_data.csv`
- `sports_data.csv`
- `technology_data.csv`

## Dataset Overview

The dataset is made by combining all 5 CSV files into one DataFrame.

- Total samples: about **10,000**
- Roughly **2,000 per class**

Main columns used:

- `content` (input text)
- `category` (target label)

## What We Did in the Notebook

1. **Data prep + preprocessing**
   - Loaded and merged all CSV files
   - Removed rows with missing `content` or `category`
   - Encoded labels using `LabelEncoder`
   - Tokenized text using Keras `Tokenizer`
   - Padded/truncated sequences to fixed length

2. **Built and compared models**
   - CNN model #1 (kernel size 5)
   - CNN model #2 (kernel size 3)
   - Transformer baseline (4 attention heads)
   - Transformer variant (8 attention heads)

3. **Training and visualization**
   - Used stratified 80/20 train-test split
   - Plotted training/validation accuracy and loss

## Requirements

Recommended Python version: **3.10+**

Install packages:

```bash
pip install pandas numpy scikit-learn matplotlib tensorflow
```

## How to Run

1. Open `ML_Lab7.ipynb` in VS Code (or Jupyter).
2. Make sure all 5 CSV files are in this same folder.
3. Check the `folder_path` variable in the notebook (set it to the correct path if needed).
4. Run cells from top to bottom.

## Notes

- On Apple Silicon Macs, TensorFlow setup can vary depending on Python version/environment.
