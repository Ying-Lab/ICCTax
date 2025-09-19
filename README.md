# ICCTax
## Overview
ICCTax is a taxonomy classification tool powered by the HyenaDNA foundation model, which can effectively classify the four superkingdoms—Archaea, Bacteria, Eukaryotes, and Viruses——covering 43 phyla and 155 genera.The HyenaDNA foundation model is a large language model designed for long-range genomic sequences, offering single nucleotide resolution for accurate taxonomic classification. ICCTax can perform analysis of community composition in metagenomes.

## 🔧Installation
ICCTax is a Python package. To install it, run the following command in your terminal:
```
git clone https://github.com/Ying-Lab/ICCTax.git && cd ICCTax
```

Create a conda environment with Python 3.8+ and configure it using requirements.txt after activating the environment.
```
conda create -n ICCTax python=3.8
conda activate ICCTax
pip install -r requirements.txt
```
<!-- 加载预训练模型的步骤 -->

## 📦 How to Load the Pretrained HyenaDNA Model
To ensure the ICCTax model can correctly load the pretrained HyenaDNA backbone, please follow these steps:
1. **Navigate to the file** `ICCTax/model/model_allinone.py`.
2. **Locate the following line** (around the model loading logic), Replace the placeholder path '/your/path/ICCTax/hyena/' with the absolute path to your local hyena/ folder. For example:

   ```python
   self.model = HyenaDNAPreTrainedModel.from_pretrained(
       '/your/path/ICCTax/hyena/',
       pretrained_model_name,
   )
   ```
3. Save the file.<br>

💡 The hyena/ folder must contain the pretrained model configuration and weights (e.g., config.json, pytorch_model.bin, etc.).<br>
This step ensures the HyenaDNA model is loaded properly and prevents file-not-found errors during ICCTax inference.

<!-- 测试部分 -->
## Run ICCTax
### 🚀 ICCTax Mode1: Fixed-Length Inference (First 1,500 bp Only)
This script performs hierarchical taxonomic classification using the ICCTax model.  
**Only the first 1,500 base pairs of each input sequence are used** for inference.  
Full sequences ≤1,500 bp are processed directly.
### 🔧 Command-Line Arguments
``` 
python Predict_only1500bp.py \
  --fasta FASTA_PATH \
  --model_path MODEL_PATH \
  --output OUTPUT_PATH \
  --mapping_dir MAPPING_DIRECTORY \
  [--batch_size BATCH_SIZE]
```
| Argument        | Description                                                                                                                 |
| --------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `--fasta`       | Path to the input FASTA file (**required**)                                                                                 |
| `--model_path`  | Path to the trained ICCTax model checkpoint (.pth) (**required**)                                                           |
| `--output`      | Output file path for saving predicted labels (**required**)                                                                 |
| `--mapping_dir` | Directory containing mapping Excel files: `Supk_mapping.xlsx`, `Phyl_mapping.xlsx`, and `Genus_mapping.xlsx` (**required**) |
| `--batch_size`  | Number of sequences processed per batch during inference (**default: 256**)                                                 |

### Example
```
python Predict_only1500bp.py --fasta test.fasta --model_path ICCTax.pth --output ICCTax_only1500bp.txt --mapping_dir ./mapping
```
### 🚀 ICCTax Mode2: Sliding-Window Inference (Sliding over Long Sequences)
### Optional: enable overlap with 100bp (default) stride
``` 
python Predict_chunk.py \
  --fasta FASTA_PATH \
  --model_path MODEL_PATH \
  --output OUTPUT_PATH \
  --mapping_dir MAPPING_DIRECTORY \
  --sliding_window
  [--batch_size BATCH_SIZE]
  [--max_length MAX_LENGTH]
```
| Argument        | Description                                                                                                                 |
| --------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `--fasta`       | Path to the input FASTA file (**required**)                                                                                 |
| `--model_path`  | Path to the trained ICCTax model checkpoint (.pth) (**required**)                                                           |
| `--output`      | Output file path for saving predicted labels (**required**)                                                                 |
| `--mapping_dir` | Directory containing mapping Excel files: `Supk_mapping.xlsx`, `Phyl_mapping.xlsx`, and `Genus_mapping.xlsx` (**required**) |
| `--batch_size`  | Number of sequences processed per batch during inference (**default: 256**)                                                 |
| `--max_length`  | Maximum sequence chunk length for inference, sequences longer than this will be split (**default: 1500**)                   |
| `--sliding_window`| Maximum sequence chunk length for inference, sequences longer than this will be split (**default: 1500**)                   |

### Example
```
python Predict_slidewindow.py --fasta test.fasta --model_path ICCTax.pth --output ICCTax_window.txt --mapping_dir ./mapping --sliding_window
```

### 🚀 ICCTax Mode3: Chunk-Based Inference (Sliding over Long Sequences)
Sequences longer than 1500 nt are split into equal chunks, one prediction (average) per sequence.
``` 
python Predict_chunk.py \
  --fasta FASTA_PATH \
  --model_path MODEL_PATH \
  --output OUTPUT_PATH \
  --mapping_dir MAPPING_DIRECTORY \
  [--batch_size BATCH_SIZE]
  [--max_length MAX_LENGTH]
```
| Argument        | Description                                                                                                                 |
| --------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `--fasta`       | Path to the input FASTA file (**required**)                                                                                 |
| `--model_path`  | Path to the trained ICCTax model checkpoint (.pth) (**required**)                                                           |
| `--output`      | Output file path for saving predicted labels (**required**)                                                                 |
| `--mapping_dir` | Directory containing mapping Excel files: `Supk_mapping.xlsx`, `Phyl_mapping.xlsx`, and `Genus_mapping.xlsx` (**required**) |
| `--batch_size`  | Number of sequences processed per batch during inference (**default: 256**)                                                 |
| `--max_length`  | Maximum sequence chunk length for inference, sequences longer than this will be split (**default: 1500**)                   |

### Example
```
python Predict_chunk.py --fasta test.fasta --model_path ICCTax.pth --output ICCTax_chunk.txt --mapping_dir ./mapping
```

<!-- 训练模型 -->
<!-- 具体操作说明 -->
