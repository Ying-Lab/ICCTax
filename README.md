# ICCTax
## Overview
ICCTax is a taxonomy classification tool powered by the HyenaDNA foundation model, which can effectively classify the four superkingdoms—Archaea, Bacteria, Eukaryotes, and Viruses—and 43 phyla within these superkingdoms.The HyenaDNA foundation model is a large language model designed for long-range genomic sequences, offering single nucleotide resolution for accurate taxonomic classification. ICCTax can perform analysis of community composition in metagenomes.

## Installation
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

Load the hyenadna pre-training module in  **model_allinone.py**.
```
cd ICCTax/model
/your/path/ICCTax/hyena/
```

## Run ICCTax model
``` 
predict.py [-h] 
           --input_fasta INPUT_FASTA 
           --output_labels_file OUTPUT_LABELS_FILE 
           --model_path MODEL_PATH 
           --supk_mapping_file SUPK_MAPPING_FILE 
           --phyl_mapping_file PHYL_MAPPING_FILE 
           [--batch_size BATCH_SIZE] 
           [--max_length MAX_LENGTH]
```
--input_fasta          &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;       Path to the input FASTA file  
--output_labels_file   &nbsp;&nbsp;&nbsp;    Path to save the output labels  
--model_path           &nbsp;&nbsp;&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;      Path to the ICCTax model file  
--supk_mapping_file     &nbsp;&nbsp;  Path to the Supk mapping file (Excel)  
--phyl_mapping_file    &nbsp;&nbsp;&nbsp;    Path to the Phyl mapping file (Excel)  
--batch_size           &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;     Batch size for predictions (**default=16**)  
--max_length           &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp;      Maximum length for sequence splitting (**default=1500**) 

## Example
```
python predict.py --input_fasta /ICCTax/test.fasta --output_labels_file /ICCTax/output.txt --model_path /ICCTax/ICCTax.pth --supk_mapping_file ICCTax/mapping/Supk_mapping.xlsx --phyl_mapping_file /ICCTax/mapping/Phyl_mapping.xlsx --batch_size 16 --max_length 1500
```
