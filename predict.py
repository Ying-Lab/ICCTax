import torch
import argparse
from model.tokenizer_hyena import CharacterTokenizer
from model.model_allinone import HyenaModel
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Predict Supk and Phyl labels for sequences in a FASTA file")
    parser.add_argument('--input_fasta', type=str, required=True, help="Path to the input FASTA file")
    parser.add_argument('--output_labels_file', type=str, required=True, help="Path to save the output labels")
    parser.add_argument('--model_path', type=str, required=True,default='/your/path/ICCTax/ICCTax.pth', help="Path to the ICCTax model file")
    parser.add_argument('--supk_mapping_file', type=str, required=True,default='/your/path/ICCTax/mapping/Supk_mapping.xlsx', help="Path to the Supk mapping file (Excel)")
    parser.add_argument('--phyl_mapping_file', type=str, required=True, default='/your/path/ICCTax/mapping/Phyl_mapping.xlsx',help="Path to the Phyl mapping file (Excel)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for predictions")
    parser.add_argument('--max_length', type=int, default=1500, help="Maximum length for sequence splitting")
    
    return parser.parse_args()

def read_fasta(fasta_file):
    sequences = []
    
    with open(fasta_file, 'r') as f:
        sequence = ""
        for line in f:
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                sequence = "" 
            else:
                sequence += line.strip() 
        if sequence: 
            sequences.append(sequence)
    
    return sequences

def split_sequence(sequence, max_length=1500):
    chunks = [sequence[i:i + max_length] for i in range(0, len(sequence), max_length)]
    if len(chunks[-1]) < max_length:
        chunks[-1] = chunks[-1].ljust(max_length, 'N') 
    return chunks

def predict_sequences(sequences, model, tokenizer, device, batch_size):
    all_segments = []
    predicted_supk_labels = []
    predicted_phyl_labels = []
    for sequence in tqdm(sequences):
        all_segments = split_sequence(sequence)
        all_supk_probs, all_phyl_probs = [], []
        
        for i in range(0, len(all_segments), batch_size):
            batch_segments = all_segments[i:i + batch_size]
            
            batch_tokenized = tokenizer(batch_segments, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(device)
            
            with torch.no_grad():
                _, supk_logits, phyl_logits, _ = model(batch_tokenized)
            
            supk_prob = nn.Softmax(dim=1)(supk_logits).cpu().numpy()
            phyl_prob = nn.Softmax(dim=1)(phyl_logits).cpu().numpy()

            all_supk_probs.append(supk_prob)
            all_phyl_probs.append(phyl_prob)
        
        all_supk_probs = np.concatenate(all_supk_probs, axis=0)
        all_phyl_probs = np.concatenate(all_phyl_probs, axis=0)
        
        average_supk_probs = np.mean(all_supk_probs, axis=0)
        average_phyl_probs = np.mean(all_phyl_probs, axis=0)
        
        supk_labels = np.argmax(average_supk_probs, axis=0)
        phyl_labels = np.argmax(average_phyl_probs, axis=0)
        
        predicted_supk_labels.append(supk_labels)
        predicted_phyl_labels.append(phyl_labels)
        
    return predicted_supk_labels, predicted_phyl_labels

def main():
    # Parse command-line arguments
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize tokenizer and model
    dnatokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],
        model_max_length=args.max_length + 2,
        add_special_tokens=False, 
        padding_side='left',
    )

    # Load Supk and Phyl mappings
    supk_mapping_df = pd.read_excel(args.supk_mapping_file, header=None)
    phyl_mapping_df = pd.read_excel(args.phyl_mapping_file, header=None)

    supk_mapping = dict(zip(supk_mapping_df[1], supk_mapping_df[0]))
    phyl_mapping = dict(zip(phyl_mapping_df[1], phyl_mapping_df[0]))

    # Load model
    model = HyenaModel([4, 44, 156]).to(device)
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model.eval()

    # Read input sequences
    sequences = read_fasta(args.input_fasta)
    # Perform predictions
    print("Start processing and predicting sequences...")
    predicted_supk_labels, predicted_phyl_labels = predict_sequences(sequences, model, dnatokenizer, device, args.batch_size)

    # Map the predicted labels to their corresponding values
    mapped_supk_labels = [supk_mapping[label] for label in predicted_supk_labels]
    mapped_phyl_labels = [phyl_mapping[label] for label in predicted_phyl_labels]

    # Write the results to the output file
    with open(args.output_labels_file, 'w') as f:
        for i in range(len(sequences)):
            f.write(f"Seq {i + 1}: Supk Label = {mapped_supk_labels[i]}, Phyl Label = {mapped_phyl_labels[i]}\n")

    print(f"The predicted labels have been saved to {args.output_labels_file}")

if __name__ == "__main__":
    main()
