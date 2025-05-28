import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from model.tokenizer_hyena import CharacterTokenizer
from model.model_allinone import HyenaModel
# only the first 1,500 bp is used for sequences longer than 1,500 bp; full sequences are used if ≤1,500 bp.
# Sample
# python Predict_only1500bp.py --fasta test_head20.fasta --model_path ICCTax.pth --output ICCTax_only1500bp.txt --mapping_dir ./mapping


def parse_args():
    parser = argparse.ArgumentParser(description="ICCTax Inference Script")
    parser.add_argument("--fasta", type=str, required=True, help="Path to input fasta file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--output", type=str, required=True, help="Path to output file for predicted labels")
    parser.add_argument("--mapping_dir", type=str, required=True, help="Directory containing mapping Excel files")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference")
    return parser.parse_args()

def read_fasta(fasta_file, max_length=1500):
    sequences = []
    with open(fasta_file, 'r') as f:
        sequence = ""
        for line in f:
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence[:max_length] if len(sequence) > max_length else sequence)
                sequence = ""
            else:
                sequence += line.strip()
        if sequence:
            sequences.append(sequence[:max_length] if len(sequence) > max_length else sequence)
    return sequences


def predict_sequences(sequences, model, tokenizer, device, batch_size):
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    all_supk_probs, all_phyl_probs, all_genus_probs = [], [], []

    for i in tqdm(range(num_batches), desc="Predicting batches"):
        batch_segments = sequences[i * batch_size: (i + 1) * batch_size]
        segment_tokens = tokenizer(batch_segments, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)

        with torch.no_grad():
            _, supk_logits, phyl_logits, genus_logits = model(segment_tokens)

        all_supk_probs.append(nn.Softmax(dim=1)(supk_logits).cpu().numpy())
        all_phyl_probs.append(nn.Softmax(dim=1)(phyl_logits).cpu().numpy())
        all_genus_probs.append(nn.Softmax(dim=1)(genus_logits).cpu().numpy())

    supk_labels = np.argmax(np.concatenate(all_supk_probs, axis=0), axis=1)
    phyl_labels = np.argmax(np.concatenate(all_phyl_probs, axis=0), axis=1)
    genus_labels = np.argmax(np.concatenate(all_genus_probs, axis=0), axis=1)

    return supk_labels, phyl_labels, genus_labels


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_length = 1500

    print("Loading tokenizer and model...")
    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],
        model_max_length=max_length + 2,
        add_special_tokens=False,
        padding_side='left',
    )

    model = HyenaModel([4, 44, 156]).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    print("Reading sequences from fasta file...")
    sequences = read_fasta(args.fasta)

    print("Performing sequence classification...")
    supk_labels, phyl_labels, genus_labels = predict_sequences(sequences, model, tokenizer, device, args.batch_size)

    print("Loading mapping files...")
    supk_map = pd.read_excel(f"{args.mapping_dir}/Supk_mapping.xlsx", header=None, names=["name", "id"])
    phyl_map = pd.read_excel(f"{args.mapping_dir}/Phyl_mapping.xlsx", header=None, names=["name", "id"])
    genus_map = pd.read_excel(f"{args.mapping_dir}/Genus_mapping.xlsx", header=None, names=["name", "id"])

    supk_dict = dict(zip(supk_map["id"], supk_map["name"]))
    phyl_dict = dict(zip(phyl_map["id"], phyl_map["name"]))
    genus_dict = dict(zip(genus_map["id"], genus_map["name"]))

    print("Saving mapped predictions to output file...")
    with open(args.output, 'w') as f:
        for i in range(len(sequences)):
            supk_name = supk_dict.get(supk_labels[i], "Unknown")
            phyl_name = phyl_dict.get(phyl_labels[i], "Unknown")
            genus_name = genus_dict.get(genus_labels[i], "Unknown")
            f.write(f"Sequence {i + 1}: Supk = {supk_name}, Phyl = {phyl_name}, Genus = {genus_name}\n")

    print(f"✅ Predicted labels have been saved to {args.output}")


if __name__ == "__main__":
    main()
