import os
import torch
import argparse
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from model.tokenizer_hyena import CharacterTokenizer
from model.model_allinone import HyenaModel

# CUDA_VISIBLE_DEVICES=7 python Predict_slidewindow.py --fasta /home/gyc/Taraocean/fasta/GCA_001757525.1.fasta --model_path ICCTax.pth --output /home/gyc/Taraocean/result/GCA_001757525.txt --mapping_dir ./mapping --sliding_window

def parse_args():
    parser = argparse.ArgumentParser(description="ICCTax")
    parser.add_argument('--fasta', type=str, required=True, help="Path to the input FASTA file")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained ICCTax model (.pth file)")
    parser.add_argument('--output', type=str, required=True, help="Path to save the output label predictions")
    parser.add_argument('--mapping_dir', type=str, required=True, help="Directory containing Supk, Phyl, and Genus mapping Excel files")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for inference (default: 256)")
    parser.add_argument('--max_length', type=int, default=1500, help="Maximum length for each input chunk (default: 1500)")
    parser.add_argument('--sliding_window', action='store_true', help="Enable sliding window mode with 100bp stride")
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

def split_sequence_sliding(sequence, window_size=1500, stride=100):
    chunks = []
    # 正常滑动窗口
    for i in range(0, len(sequence) - window_size + 1, stride):
        chunk = sequence[i:i + window_size]
        chunks.append(chunk)

    # 如果序列太短不足一个窗口，pad 到 1500bp
    if len(sequence) < window_size:
        chunks.append(sequence.rjust(window_size, 'N'))
    # 如果末尾剩余一段没被覆盖，也加进来（防止尾部信息丢失）
    elif (len(sequence) - window_size) % stride != 0:
        chunks.append(sequence[-window_size:])

    return chunks

def split_sequence(sequence, max_length=1500):
    chunks = [sequence[i:i + max_length] for i in range(0, len(sequence), max_length)]
    if len(chunks[-1]) < max_length:
        chunks[-1] = chunks[-1].ljust(max_length, 'N')
    return chunks

def predict_sequences(sequences, model, tokenizer, device, batch_size, max_length, use_sliding):
    predicted_supk_labels = []
    predicted_phyl_labels = []
    predicted_genus_labels = []
    for sequence in tqdm(sequences, desc="Predicting"):
        if use_sliding:
            segments = split_sequence_sliding(sequence, window_size=max_length, stride=100)
        else:
            segments = split_sequence(sequence, max_length)

        all_supk_probs, all_phyl_probs, all_genus_probs = [], [], []

        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(device)
            with torch.no_grad():
                _, supk_logits, phyl_logits, genus_logits = model(inputs)
            all_supk_probs.append(nn.Softmax(dim=1)(supk_logits).cpu().numpy())
            all_phyl_probs.append(nn.Softmax(dim=1)(phyl_logits).cpu().numpy())
            all_genus_probs.append(nn.Softmax(dim=1)(genus_logits).cpu().numpy())

        avg_supk = np.mean(np.concatenate(all_supk_probs, axis=0), axis=0)
        avg_phyl = np.mean(np.concatenate(all_phyl_probs, axis=0), axis=0)
        avg_genus = np.mean(np.concatenate(all_genus_probs, axis=0), axis=0)

        predicted_supk_labels.append(np.argmax(avg_supk))
        predicted_phyl_labels.append(np.argmax(avg_phyl))
        predicted_genus_labels.append(np.argmax(avg_genus))

    return predicted_supk_labels, predicted_phyl_labels, predicted_genus_labels

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],
        model_max_length=args.max_length + 2,
        add_special_tokens=False,
        padding_side='left'
    )

    model = HyenaModel([4, 44, 156]).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Load mappings
    supk_map = pd.read_excel(os.path.join(args.mapping_dir, "Supk_mapping.xlsx"), header=None)
    phyl_map = pd.read_excel(os.path.join(args.mapping_dir, "Phyl_mapping.xlsx"), header=None)
    genus_map = pd.read_excel(os.path.join(args.mapping_dir, "Genus_mapping.xlsx"), header=None)

    supk_dict = dict(zip(supk_map[1], supk_map[0]))
    phyl_dict = dict(zip(phyl_map[1], phyl_map[0]))
    genus_dict = dict(zip(genus_map[1], genus_map[0]))

    sequences = read_fasta(args.fasta)
    print("Start processing and predicting sequences...")

    supk_labels, phyl_labels, genus_labels = predict_sequences(
        sequences, model, tokenizer, device,
        args.batch_size, args.max_length, args.sliding_window
    )

    with open(args.output, 'w') as f:
        for i in range(len(sequences)):
            f.write(f"Sequence {i+1}: Supk = {supk_dict.get(supk_labels[i], 'Unknown')}, "
                    f"Phyl = {phyl_dict.get(phyl_labels[i], 'Unknown')}, "
                    f"Genus = {genus_dict.get(genus_labels[i], 'Unknown')}\n")

    print(f"✅ Predicted labels have been saved to {args.output}")

if __name__ == "__main__":
    main()
