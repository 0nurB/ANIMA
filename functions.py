def extract_esm2_embeddings_from_fasta(
    fasta_path,
    output_prefix,
    model_path="esm2_t30_150M_UR50D.pt",
    batch_size=32,
    save_every=100):
    import esm
    import torch
    import pandas as pd
    import os
    import warnings
    
    warnings.filterwarnings("ignore", category=UserWarning, module="esm")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # =========================
    # Load model
    # =========================
    model_data = torch.load(model_path, map_location=device, weights_only=False)
    model, alphabet = esm.pretrained.load_model_and_alphabet_core(
        "esm2_t30_150M_UR50D",
        model_data
    )

    model = model.to(device)
    model.eval()

    batch_converter = alphabet.get_batch_converter()

    # =========================
    # Read FASTA
    # =========================
    headers = []
    seqs = []

    with open(fasta_path, "r") as f:
        current_header = None
        current_seq = []

        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header is not None:
                    headers.append(current_header)
                    seqs.append("".join(current_seq))
                current_header = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        if current_header is not None:
            headers.append(current_header)
            seqs.append("".join(current_seq))

    df_fasta = pd.DataFrame({"header": headers, "sequence": seqs})
    print(f"Total number of sequences: {len(df_fasta)}")

    # =========================
    # Extraction
    # =========================
    all_rows = []
    saved_batches = 0
    batch_sequences = []
    batch_headers = []

    for i, row in enumerate(df_fasta.itertuples(index=False)):
        header, sequence = row.header, row.sequence
        batch_headers.append(header)
        batch_sequences.append((header, sequence))

        if len(batch_sequences) == batch_size or i == len(df_fasta) - 1:

            try:
                batch_labels, batch_strs, batch_tokens = batch_converter(batch_sequences)
                batch_tokens = batch_tokens.to(device)

                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[30], return_contacts=False)
                    token_representations = results["representations"][30]

                    for j, tokens_len in enumerate(
                        (batch_tokens != alphabet.padding_idx).sum(1)
                    ):
                        seq_embedding = (
                            token_representations[j, 1:tokens_len - 1]
                            .mean(0)
                            .cpu()
                            .numpy()
                        )
                        all_rows.append([batch_headers[j]] + seq_embedding.tolist())

            except RuntimeError as e:
                print(f"Erro de memoria: {e}")
                torch.cuda.empty_cache()
                continue

            # Clear memory
            del batch_tokens, results, token_representations
            torch.cuda.empty_cache()

            batch_sequences = []
            batch_headers = []

            if (i + 1) % save_every == 0 or i == len(df_fasta) - 1:
                save_path = f"{output_prefix}_part{saved_batches}.csv"
                
                df_out = pd.DataFrame(all_rows)
                df_out.to_csv(save_path, index=False, header=False)

                saved_batches += 1
                all_rows = []

    print("Extraction completed.")


def merge_and_cleanup_embeddings(output_prefix):
    import pandas as pd
    import glob
    import os

    part_files = sorted(glob.glob(f"{output_prefix}_part*.csv"))

    if not part_files:
        print("No files were found to join.")
        return

    dfs = [pd.read_csv(f, header=None) for f in part_files]
    df_final = pd.concat(dfs, ignore_index=True)

    # Caminho final (sem _part)
    final_path = f"{output_prefix}_feats.csv"

    # Salva arquivo final
    df_final.to_csv(final_path, index=False, header=False)
    print(f"Arquivo final salvo em: {final_path}")

    for f in part_files:
        os.remove(f)
        
    print("Process completed.")



def build_X_y_from_embeddings(feat_path, combined_path):
    import pandas as pd

    # =========================
    # Load embeddings
    # =========================
    df_feats = pd.read_csv(feat_path, header=None)
    df_feats = df_feats.rename(columns={0: "protein1"})

    # =========================
    # Load combined pairs
    # =========================
    df_combined = pd.read_csv(combined_path)
    df_combined = df_combined[['protein1', 'protein2', 'Label']]

    # =========================
    # Merge protein1 embeddings
    # =========================
    df_merged = df_combined.merge(df_feats, on="protein1", how="left")

    # =========================
    # Merge protein2 embeddings
    # =========================
    df_feats_p2 = df_feats.rename(columns={"protein1": "protein2"})
    df_merged = df_merged.merge(df_feats_p2, on="protein2", how="left")

    # =========================
    # Build X and y
    # =========================
    feature_cols = df_merged.columns[3:]
    X = df_merged[feature_cols]
    y = df_merged["Label"]

    print("Shape X:", X.shape)
    print("Shape y:", y.shape)

    return X, y, df_combined

def build_X_y_from_embeddings_no_label(feat_path, combined_path):
    import pandas as pd

    # =========================
    # Load embeddings
    # =========================
    df_feats = pd.read_csv(feat_path, header=None)
    df_feats = df_feats.rename(columns={0: "protein1"})

    # =========================
    # Load combined pairs
    # =========================
    df_combined = pd.read_csv(combined_path)
    df_combined = df_combined[['protein1', 'protein2']] #, 'Label']]

    # =========================
    # Merge protein1 embeddings
    # =========================
    df_merged = df_combined.merge(df_feats, on="protein1", how="left")

    # =========================
    # Merge protein2 embeddings
    # =========================
    df_feats_p2 = df_feats.rename(columns={"protein1": "protein2"})
    df_merged = df_merged.merge(df_feats_p2, on="protein2", how="left")

    # =========================
    # Build X and y
    # =========================
    feature_cols = df_merged.columns[2:]
    X = df_merged[feature_cols]
    #y = df_merged["Label"]

    print("Shape X:", X.shape)
    #print("Shape y:", y.shape)

    return X, df_combined

def evaluate_mlp(X, y,
                 scaler_path="data/trained_model/scaler_metazoa_900sc.pkl",
                 model_path="data/trained_model/MLP_model_metazoa.pt",
                 threshold=0.5):

    
    import warnings
    warnings.filterwarnings("ignore")

    import joblib
    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        roc_auc_score,
        average_precision_score
    )

    # =========================
    # Device
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("Using device:", device)

    # =========================
    # Load scaler and scale data
    # =========================
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X).astype(np.float32)

    # =========================
    # Define MLP
    # =========================
    class MLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.25),

                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.25),

                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.25),

                nn.Linear(128, 1)
            )

        def forward(self, x):
            return self.net(x)

    # =========================
    # Load model
    # =========================
    input_dim = X_scaled.shape[1]
    model = MLP(input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # =========================
    # Prediction
    # =========================
    X_tensor = torch.tensor(X_scaled, device=device)

    with torch.no_grad():
        logits = model(X_tensor)
        y_proba = torch.sigmoid(logits).cpu().numpy().ravel()

    y_pred = (y_proba >= threshold).astype(int)

    # =========================
    # Metrics
    # =========================
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    aupr = average_precision_score(y, y_proba)
    auroc = roc_auc_score(y, y_proba)

    print("\n=== Final Metrics ===")
    print(f"Accuracy:    {accuracy_score(y, y_pred):.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-score:    {f1:.4f}")
    print(f"AUPR:        {aupr:.4f}")
    print(f"AUROC:       {auroc:.4f}")

    return y_pred, y_proba

def evaluate_mlp_no_label(X,
                 scaler_path="data/trained_model/scaler_metazoa_900sc.pkl",
                 model_path="data/trained_model/MLP_model_metazoa.pt",
                 threshold=0.5):

    
    import warnings
    warnings.filterwarnings("ignore")

    import joblib
    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        roc_auc_score,
        average_precision_score
    )

    # =========================
    # Device
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("Using device:", device)

    # =========================
    # Load scaler and scale data
    # =========================
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X).astype(np.float32)

    # =========================
    # Define MLP
    # =========================
    class MLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.25),

                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.25),

                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.25),

                nn.Linear(128, 1)
            )

        def forward(self, x):
            return self.net(x)

    # =========================
    # Load model
    # =========================
    input_dim = X_scaled.shape[1]
    model = MLP(input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # =========================
    # Prediction
    # =========================
    X_tensor = torch.tensor(X_scaled, device=device)

    with torch.no_grad():
        logits = model(X_tensor)
        y_proba = torch.sigmoid(logits).cpu().numpy().ravel()

    y_pred = (y_proba >= threshold).astype(int)
    print('Predictions made')
    return y_pred, y_proba


def evaluate_df(df_result, y, y_pred, y_proba, range1, range2):
    """
    df_result: Filtered DataFrame defining the indices to be used
    y, y_pred, y_proba: arrays or Series with true labels, predictions, and probabilities
    range1, range2: strings indicating the source ranges (e.g., "0-40", "40-60")
    plot_hist: if True, plots the final histogram
    """
    import pandas as pd
    from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    f1_score
    )

    filtered_indices = df_result.index

    # Select corresponding samples
    y_filtered = y.loc[filtered_indices] if hasattr(y, 'loc') else y[filtered_indices]
    y_pred_filtered = y_pred.loc[filtered_indices] if hasattr(y_pred, 'loc') else y_pred[filtered_indices]
    y_proba_filtered = y_proba.loc[filtered_indices] if hasattr(y_proba, 'loc') else y_proba[filtered_indices]

    # Convert to DataFrame
    df = pd.DataFrame({
        'y_true': y_filtered,
        'y_pred': y_pred_filtered,
        'y_proba': y_proba_filtered
    })

    # Class balancing
    df_pos = df[df['y_true'] == 1]
    df_neg = df[df['y_true'] == 0]

    min_count = min(len(df_pos), len(df_neg))
    df_balanced = pd.concat([
        df_pos.sample(n=min_count, random_state=42),
        df_neg.sample(n=min_count, random_state=42)
    ]).sample(frac=1, random_state=42)

    acc = accuracy_score(df_balanced['y_true'], df_balanced['y_pred'])

    results = []
    thresholds = [0.5]

    for thresh in thresholds:
        y_pred_thresh = (df_balanced['y_proba'] >= thresh).astype(int)

        prec = precision_score(df_balanced['y_true'], y_pred_thresh, zero_division=0)
        rec = recall_score(df_balanced['y_true'], y_pred_thresh, zero_division=0)
        f1 = f1_score(df_balanced['y_true'], y_pred_thresh, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(df_balanced['y_true'], y_pred_thresh).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        auroc = roc_auc_score(df_balanced['y_true'], df_balanced['y_proba'])
        aupr = average_precision_score(df_balanced['y_true'], df_balanced['y_proba'])

        results.append({
            'Range1': range1,
            'Range2': range2,
            'Threshold': thresh,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'Specificity': specificity,
            'Standard_Accuracy': acc,
            'AUPR': aupr,
            'AUROC': auroc
        })

    df_final_results = pd.DataFrame(results)
    return df_final_results


def filter_interactions_by_two_scores(df_first, df_combined, max_range, min_range):
    """
    Filters df_combined to keep only interactions where:
    - Both proteins are present in df_first.
    - The HIGHER score between the two proteins falls within max_range.
    - The LOWER score between the two proteins falls within min_range.

    Parameters:
    - df_first: DataFrame with columns ['qseqid', 'score']
    - df_combined: DataFrame with columns ['protein1', 'protein2', 'Label']
    - max_range: str in the format "start-end", e.g., "40-60"
    - min_range: str in the format "start-end", e.g., "80-100"

    Returns:
    - Filtered DataFrame.
    """
    import pandas as pd
    try:
        max_start, max_end = map(float, max_range.split("-"))
        min_start, min_end = map(float, min_range.split("-"))
    except:
        raise ValueError(
            "Ranges must be in the format 'start-end', for example '40-60'."
        )

    # Set of valid proteins
    valid_proteins = set(df_first["qseqid"])

    # Keep only interactions where both proteins are present
    df_filtered = df_combined[
        (df_combined["protein1"].isin(valid_proteins)) &
        (df_combined["protein2"].isin(valid_proteins))
    ].copy()

    # Dictionary mapping protein to score
    score_dict = df_first.set_index("qseqid")["score"].to_dict()

    # Map scores
    df_filtered["score1"] = df_filtered["protein1"].map(score_dict)
    df_filtered["score2"] = df_filtered["protein2"].map(score_dict)

    # Compute max and min scores
    df_filtered["score_max"] = df_filtered[["score1", "score2"]].max(axis=1)
    df_filtered["score_min"] = df_filtered[["score1", "score2"]].min(axis=1)

    # Apply both filters
    df_result = df_filtered[
        (df_filtered["score_max"] > max_start) & (df_filtered["score_max"] <= max_end) &
        (df_filtered["score_min"] > min_start) & (df_filtered["score_min"] <= min_end)
    ]

    return df_result




def evaluate_by_score_ranges(
    ranges,
    alignment_df,
    original_ppi_df,
    y,
    y_pred,
    y_proba
):
    """
    Evaluates model performance across multiple score ranges.

    Parameters
    ----------
    ranges : list of lists
        List of score range pairs, e.g. [["80-100","60-80"], ...]
    alignment_df : DataFrame
        DataFrame containing protein scores (qseqid, score)
    original_ppi_df : DataFrame
        PPI DataFrame with columns ['protein1', 'protein2', 'Label']
    y, y_pred, y_proba : array-like or Series
        True labels, predicted labels, and predicted probabilities

    Returns
    -------
    df_all_results : DataFrame
        Evaluation metrics for all valid score ranges
    df_all_predictions : DataFrame
        Filtered interactions with predicted labels
    """
    import pandas as pd
    all_results = []
    all_predictions = []

    for score_range in ranges:
        range1, range2 = score_range

        df_filtered = filter_interactions_by_two_scores(
            alignment_df,
            original_ppi_df,
            range1,
            range2
        )

        if not df_filtered.empty:
            # Evaluate metrics
            df_summary = evaluate_df(
                df_filtered,
                y,
                y_pred,
                y_proba,
                range1,
                range2
            )
            all_results.append(df_summary)

            # Store predictions
            df_pred = df_filtered.copy()
            df_pred['Predicted_Label'] = y_pred.loc[df_filtered.index] \
                if hasattr(y_pred, 'loc') else y_pred[df_filtered.index]

            all_predictions.append(df_pred)

    df_all_results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    df_all_predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    return df_all_results, df_all_predictions


def evaluate_by_score_ranges_no_label(
    ranges,
    alignment_df,
    original_ppi_df,
    y_pred
):
    """
    Evaluates model performance across multiple score ranges.

    Parameters
    ----------
    ranges : list of lists
        List of score range pairs, e.g. [["80-100","60-80"], ...]
    alignment_df : DataFrame
        DataFrame containing protein scores (qseqid, score)
    original_ppi_df : DataFrame
        PPI DataFrame with columns ['protein1', 'protein2', 'Label']
    y, y_pred, y_proba : array-like or Series
        True labels, predicted labels, and predicted probabilities

    Returns
    -------
    df_all_results : DataFrame
        Evaluation metrics for all valid score ranges
    df_all_predictions : DataFrame
        Filtered interactions with predicted labels
    """
    import pandas as pd
    all_results = []
    all_predictions = []

    for score_range in ranges:
        range1, range2 = score_range

        df_filtered = filter_interactions_by_two_scores(
            alignment_df,
            original_ppi_df,
            range1,
            range2
        )

        if not df_filtered.empty:
            df_pred = df_filtered.copy()
            df_pred['Predicted_Label'] = y_pred.loc[df_filtered.index] \
                if hasattr(y_pred, 'loc') else y_pred[df_filtered.index]

            all_predictions.append(df_pred)

    df_all_predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    return df_all_predictions
