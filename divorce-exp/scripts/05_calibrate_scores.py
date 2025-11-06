"""
Calibrate LLM Scores as Predictors (kx + b) for Divorce Prediction

Methods:
1) participants: feature = (10 - husband_willingness) * (10 - wife_willingness)
2) observer: feature = observer_divorce_score (0-10)
3) mixed: features = [observer, participants]
4) linear: features = [10 - wife_willingness, 10 - husband_willingness, participants]

Trains Logistic Regression on the chosen features and evaluates on a hold-out split,
ensuring split is restricted to couples present in the evaluation results, and using the
same couple subset for baseline (divorce_clean.csv) to keep parity.
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix


def build_feature_table(eval_results_path: str) -> pd.DataFrame:
    data = json.loads(Path(eval_results_path).read_text(encoding='utf-8'))
    preds = data.get('detailed_predictions', [])

    # Group by couple_id
    rows = {}
    for r in preds:
        cid = r.get('couple_id')
        if cid is None:
            continue
        if cid not in rows:
            rows[cid] = {
                'couple_id': cid,
                'y': r.get('ground_truth', 0),
                'observer_score': np.nan,
                'participant_final_score': np.nan,  # divorce likelihood average used earlier
                'husband_willingness': np.nan,
                'wife_willingness': np.nan,
            }
        if r.get('method') == 'observer':
            rows[cid]['observer_score'] = r.get('score', np.nan)
        elif r.get('method') == 'participant':
            rows[cid]['participant_final_score'] = r.get('score', np.nan)
            h = r.get('husband') or {}
            w = r.get('wife') or {}
            rows[cid]['husband_willingness'] = h.get('score', np.nan)
            rows[cid]['wife_willingness'] = w.get('score', np.nan)

    df = pd.DataFrame(list(rows.values())).sort_values('couple_id').reset_index(drop=True)

    # Derived features
    df['h_divorce'] = df['husband_willingness']
    df['w_divorce'] = df['wife_willingness']
    df['participants_product'] = df['h_divorce'] * df['w_divorce']

    return df


def train_and_report(
    X: np.ndarray,
    y: np.ndarray,
    feature_names,
    random_state=42,
    train_indices: np.ndarray | None = None,
    test_indices: np.ndarray | None = None,
):
    """Train LR and report metrics.

    If train_indices/test_indices are provided, use a fixed split; otherwise use 80/20 stratified.
    """
    if train_indices is not None and test_indices is not None:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    lr.fit(X_train, y_train)

    y_proba = lr.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = float('nan')

    report = classification_report(y_test, y_pred, target_names=['Married', 'Divorced'], zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1]).tolist()

    coefs = lr.coef_[0].tolist()
    intercept = float(lr.intercept_[0])

    result = {
        'accuracy': float(acc),
        'auc': float(auc),
        'coefficients': {name: coef for name, coef in zip(feature_names, coefs)},
        'intercept': intercept,
        'confusion_matrix': cm,
        'n_train': int(len(y_train)),
        'n_test': int(len(y_test))
    }
    return result


def main():
    parser = argparse.ArgumentParser(description='Calibrate LLM scores (kx+b) for divorce prediction')
    parser.add_argument('--eval-results', type=str, required=True, help='Path to evaluation_results_*.json')
    parser.add_argument('--clean-data', type=str, default='divorce_clean.csv', help='Clean feature CSV for baseline parity checks')
    parser.add_argument('--output', type=str, default='calibration_results.json', help='Where to save calibration report')
    parser.add_argument('--train-size', type=int, default=None, help='If set, use a fixed stratified split: train-size for training, remainder for testing')
    parser.add_argument('--split-seed', type=int, default=42, help='Random seed for fixed stratified split')
    args = parser.parse_args()

    df_scores = build_feature_table(args.eval_results)

    # Build a fixed stratified split over available couple_ids if requested
    fixed_train_idx = None
    fixed_test_idx = None
    if args.train_size is not None:
        n = int(args.train_size)
        if n < 2:
            raise ValueError('train-size must be >= 2')
        # We'll compute indices on the intersection of rows that have ALL potential features; then reuse mapping per model.
        # For consistency, use rows with any y defined.
        df_any = df_scores.dropna(subset=['y'])
        if df_any['y'].nunique() < 2:
            raise ValueError('Need both classes present to create a stratified split')
        # Build arrays
        y_any = df_any['y'].to_numpy()
        idx_any = np.arange(len(df_any))
        # Stratified sample exactly n train
        # Do simple per-class proportional selection with ceil/floor
        rng = np.random.RandomState(args.split_seed)
        pos_idx = idx_any[y_any == 1]
        neg_idx = idx_any[y_any == 0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            raise ValueError('Both classes required for fixed split')
        half = n // 2
        rem = n - 2 * half
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        take_pos = min(half + (1 if rem > 0 and len(pos_idx) >= len(neg_idx) else 0), len(pos_idx))
        take_neg = min(n - take_pos, len(neg_idx))
        train_any = np.concatenate([pos_idx[:take_pos], neg_idx[:take_neg]])
        # If still short, top up from remaining
        if train_any.size < n:
            remaining = np.concatenate([pos_idx[take_pos:], neg_idx[take_neg:]])
            rng.shuffle(remaining)
            extra = remaining[: max(0, n - train_any.size)]
            train_any = np.concatenate([train_any, extra])
        # Define test as complement
        mask = np.ones(len(df_any), dtype=bool)
        mask[train_any] = False
        test_any = np.where(mask)[0]
        # We'll need to map these indices to per-model filtered DataFrames later.
        df_any = df_any.reset_index().rename(columns={'index': '_orig_idx'})
    else:
        df_any = None

    # Drop rows with missing essential features for each model
    models = {}

    def split_indices_for(df_model: pd.DataFrame):
        """Map fixed split indices to this df_model.

        We match by couple_id against df_any to recover original index.
        """
        if df_any is None:
            return None, None
        # Build map from couple_id to row position in df_any
        map_any = {int(r.couple_id): i for i, r in df_any.iterrows()}
        # Collect per-model indices that exist in df_any
        model_idx = []
        for i, r in df_model.reset_index().iterrows():
            ci = int(r.couple_id)
            if ci in map_any:
                model_idx.append((i, map_any[ci]))
        if not model_idx:
            return None, None
        model_idx = np.array(model_idx)
        # df_any indices for this model
        any_positions = model_idx[:, 1]
        # recover which of those are in train_any vs test_any
        train_mask = np.isin(any_positions, train_any)
        test_mask = np.isin(any_positions, test_any)
        train_local = model_idx[train_mask, 0]
        test_local = model_idx[test_mask, 0]
        if train_local.size == 0 or test_local.size == 0:
            return None, None
        return train_local.astype(int), test_local.astype(int)

    # 1) participants: x = participants_product
    df1 = df_scores.dropna(subset=['participants_product'])
    if len(df1) >= 10 and df1['y'].nunique() == 2:
        X = df1[['participants_product']].to_numpy()
        y = df1['y'].to_numpy()
        ti, vi = split_indices_for(df1)
        models['participants_kx_plus_b'] = train_and_report(X, y, ['participants_product'], random_state=args.split_seed, train_indices=ti, test_indices=vi)
    else:
        models['participants_kx_plus_b'] = {'error': 'Insufficient data (need >=10 samples and both classes).'}

    # 2) observer: x = observer_score
    df2 = df_scores.dropna(subset=['observer_score'])
    if len(df2) >= 10 and df2['y'].nunique() == 2:
        X = df2[['observer_score']].to_numpy()
        y = df2['y'].to_numpy()
        ti, vi = split_indices_for(df2)
        models['observer_kx_plus_b'] = train_and_report(X, y, ['observer_score'], random_state=args.split_seed, train_indices=ti, test_indices=vi)
    else:
        models['observer_kx_plus_b'] = {'error': 'Insufficient data (need >=10 samples and both classes).'}

    # 3) mixed: x = [observer_score, participants_product]
    df3 = df_scores.dropna(subset=['observer_score', 'participants_product'])
    if len(df3) >= 10 and df3['y'].nunique() == 2:
        X = df3[['observer_score', 'participants_product']].to_numpy()
        y = df3['y'].to_numpy()
        ti, vi = split_indices_for(df3)
        models['mixed_linear'] = train_and_report(X, y, ['observer_score', 'participants_product'], random_state=args.split_seed, train_indices=ti, test_indices=vi)
    else:
        models['mixed_linear'] = {'error': 'Insufficient data (need >=10 samples and both classes).'}

    # 4) linear: x = [w_divorce, h_divorce, participants_product]
    df4 = df_scores.dropna(subset=['w_divorce', 'h_divorce', 'participants_product'])
    if len(df4) >= 10 and df4['y'].nunique() == 2:
        X = df4[['w_divorce', 'h_divorce', 'participants_product']].to_numpy()
        y = df4['y'].to_numpy()
        ti, vi = split_indices_for(df4)
        models['linear_whp'] = train_and_report(X, y, ['w_divorce', 'h_divorce', 'participants_product'], random_state=args.split_seed, train_indices=ti, test_indices=vi)
    else:
        models['linear_whp'] = {'error': 'Insufficient data (need >=10 samples and both classes).'}

    # Baseline on same couples subset (parity check)
    try:
        df_clean = pd.read_csv(args.clean_data)
        subset_ids = sorted(df_scores['couple_id'].astype(int).unique().tolist())
        df_subset = df_clean.iloc[subset_ids].copy()
        Xb = df_subset.drop('Class', axis=1)
        yb = (df_subset['Class'] == 0).astype(int).to_numpy()
        # Align baseline split with fixed split if provided by mapping couple ids
        if df_any is not None:
            # Couple ids in df_subset correspond to index positions
            # Build local indices arrays for baseline
            # Map df_subset rows (which correspond to couple indices) to df_any couple ids
            # Here, df_subset rows order equals subset_ids; df_any holds _orig_idx from original df_scores
            # We select rows whose couple_id is in df_any
            subset_map = {int(cid): i for i, cid in enumerate(subset_ids)}
            any_couple_ids = df_any['couple_id'].astype(int).tolist()
            local_positions = [subset_map[cid] for cid in any_couple_ids if cid in subset_map]
            local_positions = np.array(local_positions)
            # build masks using train_any/test_any positions
            # We already computed train_any/test_any over df_any rows
            train_local = local_positions[train_any]
            test_local = local_positions[test_any]
            base = train_and_report(
                Xb.to_numpy(), yb, [f'f{i}' for i in range(Xb.shape[1])], random_state=args.split_seed,
                train_indices=train_local, test_indices=test_local
            )
        else:
            base = train_and_report(Xb.to_numpy(), yb, [f'f{i}' for i in range(Xb.shape[1])], random_state=args.split_seed)
    except Exception as e:
        base = {'error': f'Baseline parity training failed: {e}'}

    report = {
        'n_couples_in_scores': int(len(df_scores)),
        'models': models,
        'baseline_same_subset': base
    }

    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    # also dump the features used for debugging
    Path('llm_scores_features.csv').write_text(df_scores.to_csv(index=False), encoding='utf-8')
    print('\n✅ Saved calibration report to:', args.output)
    print('✅ Saved feature table to: llm_scores_features.csv')


if __name__ == '__main__':
    main()
