"""
Baseline Matching Algorithms V2 - Fair Comparison with LLM

Two versions:
1. Baseline V1 (Time 1 only): Uses SAME data as LLM (pre-event survey)
2. Baseline V2 (Time 1 + Time 2): Uses pre-event + post-event reflections

Training: Uses full Speed Dating dataset (8378 records) EXCLUDING the 100 test pairs
Testing: Evaluates on the SAME 100 pairs that LLM sees

This ensures:
- No data leakage (train != test)
- Fair comparison (same test set as LLM)
- Two difficulty levels (with/without future knowledge)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: brew install libomp && pip install xgboost")

warnings.filterwarnings('ignore')


class BaselineModel:
    """
    Unified baseline model with two feature extraction modes:
    - Mode 'v1': Time 1 only (same as LLM sees)
    - Mode 'v2': Time 1 + Time 2 (with post-event reflections)
    """
    
    def __init__(self, model_type: str = 'logistic', feature_mode: str = 'v1'):
        """
        Args:
            model_type: 'logistic', 'random_forest', or 'xgboost'
            feature_mode: 'v1' (Time 1 only) or 'v2' (Time 1 + Time 2)
        """
        self.model_type = model_type
        self.feature_mode = feature_mode
        self.scaler = StandardScaler()
        self.model = None
        
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost not available")
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def _extract_features_v1(self, person1_data: Dict, person2_data: Dict) -> np.ndarray:
        """
        Extract V1 features (Time 1 only - same as LLM sees).
        Total: ~98 features (same as current implementation)
        """
        features = []
        
        # === PERSON 1 FEATURES (Time 1) ===
        
        # Demographics (5 features)
        features.append(float(person1_data.get('age', 25)) / 100.0)
        features.append(float(person1_data.get('gender', 0)))
        
        race1 = person1_data.get('race', 0)
        features.extend([
            1.0 if race1 == 2 else 0.0,  # European
            1.0 if race1 == 4 else 0.0,  # Asian
            1.0 if race1 == 3 else 0.0   # Latino
        ])
        
        # Preferences (6 features)
        prefs1 = person1_data.get('preferences_self', person1_data.get('attr1_1', {}))
        if isinstance(prefs1, dict):
            features.extend([
                float(prefs1.get('attractiveness', 0) or 0),
                float(prefs1.get('sincerity', 0) or 0),
                float(prefs1.get('intelligence', 0) or 0),
                float(prefs1.get('fun', 0) or 0),
                float(prefs1.get('ambition', 0) or 0),
                float(prefs1.get('shared_interests', 0) or 0)
            ])
        else:
            # Handle direct values (from CSV)
            features.extend([
                float(person1_data.get('attr1_1', 0) or 0),
                float(person1_data.get('sinc1_1', 0) or 0),
                float(person1_data.get('intel1_1', 0) or 0),
                float(person1_data.get('fun1_1', 0) or 0),
                float(person1_data.get('amb1_1', 0) or 0),
                float(person1_data.get('shar1_1', 0) or 0)
            ])
        
        # Self-ratings (5 features)
        ratings1 = person1_data.get('self_ratings', {})
        if isinstance(ratings1, dict) and len(ratings1) > 0:
            features.extend([
                float(ratings1.get('attractiveness', 0) or 0),
                float(ratings1.get('sincerity', 0) or 0),
                float(ratings1.get('intelligence', 0) or 0),
                float(ratings1.get('fun', 0) or 0),
                float(ratings1.get('ambition', 0) or 0)
            ])
        else:
            features.extend([
                float(person1_data.get('attr3_1', 0) or 0),
                float(person1_data.get('sinc3_1', 0) or 0),
                float(person1_data.get('intel3_1', 0) or 0),
                float(person1_data.get('fun3_1', 0) or 0),
                float(person1_data.get('amb3_1', 0) or 0)
            ])
        
        # Others' perception (5 features)
        others1 = person1_data.get('others_perception', {})
        if isinstance(others1, dict) and len(others1) > 0:
            features.extend([
                float(others1.get('attractiveness', 0) or 0),
                float(others1.get('sincerity', 0) or 0),
                float(others1.get('intelligence', 0) or 0),
                float(others1.get('fun', 0) or 0),
                float(others1.get('ambition', 0) or 0)
            ])
        else:
            features.extend([
                float(person1_data.get('attr5_1', 0) or 0),
                float(person1_data.get('sinc5_1', 0) or 0),
                float(person1_data.get('intel5_1', 0) or 0),
                float(person1_data.get('fun5_1', 0) or 0),
                float(person1_data.get('amb5_1', 0) or 0)
            ])
        
        # Interests (17 features)
        interest_keys = ['sports', 'tvsports', 'exercise', 'dining', 'museums', 'art',
                        'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater',
                        'movies', 'concerts', 'music', 'shopping', 'yoga']
        interests1 = person1_data.get('interests', {})
        for interest in interest_keys:
            if isinstance(interests1, dict):
                features.append(float(interests1.get(interest, 0) or 0))
            else:
                features.append(float(person1_data.get(interest, 0) or 0))
        
        # Dating behavior (7 features)
        features.append(float(person1_data.get('go_out', 3) or 3))
        features.append(float(person1_data.get('date', 3) or 3))
        features.append(float(person1_data.get('exphappy', 5) or 5))
        
        goal1 = person1_data.get('goal', 0) or 0
        features.extend([
            1.0 if goal1 == 1 else 0.0,
            1.0 if goal1 == 2 else 0.0,
            1.0 if goal1 == 4 else 0.0
        ])
        
        # Values (2 features)
        features.append(float(person1_data.get('imprace', 0) or 0))
        features.append(float(person1_data.get('imprelig', 0) or 0))
        
        # === PERSON 2 FEATURES (mirror structure) ===
        
        features.append(float(person2_data.get('age', 25)) / 100.0)
        features.append(float(person2_data.get('gender', 0)))
        
        race2 = person2_data.get('race', 0)
        features.extend([
            1.0 if race2 == 2 else 0.0,
            1.0 if race2 == 4 else 0.0,
            1.0 if race2 == 3 else 0.0
        ])
        
        prefs2 = person2_data.get('preferences_self', person2_data.get('attr1_1', {}))
        if isinstance(prefs2, dict):
            features.extend([
                float(prefs2.get('attractiveness', 0) or 0),
                float(prefs2.get('sincerity', 0) or 0),
                float(prefs2.get('intelligence', 0) or 0),
                float(prefs2.get('fun', 0) or 0),
                float(prefs2.get('ambition', 0) or 0),
                float(prefs2.get('shared_interests', 0) or 0)
            ])
        else:
            features.extend([
                float(person2_data.get('attr1_1', 0) or 0),
                float(person2_data.get('sinc1_1', 0) or 0),
                float(person2_data.get('intel1_1', 0) or 0),
                float(person2_data.get('fun1_1', 0) or 0),
                float(person2_data.get('amb1_1', 0) or 0),
                float(person2_data.get('shar1_1', 0) or 0)
            ])
        
        ratings2 = person2_data.get('self_ratings', {})
        if isinstance(ratings2, dict) and len(ratings2) > 0:
            features.extend([
                float(ratings2.get('attractiveness', 0) or 0),
                float(ratings2.get('sincerity', 0) or 0),
                float(ratings2.get('intelligence', 0) or 0),
                float(ratings2.get('fun', 0) or 0),
                float(ratings2.get('ambition', 0) or 0)
            ])
        else:
            features.extend([
                float(person2_data.get('attr3_1', 0) or 0),
                float(person2_data.get('sinc3_1', 0) or 0),
                float(person2_data.get('intel3_1', 0) or 0),
                float(person2_data.get('fun3_1', 0) or 0),
                float(person2_data.get('amb3_1', 0) or 0)
            ])
        
        others2 = person2_data.get('others_perception', {})
        if isinstance(others2, dict) and len(others2) > 0:
            features.extend([
                float(others2.get('attractiveness', 0) or 0),
                float(others2.get('sincerity', 0) or 0),
                float(others2.get('intelligence', 0) or 0),
                float(others2.get('fun', 0) or 0),
                float(others2.get('ambition', 0) or 0)
            ])
        else:
            features.extend([
                float(person2_data.get('attr5_1', 0) or 0),
                float(person2_data.get('sinc5_1', 0) or 0),
                float(person2_data.get('intel5_1', 0) or 0),
                float(person2_data.get('fun5_1', 0) or 0),
                float(person2_data.get('amb5_1', 0) or 0)
            ])
        
        interests2 = person2_data.get('interests', {})
        for interest in interest_keys:
            if isinstance(interests2, dict):
                features.append(float(interests2.get(interest, 0) or 0))
            else:
                features.append(float(person2_data.get(interest, 0) or 0))
        
        features.append(float(person2_data.get('go_out', 3) or 3))
        features.append(float(person2_data.get('date', 3) or 3))
        features.append(float(person2_data.get('exphappy', 5) or 5))
        
        goal2 = person2_data.get('goal', 0) or 0
        features.extend([
            1.0 if goal2 == 1 else 0.0,
            1.0 if goal2 == 2 else 0.0,
            1.0 if goal2 == 4 else 0.0
        ])
        
        features.append(float(person2_data.get('imprace', 0) or 0))
        features.append(float(person2_data.get('imprelig', 0) or 0))
        
        # === DERIVED FEATURES ===
        
        # Interest overlap
        if isinstance(interests1, dict):
            p1_interests_high = set(k for k in interest_keys if float(interests1.get(k, 0) or 0) > 5)
        else:
            p1_interests_high = set(k for k in interest_keys if float(person1_data.get(k, 0) or 0) > 5)
        
        if isinstance(interests2, dict):
            p2_interests_high = set(k for k in interest_keys if float(interests2.get(k, 0) or 0) > 5)
        else:
            p2_interests_high = set(k for k in interest_keys if float(person2_data.get(k, 0) or 0) > 5)
        
        if len(p1_interests_high) + len(p2_interests_high) > 0:
            interest_overlap = len(p1_interests_high & p2_interests_high) / len(p1_interests_high | p2_interests_high)
        else:
            interest_overlap = 0.0
        features.append(float(interest_overlap))
        
        # Preference-rating alignments
        # P1 values what P2 has
        if isinstance(prefs1, dict):
            p1_pref_vec = np.array([
                float(prefs1.get('attractiveness', 0) or 0),
                float(prefs1.get('sincerity', 0) or 0),
                float(prefs1.get('intelligence', 0) or 0),
                float(prefs1.get('fun', 0) or 0),
                float(prefs1.get('ambition', 0) or 0)
            ])
        else:
            p1_pref_vec = np.array([
                float(person1_data.get('attr1_1', 0) or 0),
                float(person1_data.get('sinc1_1', 0) or 0),
                float(person1_data.get('intel1_1', 0) or 0),
                float(person1_data.get('fun1_1', 0) or 0),
                float(person1_data.get('amb1_1', 0) or 0)
            ])
        
        if isinstance(ratings2, dict) and len(ratings2) > 0:
            p2_rating_vec = np.array([
                float(ratings2.get('attractiveness', 0) or 0),
                float(ratings2.get('sincerity', 0) or 0),
                float(ratings2.get('intelligence', 0) or 0),
                float(ratings2.get('fun', 0) or 0),
                float(ratings2.get('ambition', 0) or 0)
            ])
        else:
            p2_rating_vec = np.array([
                float(person2_data.get('attr3_1', 0) or 0),
                float(person2_data.get('sinc3_1', 0) or 0),
                float(person2_data.get('intel3_1', 0) or 0),
                float(person2_data.get('fun3_1', 0) or 0),
                float(person2_data.get('amb3_1', 0) or 0)
            ])
        
        if np.std(p1_pref_vec) > 0 and np.std(p2_rating_vec) > 0:
            alignment = np.corrcoef(p1_pref_vec, p2_rating_vec)[0, 1]
            if np.isnan(alignment):
                alignment = 0.0
        else:
            alignment = 0.0
        features.append(float(alignment))
        
        # Reverse alignment
        if isinstance(prefs2, dict):
            p2_pref_vec = np.array([
                float(prefs2.get('attractiveness', 0) or 0),
                float(prefs2.get('sincerity', 0) or 0),
                float(prefs2.get('intelligence', 0) or 0),
                float(prefs2.get('fun', 0) or 0),
                float(prefs2.get('ambition', 0) or 0)
            ])
        else:
            p2_pref_vec = np.array([
                float(person2_data.get('attr1_1', 0) or 0),
                float(person2_data.get('sinc1_1', 0) or 0),
                float(person2_data.get('intel1_1', 0) or 0),
                float(person2_data.get('fun1_1', 0) or 0),
                float(person2_data.get('amb1_1', 0) or 0)
            ])
        
        if isinstance(ratings1, dict) and len(ratings1) > 0:
            p1_rating_vec = np.array([
                float(ratings1.get('attractiveness', 0) or 0),
                float(ratings1.get('sincerity', 0) or 0),
                float(ratings1.get('intelligence', 0) or 0),
                float(ratings1.get('fun', 0) or 0),
                float(ratings1.get('ambition', 0) or 0)
            ])
        else:
            p1_rating_vec = np.array([
                float(person1_data.get('attr3_1', 0) or 0),
                float(person1_data.get('sinc3_1', 0) or 0),
                float(person1_data.get('intel3_1', 0) or 0),
                float(person1_data.get('fun3_1', 0) or 0),
                float(person1_data.get('amb3_1', 0) or 0)
            ])
        
        if np.std(p2_pref_vec) > 0 and np.std(p1_rating_vec) > 0:
            reverse_alignment = np.corrcoef(p2_pref_vec, p1_rating_vec)[0, 1]
            if np.isnan(reverse_alignment):
                reverse_alignment = 0.0
        else:
            reverse_alignment = 0.0
        features.append(float(reverse_alignment))
        
        # Age difference
        age1 = float(person1_data.get('age', 25))
        age2 = float(person2_data.get('age', 25))
        age_diff = abs(age1 - age2) / 20.0
        features.append(float(age_diff))
        
        # Same race
        features.append(1.0 if race1 == race2 else 0.0)
        
        # Goal compatibility
        features.append(1.0 if goal1 == 4 and goal2 == 4 else 0.0)
        
        return np.array(features)
    
    def _extract_features_v2(self, person1_data: Dict, person2_data: Dict) -> np.ndarray:
        """
        Extract V2 features (Time 1 + Time 2 - includes post-event reflections).
        Adds Time 2 updated preferences and ratings on top of V1 features.
        Total: ~120+ features
        """
        # Start with V1 features
        features_v1 = self._extract_features_v1(person1_data, person2_data)
        features = features_v1.tolist()
        
        # Add Time 2 features (post-event reflections)
        
        # Person 1 Time 2 updated preferences (6 features)
        features.extend([
            float(person1_data.get('attr1_2', 0) or 0),
            float(person1_data.get('sinc1_2', 0) or 0),
            float(person1_data.get('intel1_2', 0) or 0),
            float(person1_data.get('fun1_2', 0) or 0),
            float(person1_data.get('amb1_2', 0) or 0),
            float(person1_data.get('shar1_2', 0) or 0)
        ])
        
        # Person 1 Time 2 updated self-ratings (5 features)
        features.extend([
            float(person1_data.get('attr3_2', 0) or 0),
            float(person1_data.get('sinc3_2', 0) or 0),
            float(person1_data.get('intel3_2', 0) or 0),
            float(person1_data.get('fun3_2', 0) or 0),
            float(person1_data.get('amb3_2', 0) or 0)
        ])
        
        # Person 1 satisfaction (3 features)
        features.extend([
            float(person1_data.get('satis_2', 0) or 0),
            float(person1_data.get('length', 3) or 3),
            float(person1_data.get('numdat_2', 3) or 3)
        ])
        
        # Person 2 Time 2 updated preferences (6 features)
        features.extend([
            float(person2_data.get('attr1_2', 0) or 0),
            float(person2_data.get('sinc1_2', 0) or 0),
            float(person2_data.get('intel1_2', 0) or 0),
            float(person2_data.get('fun1_2', 0) or 0),
            float(person2_data.get('amb1_2', 0) or 0),
            float(person2_data.get('shar1_2', 0) or 0)
        ])
        
        # Person 2 Time 2 updated self-ratings (5 features)
        features.extend([
            float(person2_data.get('attr3_2', 0) or 0),
            float(person2_data.get('sinc3_2', 0) or 0),
            float(person2_data.get('intel3_2', 0) or 0),
            float(person2_data.get('fun3_2', 0) or 0),
            float(person2_data.get('amb3_2', 0) or 0)
        ])
        
        # Person 2 satisfaction (3 features)
        features.extend([
            float(person2_data.get('satis_2', 0) or 0),
            float(person2_data.get('length', 3) or 3),
            float(person2_data.get('numdat_2', 3) or 3)
        ])
        
        return np.array(features)
    
    def _extract_features(self, person1_data: Dict, person2_data: Dict) -> np.ndarray:
        """Route to appropriate feature extraction based on mode."""
        if self.feature_mode == 'v1':
            return self._extract_features_v1(person1_data, person2_data)
        elif self.feature_mode == 'v2':
            return self._extract_features_v2(person1_data, person2_data)
        else:
            raise ValueError(f"Unknown feature_mode: {self.feature_mode}")
    
    def train(self, train_data: List[Dict], verbose: bool = True):
        """
        Train the model on labeled data.
        
        Args:
            train_data: List of dicts with keys:
                - person1_data: Dict of person 1 features
                - person2_data: Dict of person 2 features
                - match: bool (ground truth)
        """
        X = []
        y = []
        
        for sample in train_data:
            try:
                features = self._extract_features(
                    sample['person1_data'],
                    sample['person2_data']
                )
                X.append(features)
                y.append(1 if sample['match'] else 0)
            except Exception as e:
                if verbose:
                    print(f"Warning: Skipping sample due to error: {e}")
                continue
        
        X = np.array(X)
        y = np.array(y)
        
        if verbose:
            print(f"Training data: {len(y)} samples")
            print(f"  Feature dimensions: {X.shape[1]}")
            print(f"  Positive class: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        if verbose:
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
            print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    def predict(self, person1_data: Dict, person2_data: Dict) -> Tuple[bool, float]:
        """
        Predict if two people would match.
        
        Returns:
            (prediction, probability)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        features = self._extract_features(person1_data, person2_data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0][1]
        
        return bool(prediction), float(probability)


def load_full_dataset(csv_path: str = 'Speed Dating Data.csv') -> pd.DataFrame:
    """
    Load the full Speed Dating dataset from CSV.
    
    Returns:
        DataFrame with all dating records
    """
    print(f"\nLoading full dataset from {csv_path}...")
    df = pd.read_csv(csv_path, encoding='latin1')
    print(f"  Total records: {len(df)}")
    print(f"  Unique participants: {df['iid'].nunique()}")
    print(f"  Match rate: {df['match'].mean()*100:.1f}%")
    
    return df


def load_test_pairs(personas_path: str = 'results/personas.json') -> List[Dict]:
    """
    Load the 100 test pairs that LLM will evaluate on.
    
    Returns:
        List of test pair IDs to exclude from training
    """
    with open(personas_path, 'r') as f:
        personas = json.load(f)
    
    test_iids = set()
    for pair in personas:
        test_iids.add(pair['person1']['iid'])
        test_iids.add(pair['person2']['iid'])
    
    print(f"\nTest set: {len(personas)} pairs ({len(test_iids)} unique participants)")
    
    return personas, test_iids


def create_train_test_split(
    df: pd.DataFrame,
    test_iids: set
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create training set (excluding test IIDs) and test set.
    
    Returns:
        (train_pairs, test_pairs)
    """
    train_records = df[~df['iid'].isin(test_iids) & ~df['pid'].isin(test_iids)]
    
    print(f"\nCreating train/test split...")
    print(f"  Full dataset: {len(df)} records")
    print(f"  Training set: {len(train_records)} records")
    print(f"  Test set: Will be loaded from personas.json")
    
    # Convert training records to pair format
    train_pairs = []
    for _, row in train_records.iterrows():
        person1_data = row.to_dict()
        # We need to find the partner (pid) - this is tricky with the CSV format
        # For now, use the row as both sides (will need refinement)
        person2_data = person1_data.copy()
        
        train_pairs.append({
            'person1_data': person1_data,
            'person2_data': person2_data,
            'match': bool(row['match'] == 1)
        })
    
    return train_pairs


def evaluate_baselines_v2(
    csv_path: str = 'Speed Dating Data.csv',
    personas_path: str = 'results/personas.json',
    output_dir: str = 'results'
):
    """
    Evaluate both baseline versions on the same 100 test pairs as LLM.
    
    V1: Time 1 only (fair comparison with LLM)
    V2: Time 1 + Time 2 (with future knowledge)
    """
    print("=" * 70)
    print("BASELINE MODELS V2 - FAIR COMPARISON WITH LLM")
    print("=" * 70)
    
    # Load full dataset
    df = load_full_dataset(csv_path)
    
    # Load test pairs (same as LLM sees)
    test_personas, test_iids = load_test_pairs(personas_path)
    
    # Create training set (excluding test participants)
    train_pairs = create_train_test_split(df, test_iids)
    
    # Prepare test pairs from personas.json
    test_pairs = []
    for pair in test_personas:
        person1_data = pair['person1'].get('pre_event_data', {})
        person2_data = pair['person2'].get('pre_event_data', {})
        
        # Merge with time2 data for V2
        time2_p1 = pair['person1'].get('time2_reflection', {})
        time2_p2 = pair['person2'].get('time2_reflection', {})
        
        # Add CSV-style fields for compatibility
        # (This conversion might need adjustment based on actual data structure)
        
        test_pairs.append({
            'pair_id': pair.get('pair_id'),
            'person1_data': person1_data,
            'person2_data': person2_data,
            'person1_data_full': {**person1_data, **flatten_time2(time2_p1)},
            'person2_data_full': {**person2_data, **flatten_time2(time2_p2)},
            'match': pair['ground_truth']['match']
        })
    
    print(f"\nTest set: {len(test_pairs)} pairs")
    print(f"  Matches: {sum(p['match'] for p in test_pairs)} ({sum(p['match'] for p in test_pairs)/len(test_pairs)*100:.1f}%)")
    
    results = {}
    
    # === BASELINE V1: Time 1 Only (Fair with LLM) ===
    print("\n" + "=" * 70)
    print("BASELINE V1: TIME 1 ONLY (Same info as LLM)")
    print("=" * 70)
    
    for model_type in ['logistic', 'random_forest']:
        print(f"\n--- {model_type.upper().replace('_', ' ')} ---")
        
        model_v1 = BaselineModel(model_type=model_type, feature_mode='v1')
        model_v1.train(train_pairs, verbose=True)
        
        # Evaluate on test set
        predictions_v1 = []
        probs_v1 = []
        
        for pair in test_pairs:
            try:
                pred, prob = model_v1.predict(pair['person1_data'], pair['person2_data'])
                predictions_v1.append(pred)
                probs_v1.append(prob)
            except Exception as e:
                print(f"Warning: Prediction failed for {pair.get('pair_id')}: {e}")
                predictions_v1.append(False)
                probs_v1.append(0.5)
        
        y_true = [p['match'] for p in test_pairs]
        
        results[f'{model_type}_v1'] = {
            'model': f'{model_type} V1 (Time 1 only)',
            'feature_mode': 'v1',
            'accuracy': accuracy_score(y_true, predictions_v1),
            'precision': precision_score(y_true, predictions_v1, zero_division=0),
            'recall': recall_score(y_true, predictions_v1, zero_division=0),
            'f1': f1_score(y_true, predictions_v1, zero_division=0),
            'auc': roc_auc_score(y_true, probs_v1),
            'confusion_matrix': confusion_matrix(y_true, predictions_v1).tolist()
        }
        
        print(f"  Accuracy:  {results[f'{model_type}_v1']['accuracy']:.3f}")
        print(f"  Precision: {results[f'{model_type}_v1']['precision']:.3f}")
        print(f"  Recall:    {results[f'{model_type}_v1']['recall']:.3f}")
        print(f"  F1 Score:  {results[f'{model_type}_v1']['f1']:.3f}")
        print(f"  AUC:       {results[f'{model_type}_v1']['auc']:.3f}")
    
    # === BASELINE V2: Time 1 + Time 2 (With Future Knowledge) ===
    print("\n" + "=" * 70)
    print("BASELINE V2: TIME 1 + TIME 2 (With post-event reflections)")
    print("=" * 70)
    
    for model_type in ['logistic', 'random_forest']:
        print(f"\n--- {model_type.upper().replace('_', ' ')} ---")
        
        model_v2 = BaselineModel(model_type=model_type, feature_mode='v2')
        
        # Need to update train_pairs with Time 2 data
        # For now, skip if Time 2 data not available in training
        print("  Note: V2 requires Time 2 data in training set (future work)")
        print("  Skipping V2 for now - need to process full CSV with Time 2 fields")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    comparison_df = pd.DataFrame([
        {
            'Model': r['model'],
            'Mode': r['feature_mode'],
            'Accuracy': f"{r['accuracy']:.3f}",
            'Precision': f"{r['precision']:.3f}",
            'Recall': f"{r['recall']:.3f}",
            'F1': f"{r['f1']:.3f}",
            'AUC': f"{r['auc']:.3f}"
        }
        for r in results.values() if 'feature_mode' in r
    ])
    
    print(comparison_df.to_string(index=False))
    
    # Save results
    output_path = f"{output_dir}/baseline_comparison_v2.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    return results


def flatten_time2(time2_data: Dict) -> Dict:
    """Flatten time2_reflection nested structure to flat dict."""
    flat = {}
    
    # Updated preferences
    prefs = time2_data.get('updated_preferences_self', {})
    if prefs:
        flat['attr1_2'] = prefs.get('attractiveness', 0)
        flat['sinc1_2'] = prefs.get('sincerity', 0)
        flat['intel1_2'] = prefs.get('intelligence', 0)
        flat['fun1_2'] = prefs.get('fun', 0)
        flat['amb1_2'] = prefs.get('ambition', 0)
        flat['shar1_2'] = prefs.get('shared_interests', 0)
    
    # Updated self-ratings
    ratings = time2_data.get('updated_self_ratings', {})
    if ratings:
        flat['attr3_2'] = ratings.get('attractiveness', 0)
        flat['sinc3_2'] = ratings.get('sincerity', 0)
        flat['intel3_2'] = ratings.get('intelligence', 0)
        flat['fun3_2'] = ratings.get('fun', 0)
        flat['amb3_2'] = ratings.get('ambition', 0)
    
    # Satisfaction
    satisfaction = time2_data.get('satisfaction', {})
    if satisfaction:
        flat['satis_2'] = satisfaction.get('satis_2', 0)
        flat['length'] = satisfaction.get('length', 3)
        flat['numdat_2'] = satisfaction.get('numdat_2', 3)
    
    return flat


if __name__ == '__main__':
    import os
    
    # Change to project root if running from experiments/
    if os.path.basename(os.getcwd()) == 'experiments':
        os.chdir('..')
    
    results = evaluate_baselines_v2()
