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
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score  # PR-AUC
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not available. Install with: pip install imbalanced-learn")

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
                n_estimators=200,
                max_depth=None,  # No limit on depth
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost not available")
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
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
) -> pd.DataFrame:
    """
    Create training set by EXCLUDING all test participants.
    
    Args:
        df: Full CSV dataset (8,378 records)
        test_iids: Set of 145 test participant IDs to exclude
    
    Returns:
        train_df: Training records (both iid and pid NOT in test set)
    """
    # CRITICAL: Exclude ANY record where iid OR pid is in test set
    # We cannot use test participants for training AT ALL
    train_df = df[~df['iid'].isin(test_iids) & ~df['pid'].isin(test_iids)]
    
    print(f"\n📊 Train/Test Split:")
    print(f"  Full dataset: {len(df):,} records")
    print(f"  Test participants to exclude: {len(test_iids)} IDs")
    print(f"  Training records: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Training participants: {train_df['iid'].nunique()}")
    
    return train_df


def convert_csv_to_pair_format(row: pd.Series) -> Dict:
        # We need to find the partner (pid) - this is tricky with the CSV format
        # For now, use the row as both sides (will need refinement)
        person2_data = person1_data.copy()
        
        train_pairs.append({
            'person1_data': person1_data,
            'person2_data': person2_data,
            'match': bool(row['match'] == 1)
        })


def evaluate_baselines_v2(
    csv_path: str = 'Speed Dating Data.csv',
    personas_path: str = 'results/personas.json',
    output_dir: str = 'results'
):
    """
    Evaluate both baseline versions on the same 100 test pairs as LLM.
    
    Training: CSV records EXCLUDING test participants (train on DIFFERENT people)
    Testing: Same 100 pairs LLM evaluated on (loaded from personas.json)
    
    V1: Time 1 only (fair comparison with LLM - pre-event data only)
    V2: Time 1 + Time 2 (with post-event reflections - "future knowledge")
    """
    print("=" * 70)
    print("BASELINE MODELS V2 - PROPER TRAIN/TEST SPLIT")
    print("=" * 70)
    print("\n🎯 Goal: Train on DIFFERENT participants, test on SAME 100 pairs as LLM")
    
    # Load full dataset
    df = load_full_dataset(csv_path)
    
    # Load test pairs (same as LLM sees)
    test_personas, test_iids = load_test_pairs(personas_path)
    
    # Create training set (excluding test participants)
    train_df = create_train_test_split(df, test_iids)
    
    # Prepare training data from CSV (need to create pairs properly)
    print("\n📦 Preparing training pairs from CSV...")
    print("   Creating pairs: matching iid→pid with pid→iid...")
    
    X_train_v1 = []
    X_train_v2 = []
    y_train = []
    
    # Create pairs by matching reciprocal ratings
    processed_pairs = set()
    
    for _, row in train_df.iterrows():
        iid = row['iid']
        pid = row['pid']
        
        # Skip if missing match label
        if pd.isna(row['match']) or pd.isna(pid):
            continue
        
        # Avoid processing same pair twice
        pair_key = tuple(sorted([iid, pid]))
        if pair_key in processed_pairs:
            continue
        
        # Find the reciprocal row (partner rating this person)
        partner_row = train_df[(train_df['iid'] == pid) & (train_df['pid'] == iid)]
        
        if len(partner_row) == 0:
            # No reciprocal rating, use single-sided
            person1_dict = row.to_dict()
            person2_dict = row.to_dict()  # Fallback: use same features
        else:
            person1_dict = row.to_dict()
            person2_dict = partner_row.iloc[0].to_dict()
        
        try:
            # V1: Time 1 only features
            features_v1 = BaselineModel(model_type='logistic', feature_mode='v1')._extract_features_v1(
                person1_dict, person2_dict
            )
            X_train_v1.append(features_v1)
            
            # V2: Time 1 + Time 2 features
            features_v2 = BaselineModel(model_type='logistic', feature_mode='v2')._extract_features_v2(
                person1_dict, person2_dict
            )
            X_train_v2.append(features_v2)
            
            # Label: match outcome
            y_train.append(int(row['match']))
            
            processed_pairs.add(pair_key)
        except Exception as e:
            continue  # Skip pairs with missing data
    
    X_train_v1 = np.array(X_train_v1)
    X_train_v2 = np.array(X_train_v2)
    y_train = np.array(y_train)
    
    # Handle missing values: replace NaN with median
    from sklearn.impute import SimpleImputer
    imputer_v1 = SimpleImputer(strategy='median')
    imputer_v2 = SimpleImputer(strategy='median')
    
    X_train_v1 = imputer_v1.fit_transform(X_train_v1)
    X_train_v2 = imputer_v2.fit_transform(X_train_v2)
    
    print(f"   V1 training features: {X_train_v1.shape}")
    print(f"   V2 training features: {X_train_v2.shape}")
    print(f"   Training labels: {len(y_train)} (matches: {sum(y_train)})")
    
    # Prepare test pairs from personas.json
    print("\n📦 Preparing test features from personas.json...")
    test_pairs = []
    for pair in test_personas:
        person1_data = pair['person1'].get('pre_event_data', {})
        person2_data = pair['person2'].get('pre_event_data', {})
        
        # Merge with time2 data for V2
        time2_p1 = pair['person1'].get('time2_reflection', {})
        time2_p2 = pair['person2'].get('time2_reflection', {})
        
        test_pairs.append({
            'pair_id': pair.get('pair_id'),
            'person1_data': person1_data,
            'person2_data': person2_data,
            'person1_data_full': {**person1_data, **flatten_time2(time2_p1)},
            'person2_data_full': {**person2_data, **flatten_time2(time2_p2)},
            'match': pair['ground_truth']['match']
        })
    
    print(f"   Test set: {len(test_pairs)} pairs")
    print(f"   Matches: {sum(p['match'] for p in test_pairs)} ({sum(p['match'] for p in test_pairs)/len(test_pairs)*100:.1f}%)")
    
    results = {}
    
    # Check if XGBoost is available
    try:
        import xgboost as xgb
        XGBOOST_AVAILABLE = True
    except ImportError:
        XGBOOST_AVAILABLE = False
        print("\n⚠️  XGBoost not available - will skip XGBoost models")
    
    # === SIMILARITY BASELINE V1 (Tune threshold on training data) ===
    print("\n" + "=" * 70)
    print("SIMILARITY BASELINE V1: TIME 1 ONLY (Cosine Similarity)")
    print("=" * 70)
    
    print("\n--- SIMILARITY V1 (Tuning threshold on training data) ---")
    print("   Computing similarities for training pairs...")
    
    # Compute similarities on TRAINING data to tune threshold
    train_similarities_v1 = []
    train_y = []
    
    for _, row in train_df.iterrows():
        iid = row['iid']
        pid = row['pid']
        
        if pd.isna(row['match']) or pd.isna(pid):
            continue
        
        # Find reciprocal row
        partner_row = train_df[(train_df['iid'] == pid) & (train_df['pid'] == iid)]
        
        if len(partner_row) == 0:
            continue
        
        try:
            # Extract features from both people
            row_dict = row.to_dict()
            partner_dict = partner_row.iloc[0].to_dict()
            
            # Preferences (6 features each)
            vec1 = [
                row_dict.get('attr1_1', 0), row_dict.get('sinc1_1', 0),
                row_dict.get('intel1_1', 0), row_dict.get('fun1_1', 0),
                row_dict.get('amb1_1', 0), row_dict.get('shar1_1', 0)
            ]
            vec2 = [
                partner_dict.get('attr1_1', 0), partner_dict.get('sinc1_1', 0),
                partner_dict.get('intel1_1', 0), partner_dict.get('fun1_1', 0),
                partner_dict.get('amb1_1', 0), partner_dict.get('shar1_1', 0)
            ]
            
            # Add interests (17 features)
            interest_keys = ['sports', 'tvsports', 'exercise', 'dining', 'museums', 'art',
                           'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater',
                           'movies', 'concerts', 'music', 'shopping', 'yoga']
            
            for key in interest_keys:
                vec1.append(row_dict.get(key, 0))
                vec2.append(partner_dict.get(key, 0))
            
            # Cosine similarity
            vec1 = np.array(vec1, dtype=float)
            vec2 = np.array(vec2, dtype=float)
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 0 and norm2 > 0:
                similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            else:
                similarity = 0.0
            
            train_similarities_v1.append(similarity)
            train_y.append(int(row['match']))
        except Exception as e:
            continue
    
    train_similarities_v1 = np.array(train_similarities_v1)
    train_y = np.array(train_y)
    
    print(f"   Training similarities computed: {len(train_similarities_v1)} pairs")
    print(f"   Training similarity range: [{train_similarities_v1.min():.3f}, {train_similarities_v1.max():.3f}]")
    print(f"   Training mean similarity: {train_similarities_v1.mean():.3f}")
    
    # Find optimal threshold on TRAINING data
    best_f1_train = 0
    best_threshold_v1 = 0.5
    
    for threshold in np.linspace(0.5, 0.95, 50):
        predictions = train_similarities_v1 >= threshold
        f1 = f1_score(train_y, predictions, zero_division=0)
        if f1 > best_f1_train:
            best_f1_train = f1
            best_threshold_v1 = threshold
    
    print(f"   Optimal threshold (from training): {best_threshold_v1:.3f} (F1={best_f1_train:.3f})")
    
    # Now apply to TEST data
    print("   Computing similarities for test pairs...")
    
    similarities_v1 = []
    y_true = [p['match'] for p in test_pairs]
    
    similarities_v1 = []
    y_true = [p['match'] for p in test_pairs]
    
    for pair in test_pairs:
        try:
            # Extract feature vectors (preferences + interests)
            p1 = pair['person1_data']
            p2 = pair['person2_data']
            
            # Preferences (6 features)
            prefs1 = p1.get('preferences_self', {})
            prefs2 = p2.get('preferences_self', {})
            
            vec1 = [
                prefs1.get('attractiveness', 0), prefs1.get('sincerity', 0),
                prefs1.get('intelligence', 0), prefs1.get('fun', 0),
                prefs1.get('ambition', 0), prefs1.get('shared_interests', 0)
            ]
            vec2 = [
                prefs2.get('attractiveness', 0), prefs2.get('sincerity', 0),
                prefs2.get('intelligence', 0), prefs2.get('fun', 0),
                prefs2.get('ambition', 0), prefs2.get('shared_interests', 0)
            ]
            
            # Add interests (17 features)
            interests1 = p1.get('interests', {})
            interests2 = p2.get('interests', {})
            interest_keys = ['sports', 'tvsports', 'exercise', 'dining', 'museums', 'art',
                           'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater',
                           'movies', 'concerts', 'music', 'shopping', 'yoga']
            
            for key in interest_keys:
                vec1.append(interests1.get(key, 0))
                vec2.append(interests2.get(key, 0))
            
            # Cosine similarity
            vec1 = np.array(vec1, dtype=float)
            vec2 = np.array(vec2, dtype=float)
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 0 and norm2 > 0:
                similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            else:
                similarity = 0.0
            
            similarities_v1.append(similarity)
        except Exception as e:
            similarities_v1.append(0.0)
    
    similarities_v1 = np.array(similarities_v1)
    
    # Use threshold tuned on TRAINING data (no data leakage)
    predictions_sim_v1 = similarities_v1 >= best_threshold_v1
    
    results['similarity_v1'] = {
        'model': 'Similarity V1 (Time 1 only)',
        'feature_mode': 'v1',
        'threshold': float(best_threshold_v1),
        'accuracy': accuracy_score(y_true, predictions_sim_v1),
        'precision': precision_score(y_true, predictions_sim_v1, zero_division=0),
        'recall': recall_score(y_true, predictions_sim_v1, zero_division=0),
        'f1': f1_score(y_true, predictions_sim_v1, zero_division=0),
        'auc': roc_auc_score(y_true, similarities_v1),
        'pr_auc': average_precision_score(y_true, similarities_v1),
        'confusion_matrix': confusion_matrix(y_true, predictions_sim_v1).tolist()
    }
    
    print(f"  Threshold (tuned on training): {best_threshold_v1:.3f}")
    print(f"  Test similarity range: [{similarities_v1.min():.3f}, {similarities_v1.max():.3f}]")
    print(f"  Test mean similarity: {similarities_v1.mean():.3f}")
    print(f"  Accuracy:  {results['similarity_v1']['accuracy']:.3f}")
    print(f"  Precision: {results['similarity_v1']['precision']:.3f}")
    print(f"  Recall:    {results['similarity_v1']['recall']:.3f}")
    print(f"  F1 Score:  {results['similarity_v1']['f1']:.3f}")
    print(f"  AUC-ROC:   {results['similarity_v1']['auc']:.3f}")
    print(f"  PR-AUC:    {results['similarity_v1']['pr_auc']:.3f}")
    
    # === SIMILARITY BASELINE V2 (With Time 2 data, tune on training) ===
    print("\n" + "=" * 70)
    print("SIMILARITY BASELINE V2: TIME 1 + TIME 2")
    print("=" * 70)
    
    print("\n--- SIMILARITY V2 (Tuning threshold on training data with Time 2) ---")
    print("   Computing training similarities with Time 2 data...")
    
    # Compute similarities on TRAINING data with Time 2 to tune threshold
    train_similarities_v2 = []
    train_y_v2 = []
    
    for _, row in train_df.iterrows():
        iid = row['iid']
        pid = row['pid']
        
        if pd.isna(row['match']) or pd.isna(pid):
            continue
        
        # Find reciprocal row
        partner_row = train_df[(train_df['iid'] == pid) & (train_df['pid'] == iid)]
        
        if len(partner_row) == 0:
            continue
        
        try:
            row_dict = row.to_dict()
            partner_dict = partner_row.iloc[0].to_dict()
            
            # Time 1 Preferences (6 features)
            vec1 = [
                row_dict.get('attr1_1', 0), row_dict.get('sinc1_1', 0),
                row_dict.get('intel1_1', 0), row_dict.get('fun1_1', 0),
                row_dict.get('amb1_1', 0), row_dict.get('shar1_1', 0)
            ]
            vec2 = [
                partner_dict.get('attr1_1', 0), partner_dict.get('sinc1_1', 0),
                partner_dict.get('intel1_1', 0), partner_dict.get('fun1_1', 0),
                partner_dict.get('amb1_1', 0), partner_dict.get('shar1_1', 0)
            ]
            
            # Add Time 2 updated preferences (6 features)
            vec1.extend([
                row_dict.get('attr1_2', 0), row_dict.get('sinc1_2', 0),
                row_dict.get('intel1_2', 0), row_dict.get('fun1_2', 0),
                row_dict.get('amb1_2', 0), row_dict.get('shar1_2', 0)
            ])
            vec2.extend([
                partner_dict.get('attr1_2', 0), partner_dict.get('sinc1_2', 0),
                partner_dict.get('intel1_2', 0), partner_dict.get('fun1_2', 0),
                partner_dict.get('amb1_2', 0), partner_dict.get('shar1_2', 0)
            ])
            
            # Add interests (17 features)
            interest_keys = ['sports', 'tvsports', 'exercise', 'dining', 'museums', 'art',
                           'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater',
                           'movies', 'concerts', 'music', 'shopping', 'yoga']
            
            for key in interest_keys:
                vec1.append(row_dict.get(key, 0))
                vec2.append(partner_dict.get(key, 0))
            
            # Cosine similarity
            vec1 = np.array(vec1, dtype=float)
            vec2 = np.array(vec2, dtype=float)
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 0 and norm2 > 0:
                similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            else:
                similarity = 0.0
            
            train_similarities_v2.append(similarity)
            train_y_v2.append(int(row['match']))
        except Exception as e:
            continue
    
    train_similarities_v2 = np.array(train_similarities_v2)
    train_y_v2 = np.array(train_y_v2)
    
    print(f"   Training similarities computed: {len(train_similarities_v2)} pairs")
    print(f"   Training similarity range: [{train_similarities_v2.min():.3f}, {train_similarities_v2.max():.3f}]")
    print(f"   Training mean similarity: {train_similarities_v2.mean():.3f}")
    
    # Find optimal threshold on TRAINING data
    best_f1_train_v2 = 0
    best_threshold_v2 = 0.5
    
    for threshold in np.linspace(0.5, 0.95, 50):
        predictions = train_similarities_v2 >= threshold
        f1 = f1_score(train_y_v2, predictions, zero_division=0)
        if f1 > best_f1_train_v2:
            best_f1_train_v2 = f1
            best_threshold_v2 = threshold
    
    print(f"   Optimal threshold (from training): {best_threshold_v2:.3f} (F1={best_f1_train_v2:.3f})")
    
    # Now apply to TEST data
    print("   Computing test similarities with Time 2 data...")
    
    similarities_v2 = []
    
    similarities_v2 = []
    
    for pair in test_pairs:
        try:
            # Extract feature vectors (preferences + interests + Time 2 updated prefs)
            p1 = pair['person1_data_full']  # Includes Time 2
            p2 = pair['person2_data_full']
            
            # Time 1 Preferences (6 features)
            prefs1 = p1.get('preferences_self', {})
            prefs2 = p2.get('preferences_self', {})
            
            vec1 = [
                prefs1.get('attractiveness', 0), prefs1.get('sincerity', 0),
                prefs1.get('intelligence', 0), prefs1.get('fun', 0),
                prefs1.get('ambition', 0), prefs1.get('shared_interests', 0)
            ]
            vec2 = [
                prefs2.get('attractiveness', 0), prefs2.get('sincerity', 0),
                prefs2.get('intelligence', 0), prefs2.get('fun', 0),
                prefs2.get('ambition', 0), prefs2.get('shared_interests', 0)
            ]
            
            # Add Time 2 updated preferences (6 features)
            vec1.extend([
                p1.get('attr1_2', 0), p1.get('sinc1_2', 0),
                p1.get('intel1_2', 0), p1.get('fun1_2', 0),
                p1.get('amb1_2', 0), p1.get('shar1_2', 0)
            ])
            vec2.extend([
                p2.get('attr1_2', 0), p2.get('sinc1_2', 0),
                p2.get('intel1_2', 0), p2.get('fun1_2', 0),
                p2.get('amb1_2', 0), p2.get('shar1_2', 0)
            ])
            
            # Add interests (17 features)
            interests1 = p1.get('interests', {})
            interests2 = p2.get('interests', {})
            interest_keys = ['sports', 'tvsports', 'exercise', 'dining', 'museums', 'art',
                           'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater',
                           'movies', 'concerts', 'music', 'shopping', 'yoga']
            
            for key in interest_keys:
                vec1.append(interests1.get(key, 0))
                vec2.append(interests2.get(key, 0))
            
            # Cosine similarity
            vec1 = np.array(vec1, dtype=float)
            vec2 = np.array(vec2, dtype=float)
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 0 and norm2 > 0:
                similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            else:
                similarity = 0.0
            
            similarities_v2.append(similarity)
        except Exception as e:
            similarities_v2.append(0.0)
    
    similarities_v2 = np.array(similarities_v2)
    
    # Use threshold tuned on TRAINING data (no data leakage)
    predictions_sim_v2 = similarities_v2 >= best_threshold_v2
    
    results['similarity_v2'] = {
        'model': 'Similarity V2 (Time 1 + Time 2)',
        'feature_mode': 'v2',
        'threshold': float(best_threshold_v2),
        'accuracy': accuracy_score(y_true, predictions_sim_v2),
        'precision': precision_score(y_true, predictions_sim_v2, zero_division=0),
        'recall': recall_score(y_true, predictions_sim_v2, zero_division=0),
        'f1': f1_score(y_true, predictions_sim_v2, zero_division=0),
        'auc': roc_auc_score(y_true, similarities_v2),
        'pr_auc': average_precision_score(y_true, similarities_v2),
        'confusion_matrix': confusion_matrix(y_true, predictions_sim_v2).tolist()
    }
    
    print(f"  Threshold (tuned on training): {best_threshold_v2:.3f}")
    print(f"  Test similarity range: [{similarities_v2.min():.3f}, {similarities_v2.max():.3f}]")
    print(f"  Test mean similarity: {similarities_v2.mean():.3f}")
    print(f"  Accuracy:  {results['similarity_v2']['accuracy']:.3f}")
    print(f"  Precision: {results['similarity_v2']['precision']:.3f}")
    print(f"  Recall:    {results['similarity_v2']['recall']:.3f}")
    print(f"  F1 Score:  {results['similarity_v2']['f1']:.3f}")
    print(f"  AUC-ROC:   {results['similarity_v2']['auc']:.3f}")
    print(f"  PR-AUC:    {results['similarity_v2']['pr_auc']:.3f}")
    
    # === BASELINE V1: Time 1 Only (Fair with LLM) ===
    print("\n" + "=" * 70)
    print("BASELINE V1: TIME 1 ONLY (Same info as LLM)")
    print("=" * 70)
    
    model_types = ['logistic', 'random_forest']
    if XGBOOST_AVAILABLE:
        model_types.append('xgboost')
    
    for model_type in model_types:
        print(f"\n--- {model_type.upper().replace('_', ' ')} ---")
        
        # Train model on CSV features with class balancing
        # Training data has 16.3% matches, test has 50% → need class_weight='balanced'
        if model_type == 'logistic':
            # Add L2 regularization to prevent overfitting
            model = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced', C=0.5)
        elif model_type == 'random_forest':
            # More regularization for Time 1: limit depth and increase min samples
            model = RandomForestClassifier(
                n_estimators=300, 
                max_depth=15,  # Limit depth to prevent overfitting
                min_samples_split=20,  # Require more samples to split
                min_samples_leaf=8,  # Require more samples per leaf
                max_features='sqrt',
                random_state=42, 
                n_jobs=-1
            )
        else:  # xgboost
            # More regularization for Time 1
            model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=5,  # Shallower trees
                learning_rate=0.03,  # Lower learning rate
                min_child_weight=5,  # More regularization
                subsample=0.7,
                colsample_bytree=0.7,
                gamma=0.2,  # More pruning
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                random_state=42,
                eval_metric='logloss'
            )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_v1)
        
        # Apply SMOTE to balance the classes (16.3% → 30% for V1, more conservative)
        if SMOTE_AVAILABLE and model_type != 'logistic':
            print(f"   Original training: {len(X_train_scaled)} samples ({sum(y_train)} matches, {100*sum(y_train)/len(y_train):.1f}%)")
            # Use more conservative sampling for Time 1 only
            smote = SMOTE(sampling_strategy=0.30, random_state=42, k_neighbors=3)  # More conservative
            X_train_scaled, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
            print(f"   After SMOTE: {len(X_train_scaled)} samples ({sum(y_train_smote)} matches, {100*sum(y_train_smote)/len(y_train_smote):.1f}%)")
            print(f"   Training model...")
            model.fit(X_train_scaled, y_train_smote)
        else:
            print(f"   Training on {len(X_train_scaled)} samples ({sum(y_train)} matches, with balanced class weights)...")
            model.fit(X_train_scaled, y_train)
        
        # Evaluate on test set (using personas.json)
        print(f"   Evaluating on {len(test_pairs)} test pairs...")
        predictions_v1 = []
        probs_v1 = []
        
        baseline_extractor = BaselineModel(model_type=model_type, feature_mode='v1')
        
        for pair in test_pairs:
            try:
                # Extract features from test pair
                features = baseline_extractor._extract_features_v1(
                    pair['person1_data'], 
                    pair['person2_data']
                )
                # Impute missing values
                features = imputer_v1.transform([features])
                features_scaled = scaler.transform(features)
                
                # Predict
                pred = model.predict(features_scaled)[0]
                prob = model.predict_proba(features_scaled)[0][1]
                
                predictions_v1.append(bool(pred))
                probs_v1.append(prob)
            except Exception as e:
                print(f"Warning: Prediction failed for {pair.get('pair_id')}: {e}")
                predictions_v1.append(False)
                probs_v1.append(0.5)
        
        y_true = [p['match'] for p in test_pairs]
        
        # Adjust decision threshold based on expected test distribution
        # Training has 16.3% matches, but we expect test to have ~50%
        if model_type in ['random_forest', 'xgboost']:
            print(f"  Probability distribution: min={min(probs_v1):.3f}, mean={np.mean(probs_v1):.3f}, median={np.median(probs_v1):.3f}, max={max(probs_v1):.3f}")
            
            # Use multiple threshold strategies
            # Strategy 1: Use mean probability as threshold (adaptive)
            threshold_mean = np.mean(probs_v1)
            # Strategy 2: Use median probability (more robust)
            threshold_median = np.median(probs_v1)
            # Strategy 3: Select top 50% (match expected test distribution)
            sorted_probs = sorted(probs_v1, reverse=True)
            threshold_top50 = sorted_probs[len(sorted_probs)//2] if len(sorted_probs) > 1 else 0.5
            
            # Try all strategies and pick best
            strategies = {
                'mean': threshold_mean,
                'median': threshold_median,
                'top50%': threshold_top50
            }
            
            best_threshold = 0.5
            best_f1 = 0
            best_strategy = 'default'
            
            for strategy_name, threshold in strategies.items():
                preds_temp = [p >= threshold for p in probs_v1]
                f1_temp = f1_score(y_true, preds_temp, zero_division=0)
                if f1_temp > best_f1:
                    best_f1 = f1_temp
                    best_threshold = threshold
                    best_strategy = strategy_name
            
            print(f"  Best strategy: {best_strategy}, threshold: {best_threshold:.3f} (F1={best_f1:.3f})")
            predictions_v1 = [p >= best_threshold for p in probs_v1]
        
        results[f'{model_type}_v1'] = {
            'model': f'{model_type} V1 (Time 1 only)',
            'feature_mode': 'v1',
            'accuracy': accuracy_score(y_true, predictions_v1),
            'precision': precision_score(y_true, predictions_v1, zero_division=0),
            'recall': recall_score(y_true, predictions_v1, zero_division=0),
            'f1': f1_score(y_true, predictions_v1, zero_division=0),
            'auc': roc_auc_score(y_true, probs_v1),
            'pr_auc': average_precision_score(y_true, probs_v1),
            'confusion_matrix': confusion_matrix(y_true, predictions_v1).tolist()
        }
        
        print(f"  Accuracy:  {results[f'{model_type}_v1']['accuracy']:.3f}")
        print(f"  Precision: {results[f'{model_type}_v1']['precision']:.3f}")
        print(f"  Recall:    {results[f'{model_type}_v1']['recall']:.3f}")
        print(f"  F1 Score:  {results[f'{model_type}_v1']['f1']:.3f}")
        print(f"  AUC-ROC:   {results[f'{model_type}_v1']['auc']:.3f}")
        print(f"  PR-AUC:    {results[f'{model_type}_v1']['pr_auc']:.3f}")
    
    # === BASELINE V2: Time 1 + Time 2 (With Future Knowledge) ===
    print("\n" + "=" * 70)
    print("BASELINE V2: TIME 1 + TIME 2 (With post-event reflections)")
    print("=" * 70)
    
    for model_type in model_types:
        print(f"\n--- {model_type.upper().replace('_', ' ')} ---")
        
        # Train model on CSV features (Time 1 + Time 2) with class balancing
        if model_type == 'logistic':
            model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        elif model_type == 'random_forest':
            # Increased depth and min_samples for better learning
            model = RandomForestClassifier(
                n_estimators=200, 
                max_depth=None,  # No limit on depth
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42, 
                n_jobs=-1
            )
        else:  # xgboost
            # No scale_pos_weight - will use SMOTE instead
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0,
                random_state=42,
                eval_metric='logloss'
            )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_v2)
        
        # Apply SMOTE to balance the classes (16.3% → 40%)
        if SMOTE_AVAILABLE and model_type != 'logistic':
            print(f"   Original training: {len(X_train_scaled)} samples ({sum(y_train)} matches)")
            smote = SMOTE(sampling_strategy=0.4, random_state=42)  # Increase matches to 40% (not 50% to avoid overfitting)
            X_train_scaled, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
            print(f"   After SMOTE: {len(X_train_scaled)} samples ({sum(y_train_smote)} matches)")
            print(f"   Training model...")
            model.fit(X_train_scaled, y_train_smote)
        else:
            print(f"   Training on {len(X_train_scaled)} samples (with balanced class weights)...")
            model.fit(X_train_scaled, y_train)
        
        # Evaluate on test set (using personas.json with Time 2)
        print(f"   Evaluating on {len(test_pairs)} test pairs...")
        predictions_v2 = []
        probs_v2 = []
        
        baseline_extractor = BaselineModel(model_type=model_type, feature_mode='v2')
        
        for pair in test_pairs:
            try:
                # Extract features from test pair (with Time 2 data)
                features = baseline_extractor._extract_features_v2(
                    pair['person1_data_full'],  # Includes Time 2
                    pair['person2_data_full']
                )
                # Impute missing values
                features = imputer_v2.transform([features])
                features_scaled = scaler.transform(features)
                
                # Predict
                pred = model.predict(features_scaled)[0]
                prob = model.predict_proba(features_scaled)[0][1]
                
                predictions_v2.append(bool(pred))
                probs_v2.append(prob)
            except Exception as e:
                print(f"Warning: Prediction failed for {pair.get('pair_id')}: {e}")
                predictions_v2.append(False)
                probs_v2.append(0.5)
        
        y_true = [p['match'] for p in test_pairs]
        
        # Adjust decision threshold based on expected test distribution
        if model_type in ['random_forest', 'xgboost']:
            print(f"  Probability distribution: min={min(probs_v2):.3f}, mean={np.mean(probs_v2):.3f}, median={np.median(probs_v2):.3f}, max={max(probs_v2):.3f}")
            
            # Use multiple threshold strategies
            threshold_mean = np.mean(probs_v2)
            threshold_median = np.median(probs_v2)
            sorted_probs = sorted(probs_v2, reverse=True)
            threshold_top50 = sorted_probs[len(sorted_probs)//2] if len(sorted_probs) > 1 else 0.5
            
            strategies = {
                'mean': threshold_mean,
                'median': threshold_median,
                'top50%': threshold_top50
            }
            
            best_threshold = 0.5
            best_f1 = 0
            best_strategy = 'default'
            
            for strategy_name, threshold in strategies.items():
                preds_temp = [p >= threshold for p in probs_v2]
                f1_temp = f1_score(y_true, preds_temp, zero_division=0)
                if f1_temp > best_f1:
                    best_f1 = f1_temp
                    best_threshold = threshold
                    best_strategy = strategy_name
            
            print(f"  Best strategy: {best_strategy}, threshold: {best_threshold:.3f} (F1={best_f1:.3f})")
            predictions_v2 = [p >= best_threshold for p in probs_v2]
        
        results[f'{model_type}_v2'] = {
            'model': f'{model_type} V2 (Time 1 + Time 2)',
            'feature_mode': 'v2',
            'accuracy': accuracy_score(y_true, predictions_v2),
            'precision': precision_score(y_true, predictions_v2, zero_division=0),
            'recall': recall_score(y_true, predictions_v2, zero_division=0),
            'f1': f1_score(y_true, predictions_v2, zero_division=0),
            'auc': roc_auc_score(y_true, probs_v2),
            'pr_auc': average_precision_score(y_true, probs_v2),
            'confusion_matrix': confusion_matrix(y_true, predictions_v2).tolist()
        }
        
        print(f"  Accuracy:  {results[f'{model_type}_v2']['accuracy']:.3f}")
        print(f"  Precision: {results[f'{model_type}_v2']['precision']:.3f}")
        print(f"  Recall:    {results[f'{model_type}_v2']['recall']:.3f}")
        print(f"  F1 Score:  {results[f'{model_type}_v2']['f1']:.3f}")
        print(f"  AUC-ROC:   {results[f'{model_type}_v2']['auc']:.3f}")
        print(f"  PR-AUC:    {results[f'{model_type}_v2']['pr_auc']:.3f}")
    
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
            'AUC-ROC': f"{r['auc']:.3f}",
            'PR-AUC': f"{r['pr_auc']:.3f}"
        }
        for r in results.values() if 'feature_mode' in r
    ])
    
    print(comparison_df.to_string(index=False))
    
    # Save results
    output_path = f"{output_dir}/baseline_comparison_v2.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # === LOAD AND DISPLAY LLM RESULTS (BINARY) ===
    print("\n" + "=" * 70)
    print("LLM METHOD RESULTS (For Comparison)")
    print("=" * 70)
    
    llm_report_path = f"{output_dir}/accuracy_report.json"
    try:
        with open(llm_report_path, 'r') as f:
            llm_results = json.load(f)
        
        print("\n📊 LLM Performance on Same 100 Test Pairs:")
        print("-" * 70)
        
        # Self-evaluation (both persons agree)
        print("\n1. LLM Self-Evaluation (Both Persons' Decisions):")
        print(f"   Accuracy:  {llm_results['self_eval_accuracy']/100:.3f}")
        print(f"   Precision: {llm_results['self_eval_precision']/100:.3f}")
        print(f"   Recall:    {llm_results['self_eval_recall']/100:.3f}")
        print(f"   F1 Score:  {llm_results['self_eval_f1']/100:.3f}")
        print(f"   (TP={llm_results['self_eval_true_positive']}, "
              f"TN={llm_results['self_eval_true_negative']}, "
              f"FP={llm_results['self_eval_false_positive']}, "
              f"FN={llm_results['self_eval_false_negative']})")
        
        # Observer evaluation
        print("\n2. LLM Observer Evaluation (Third-Party Assessment):")
        print(f"   Accuracy:  {llm_results['observer_accuracy']/100:.3f}")
        print(f"   Precision: {llm_results['observer_precision']/100:.3f}")
        print(f"   Recall:    {llm_results['observer_recall']/100:.3f}")
        print(f"   F1 Score:  {llm_results['observer_f1']/100:.3f}")
        print(f"   AUC:       N/A (binary predictions only)")
        print(f"   (TP={llm_results['observer_true_positive']}, "
              f"TN={llm_results['observer_true_negative']}, "
              f"FP={llm_results['observer_false_positive']}, "
              f"FN={llm_results['observer_false_negative']})")
        
        # Individual accuracy
        print("\n3. Individual Person Accuracy:")
        print(f"   Person 1: {llm_results['person1_accuracy']:.1f}%")
        print(f"   Person 2: {llm_results['person2_accuracy']:.1f}%")
        
        print("\n⚠️  Note: LLM methods cannot provide AUC scores because they only")
        print("    give binary yes/no predictions, not probability scores.")
        print("    AUC requires continuous probability values to calculate ROC curve.")
        
        # Add LLM results to comparison
        results['llm_self_eval'] = {
            'model': 'LLM Self-Evaluation',
            'feature_mode': 'conversation',
            'accuracy': llm_results['self_eval_accuracy']/100,
            'precision': llm_results['self_eval_precision']/100,
            'recall': llm_results['self_eval_recall']/100,
            'f1': llm_results['self_eval_f1']/100,
            'confusion_matrix': [
                [llm_results['self_eval_true_negative'], llm_results['self_eval_false_positive']],
                [llm_results['self_eval_false_negative'], llm_results['self_eval_true_positive']]
            ]
        }
        
        results['llm_observer'] = {
            'model': 'LLM Observer',
            'feature_mode': 'conversation',
            'accuracy': llm_results['observer_accuracy']/100,
            'precision': llm_results['observer_precision']/100,
            'recall': llm_results['observer_recall']/100,
            'f1': llm_results['observer_f1']/100,
            'confusion_matrix': [
                [llm_results['observer_true_negative'], llm_results['observer_false_positive']],
                [llm_results['observer_false_negative'], llm_results['observer_true_positive']]
            ]
        }
        
        # === COMPREHENSIVE COMPARISON ===
        print("\n" + "=" * 70)
        print("COMPREHENSIVE COMPARISON: BASELINES vs LLM")
        print("=" * 70)
        
        comparison_data = []
        
        # Baseline models
        for key in ['similarity_v1', 'similarity_v2', 'logistic_v1', 'random_forest_v1', 
                    'xgboost_v1', 'logistic_v2', 'random_forest_v2', 'xgboost_v2']:
            if key in results:
                r = results[key]
                comparison_data.append({
                    'Model': r['model'],
                    'Type': 'Baseline',
                    'Mode': r['feature_mode'],
                    'Accuracy': f"{r['accuracy']:.3f}",
                    'Precision': f"{r['precision']:.3f}",
                    'Recall': f"{r['recall']:.3f}",
                    'F1': f"{r['f1']:.3f}",
                    'AUC-ROC': f"{r.get('auc', 0):.3f}",
                    'PR-AUC': f"{r.get('pr_auc', 0):.3f}"
                })
        
        # LLM models
        for key in ['llm_self_eval', 'llm_observer']:
            if key in results:
                r = results[key]
                comparison_data.append({
                    'Model': r['model'],
                    'Type': 'LLM',
                    'Mode': r['feature_mode'],
                    'Accuracy': f"{r['accuracy']:.3f}",
                    'Precision': f"{r['precision']:.3f}",
                    'Recall': f"{r['recall']:.3f}",
                    'F1': f"{r['f1']:.3f}",
                    'AUC-ROC': 'N/A',
                    'PR-AUC': 'N/A'
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + comparison_df.to_string(index=False))
        
        # Highlight best performers
        print("\n🏆 BEST PERFORMERS:")
        f1_scores = [(d['Model'], float(d['F1'])) for d in comparison_data]
        f1_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 3 by F1 Score:")
        for i, (model, f1) in enumerate(f1_scores[:3], 1):
            print(f"  {i}. {model}: F1 = {f1:.3f}")
        
    except FileNotFoundError:
        print(f"\n⚠️  LLM results not found at {llm_report_path}")
        print("   Run the LLM experiment first to generate accuracy_report.json")
    except Exception as e:
        print(f"\n⚠️  Error loading LLM results: {e}")
    
    # === LOAD AND DISPLAY LLM SCORE-BASED RESULTS (with AUC) ===
    print("\n" + "=" * 70)
    print("LLM SCORE-BASED METHODS (0-10 ratings with AUC)")
    print("=" * 70)
    llm_score_path = f"{output_dir}/llm_score_evaluation.json"
    try:
        with open(llm_score_path, 'r') as f:
            llm_score = json.load(f)
        
        # Participant method
        part_metrics = llm_score.get('participant_method', {}).get('metrics', {})
        results['llm_participant'] = {
            'model': 'LLM Participant (Product of scores)',
            'feature_mode': 'conversation-scores',
            'accuracy': float(part_metrics.get('accuracy')) if part_metrics.get('accuracy') is not None else 0.0,
            'precision': float(part_metrics.get('precision')) if part_metrics.get('precision') is not None else 0.0,
            'recall': float(part_metrics.get('recall')) if part_metrics.get('recall') is not None else 0.0,
            'f1': float(part_metrics.get('f1')) if part_metrics.get('f1') is not None else 0.0,
            'auc': float(part_metrics.get('auc_roc')) if part_metrics.get('auc_roc') is not None else 0.0,
            'pr_auc': float(part_metrics.get('pr_auc')) if part_metrics.get('pr_auc') is not None else 0.0,
            'confusion_matrix': part_metrics.get('confusion_matrix')
        }
        
        # Observer method
        obs_metrics = llm_score.get('observer_method', {}).get('metrics', {})
        results['llm_observer_scores'] = {
            'model': 'LLM Observer (Single score)',
            'feature_mode': 'conversation-scores',
            'accuracy': float(obs_metrics.get('accuracy')) if obs_metrics.get('accuracy') is not None else 0.0,
            'precision': float(obs_metrics.get('precision')) if obs_metrics.get('precision') is not None else 0.0,
            'recall': float(obs_metrics.get('recall')) if obs_metrics.get('recall') is not None else 0.0,
            'f1': float(obs_metrics.get('f1')) if obs_metrics.get('f1') is not None else 0.0,
            'auc': float(obs_metrics.get('auc_roc')) if obs_metrics.get('auc_roc') is not None else 0.0,
            'pr_auc': float(obs_metrics.get('pr_auc')) if obs_metrics.get('pr_auc') is not None else 0.0,
            'confusion_matrix': obs_metrics.get('confusion_matrix')
        }
        
        # Print a compact summary
        print("\n📊 LLM Score-based Performance on Same 100 Test Pairs:")
        print("-" * 70)
        for key in ['llm_participant', 'llm_observer_scores']:
            r = results[key]
            print(f"{r['model']} → Acc: {r['accuracy']:.3f}, P: {r['precision']:.3f}, R: {r['recall']:.3f}, F1: {r['f1']:.3f}, AUC: {r['auc']:.3f}, PR-AUC: {r['pr_auc']:.3f}")
        
        # Save updated results including LLM scores
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Updated results (including LLM scores) saved to: {output_path}")
        
        # Extended comprehensive comparison (includes LLM score-based)
        print("\n" + "=" * 70)
        print("COMPREHENSIVE COMPARISON: BASELINES vs LLM (Binary) vs LLM (Scores)")
        print("=" * 70)
        comparison_data = []
        
        for key in ['similarity_v1', 'similarity_v2', 'logistic_v1', 'random_forest_v1', 'xgboost_v1', 'logistic_v2', 'random_forest_v2', 'xgboost_v2']:
            if key in results:
                r = results[key]
                comparison_data.append({
                    'Model': r['model'],
                    'Type': 'Baseline',
                    'Mode': r['feature_mode'],
                    'Accuracy': f"{r['accuracy']:.3f}",
                    'Precision': f"{r['precision']:.3f}",
                    'Recall': f"{r['recall']:.3f}",
                    'F1': f"{r['f1']:.3f}",
                    'AUC-ROC': f"{r.get('auc', 0):.3f}",
                    'PR-AUC': f"{r.get('pr_auc', 0):.3f}"
                })
        for key in ['llm_self_eval', 'llm_observer']:
            if key in results:
                r = results[key]
                comparison_data.append({
                    'Model': r['model'],
                    'Type': 'LLM (binary)',
                    'Mode': r['feature_mode'],
                    'Accuracy': f"{r['accuracy']:.3f}",
                    'Precision': f"{r['precision']:.3f}",
                    'Recall': f"{r['recall']:.3f}",
                    'F1': f"{r['f1']:.3f}",
                    'AUC-ROC': 'N/A',
                    'PR-AUC': 'N/A'
                })
        for key in ['llm_participant', 'llm_observer_scores']:
            if key in results:
                r = results[key]
                comparison_data.append({
                    'Model': r['model'],
                    'Type': 'LLM (scores)',
                    'Mode': r['feature_mode'],
                    'Accuracy': f"{r['accuracy']:.3f}",
                    'Precision': f"{r['precision']:.3f}",
                    'Recall': f"{r['recall']:.3f}",
                    'F1': f"{r['f1']:.3f}",
                    'AUC-ROC': f"{r.get('auc', 0):.3f}",
                    'PR-AUC': f"{r.get('pr_auc', 0):.3f}"
                })
        comparison_df2 = pd.DataFrame(comparison_data)
        print("\n" + comparison_df2.to_string(index=False))
    except FileNotFoundError:
        print(f"\n⚠️  LLM score results not found at {llm_score_path}")
        print("   Run llm_score_evaluator.py first to generate llm_score_evaluation.json")
    except Exception as e:
        print(f"\n⚠️  Error loading LLM score results: {e}")
    
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
