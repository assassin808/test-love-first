"""
Baseline Matching Algorithms for Speed Dating Prediction

Implements traditional ML baselines to compare against LLM-based matching:
1. Similarity Baseline: Cosine similarity on preference/interest vectors (dummy baseline)
2. Logistic Regression: Simple interpretable model
3. Random Forest: Ensemble tree-based model
4. XGBoost: Gradient boosting (SOTA for tabular data)

Based on research: "Predicting Speed Dating" - Tilburg University thesis
"""

import json
import numpy as np
import pandas as pd
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
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class SimilarityBaseline:
    """
    Dummy baseline using cosine similarity on preference vectors.
    
    Features used:
    - Self preferences (attr1_1, sinc1_1, intel1_1, fun1_1, amb1_1, shar1_1)
    - Partner ratings (attr3_1, sinc3_1, intel3_1, fun3_1, amb3_1)
    - Interests (sports, tvsports, exercise, dining, museums, art, hiking, 
                gaming, clubbing, reading, tv, theater, movies, concerts, music)
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.feature_names = [
            # Self preferences (what I value)
            'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1',
            # Self ratings (how I rate myself)
            'attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1',
            # Interests
            'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art',
            'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater',
            'movies', 'concerts', 'music'
        ]
    
    def _extract_features(self, person_data: Dict) -> np.ndarray:
        """Extract feature vector from person data."""
        features = []
        for feat in self.feature_names:
            # Handle nested structure (some features might be in different levels)
            if feat in person_data:
                val = person_data[feat]
            elif 'preferences' in person_data and feat in person_data['preferences']:
                val = person_data['preferences'][feat]
            else:
                val = 0  # Default to 0 if missing
            
            # Convert to float, handle None/NaN
            try:
                features.append(float(val) if val is not None else 0.0)
            except (ValueError, TypeError):
                features.append(0.0)
        
        return np.array(features)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        # Avoid division by zero
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def predict(self, person1_data: Dict, person2_data: Dict) -> Tuple[bool, float]:
        """
        Predict if two people would match based on feature similarity.
        
        Returns:
            (prediction, similarity_score)
        """
        vec1 = self._extract_features(person1_data)
        vec2 = self._extract_features(person2_data)
        
        similarity = self._cosine_similarity(vec1, vec2)
        prediction = similarity >= self.threshold
        
        return prediction, similarity


class MLBaseline:
    """
    Machine Learning baseline using traditional classifiers.
    
    Features engineered:
    - Person 1 preferences and self-ratings
    - Person 2 preferences and self-ratings
    - Interest overlap (Jaccard similarity on interests)
    - Preference alignment (correlation between what P1 values and P2's self-ratings)
    """
    
    def __init__(self, model_type: str = 'logistic'):
        """
        Args:
            model_type: 'logistic', 'random_forest', or 'xgboost'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Initialize model
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
                raise ValueError("XGBoost not available. Install with: brew install libomp && pip install xgboost")
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def _extract_features(self, person1_data: Dict, person2_data: Dict) -> np.ndarray:
        """
        Extract feature vector for a pair.
        
        Features:
        - P1 preferences (6 features)
        - P1 self-ratings (5 features)
        - P2 preferences (6 features)
        - P2 self-ratings (5 features)
        - Interest overlap (1 feature)
        - Preference-rating alignment (1 feature)
        Total: 24 features
        """
        features = []
        
        # Person 1 preferences
        p1_prefs = ['attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']
        for feat in p1_prefs:
            val = person1_data.get(feat, 0)
            features.append(float(val) if val is not None else 0.0)
        
        # Person 1 self-ratings
        p1_ratings = ['attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1']
        for feat in p1_ratings:
            val = person1_data.get(feat, 0)
            features.append(float(val) if val is not None else 0.0)
        
        # Person 2 preferences
        for feat in p1_prefs:
            val = person2_data.get(feat, 0)
            features.append(float(val) if val is not None else 0.0)
        
        # Person 2 self-ratings
        for feat in p1_ratings:
            val = person2_data.get(feat, 0)
            features.append(float(val) if val is not None else 0.0)
        
        # Interest overlap (Jaccard similarity)
        interest_fields = ['sports', 'tvsports', 'exercise', 'dining', 'museums', 
                          'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 
                          'theater', 'movies', 'concerts', 'music']
        
        p1_interests = set()
        p2_interests = set()
        for interest in interest_fields:
            if person1_data.get(interest, 0):
                p1_interests.add(interest)
            if person2_data.get(interest, 0):
                p2_interests.add(interest)
        
        if len(p1_interests) == 0 and len(p2_interests) == 0:
            interest_overlap = 0.0
        else:
            intersection = len(p1_interests & p2_interests)
            union = len(p1_interests | p2_interests)
            interest_overlap = intersection / union if union > 0 else 0.0
        
        features.append(interest_overlap)
        
        # Preference-rating alignment
        # Correlation between what P1 values and how P2 rates themselves
        p1_pref_vals = [float(person1_data.get(f, 0) or 0) for f in p1_prefs[:5]]
        p2_rating_vals = [float(person2_data.get(f, 0) or 0) for f in p1_ratings]
        
        if sum(p1_pref_vals) > 0 and sum(p2_rating_vals) > 0:
            alignment = np.corrcoef(p1_pref_vals, p2_rating_vals)[0, 1]
            if np.isnan(alignment):
                alignment = 0.0
        else:
            alignment = 0.0
        
        features.append(alignment)
        
        return np.array(features)
    
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
            features = self._extract_features(
                sample['person1_data'],
                sample['person2_data']
            )
            X.append(features)
            y.append(1 if sample['match'] else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        if verbose:
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
            print(f"\n{self.model_type.upper()} Training Complete:")
            print(f"  Training samples: {len(y)}")
            print(f"  Positive class: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
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


def load_speed_dating_data(personas_path: str = 'results/personas.json') -> pd.DataFrame:
    """
    Load persona data and prepare for baseline training.
    
    Returns:
        DataFrame with columns: iid, gender, all feature columns
    """
    with open(personas_path, 'r') as f:
        personas_pairs = json.load(f)
    
    # Extract features from personas (handle nested pair structure)
    data = []
    for pair in personas_pairs:
        # Each pair has person1 and person2
        for person_key in ['person1', 'person2']:
            if person_key in pair:
                person = pair[person_key]
                
                row = {
                    'iid': person['iid'],
                    'gender': person['gender'],
                    'age': person.get('age', 0),
                    'pair_id': pair.get('pair_id')
                }
                
                # Add time2_reflection data if available
                time2_data = person.get('time2_reflection', {})
                
                # Updated preferences (self)
                prefs_self = time2_data.get('updated_preferences_self', {})
                row['attr1_1'] = prefs_self.get('attractiveness', 0)
                row['sinc1_1'] = prefs_self.get('sincerity', 0)
                row['intel1_1'] = prefs_self.get('intelligence', 0)
                row['fun1_1'] = prefs_self.get('fun', 0)
                row['amb1_1'] = prefs_self.get('ambition', 0)
                row['shar1_1'] = prefs_self.get('shared_interests', 0)
                
                # Self ratings
                self_ratings = time2_data.get('updated_self_ratings', {})
                row['attr3_1'] = self_ratings.get('attractiveness', 0)
                row['sinc3_1'] = self_ratings.get('sincerity', 0)
                row['intel3_1'] = self_ratings.get('intelligence', 0)
                row['fun3_1'] = self_ratings.get('fun', 0)
                row['amb3_1'] = self_ratings.get('ambition', 0)
                
                # Add interests (extract from persona_narrative if available)
                # For now, use dummy values - ideally parse from narrative
                for interest in ['sports', 'tvsports', 'exercise', 'dining', 'museums', 
                               'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 
                               'theater', 'movies', 'concerts', 'music']:
                    row[interest] = 0  # Would need to parse from narrative
                
                # Decision data (if available)
                decision_data = person.get('decision', {})
                row['dec'] = decision_data.get('dec', 0)
                
                data.append(row)
    
    return pd.DataFrame(data)


def evaluate_baselines(
    data_path: str = 'results/personas.json',
    output_path: str = 'results/baseline_comparison.json'
):
    """
    Evaluate all baseline models and compare with LLM predictions.
    
    Trains on 70% of pairs, tests on 30%.
    """
    print("=" * 60)
    print("BASELINE MATCHING ALGORITHMS EVALUATION")
    print("=" * 60)
    
    # Load persona data
    print("\nLoading persona data...")
    df = load_speed_dating_data(data_path)
    print(f"Loaded {len(df)} personas")
    
    # Load ground truth matches
    with open(data_path, 'r') as f:
        personas_pairs = json.load(f)
    
    # Create pair dataset
    print("\nCreating pair dataset...")
    pairs = []
    for pair in personas_pairs:
        if 'person1' in pair and 'person2' in pair:
            person1 = pair['person1']
            person2 = pair['person2']
            
            # Extract feature data for baseline models
            person1_data = {}
            person2_data = {}
            
            # Person 1 features
            time2_p1 = person1.get('time2_reflection', {})
            prefs_p1 = time2_p1.get('updated_preferences_self', {})
            ratings_p1 = time2_p1.get('updated_self_ratings', {})
            
            person1_data.update({
                'attr1_1': prefs_p1.get('attractiveness', 0),
                'sinc1_1': prefs_p1.get('sincerity', 0),
                'intel1_1': prefs_p1.get('intelligence', 0),
                'fun1_1': prefs_p1.get('fun', 0),
                'amb1_1': prefs_p1.get('ambition', 0),
                'shar1_1': prefs_p1.get('shared_interests', 0),
                'attr3_1': ratings_p1.get('attractiveness', 0),
                'sinc3_1': ratings_p1.get('sincerity', 0),
                'intel3_1': ratings_p1.get('intelligence', 0),
                'fun3_1': ratings_p1.get('fun', 0),
                'amb3_1': ratings_p1.get('ambition', 0),
            })
            
            # Person 2 features
            time2_p2 = person2.get('time2_reflection', {})
            prefs_p2 = time2_p2.get('updated_preferences_self', {})
            ratings_p2 = time2_p2.get('updated_self_ratings', {})
            
            person2_data.update({
                'attr1_1': prefs_p2.get('attractiveness', 0),
                'sinc1_1': prefs_p2.get('sincerity', 0),
                'intel1_1': prefs_p2.get('intelligence', 0),
                'fun1_1': prefs_p2.get('fun', 0),
                'amb1_1': prefs_p2.get('ambition', 0),
                'shar1_1': prefs_p2.get('shared_interests', 0),
                'attr3_1': ratings_p2.get('attractiveness', 0),
                'sinc3_1': ratings_p2.get('sincerity', 0),
                'intel3_1': ratings_p2.get('intelligence', 0),
                'fun3_1': ratings_p2.get('fun', 0),
                'amb3_1': ratings_p2.get('ambition', 0),
            })
            
            # Get ground truth from ground_truth field
            ground_truth = pair.get('ground_truth', {})
            match = ground_truth.get('match', 0)
            
            # Convert to boolean
            match = bool(match == 1 or match == True)
            
            pairs.append({
                'pair_id': pair.get('pair_id', f"pair_{len(pairs)}"),
                'person1_iid': person1.get('iid'),
                'person2_iid': person2.get('iid'),
                'person1_data': person1_data,
                'person2_data': person2_data,
                'match': bool(match)
            })
    
    # Remove duplicates (each pair appears twice)
    unique_pairs = []
    seen_pairs = set()
    for pair in pairs:
        pair_key = tuple(sorted([pair['person1_iid'], pair['person2_iid']]))
        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            unique_pairs.append(pair)
    
    print(f"Created {len(unique_pairs)} unique pairs")
    print(f"  Matches: {sum(p['match'] for p in unique_pairs)} ({sum(p['match'] for p in unique_pairs)/len(unique_pairs)*100:.1f}%)")
    
    # Split into train/test
    train_pairs, test_pairs = train_test_split(
        unique_pairs, 
        test_size=0.3, 
        random_state=42,
        stratify=[p['match'] for p in unique_pairs]
    )
    
    print(f"\nTrain: {len(train_pairs)} pairs")
    print(f"Test: {len(test_pairs)} pairs")
    
    # Evaluate baselines
    results = {}
    
    # 1. Similarity Baseline
    print("\n" + "=" * 60)
    print("1. SIMILARITY BASELINE (Cosine Similarity)")
    print("=" * 60)
    
    # Tune threshold on train set
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.3, 0.8, 0.05):
        sim_model = SimilarityBaseline(threshold=threshold)
        predictions = []
        
        for pair in train_pairs:
            pred, score = sim_model.predict(pair['person1_data'], pair['person2_data'])
            predictions.append(pred)
        
        f1 = f1_score([p['match'] for p in train_pairs], predictions)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Best threshold (tuned on train): {best_threshold:.2f}")
    
    # Test with best threshold
    sim_model = SimilarityBaseline(threshold=best_threshold)
    sim_predictions = []
    sim_scores = []
    
    for pair in test_pairs:
        pred, score = sim_model.predict(pair['person1_data'], pair['person2_data'])
        sim_predictions.append(pred)
        sim_scores.append(score)
    
    y_true = [p['match'] for p in test_pairs]
    
    sim_results = {
        'model': 'Similarity Baseline',
        'accuracy': accuracy_score(y_true, sim_predictions),
        'precision': precision_score(y_true, sim_predictions),
        'recall': recall_score(y_true, sim_predictions),
        'f1': f1_score(y_true, sim_predictions),
        'threshold': best_threshold,
        'confusion_matrix': confusion_matrix(y_true, sim_predictions).tolist()
    }
    
    results['similarity'] = sim_results
    
    print(f"Accuracy:  {sim_results['accuracy']:.3f}")
    print(f"Precision: {sim_results['precision']:.3f}")
    print(f"Recall:    {sim_results['recall']:.3f}")
    print(f"F1 Score:  {sim_results['f1']:.3f}")
    
    # 2. Logistic Regression
    print("\n" + "=" * 60)
    print("2. LOGISTIC REGRESSION")
    print("=" * 60)
    
    lr_model = MLBaseline(model_type='logistic')
    lr_model.train(train_pairs)
    
    lr_predictions = []
    lr_probs = []
    
    for pair in test_pairs:
        pred, prob = lr_model.predict(pair['person1_data'], pair['person2_data'])
        lr_predictions.append(pred)
        lr_probs.append(prob)
    
    lr_results = {
        'model': 'Logistic Regression',
        'accuracy': accuracy_score(y_true, lr_predictions),
        'precision': precision_score(y_true, lr_predictions),
        'recall': recall_score(y_true, lr_predictions),
        'f1': f1_score(y_true, lr_predictions),
        'auc': roc_auc_score(y_true, lr_probs),
        'confusion_matrix': confusion_matrix(y_true, lr_predictions).tolist()
    }
    
    results['logistic_regression'] = lr_results
    
    print(f"Accuracy:  {lr_results['accuracy']:.3f}")
    print(f"Precision: {lr_results['precision']:.3f}")
    print(f"Recall:    {lr_results['recall']:.3f}")
    print(f"F1 Score:  {lr_results['f1']:.3f}")
    print(f"AUC:       {lr_results['auc']:.3f}")
    
    # 3. Random Forest
    print("\n" + "=" * 60)
    print("3. RANDOM FOREST")
    print("=" * 60)
    
    rf_model = MLBaseline(model_type='random_forest')
    rf_model.train(train_pairs)
    
    rf_predictions = []
    rf_probs = []
    
    for pair in test_pairs:
        pred, prob = rf_model.predict(pair['person1_data'], pair['person2_data'])
        rf_predictions.append(pred)
        rf_probs.append(prob)
    
    rf_results = {
        'model': 'Random Forest',
        'accuracy': accuracy_score(y_true, rf_predictions),
        'precision': precision_score(y_true, rf_predictions),
        'recall': recall_score(y_true, rf_predictions),
        'f1': f1_score(y_true, rf_predictions),
        'auc': roc_auc_score(y_true, rf_probs),
        'confusion_matrix': confusion_matrix(y_true, rf_predictions).tolist()
    }
    
    results['random_forest'] = rf_results
    
    print(f"Accuracy:  {rf_results['accuracy']:.3f}")
    print(f"Precision: {rf_results['precision']:.3f}")
    print(f"Recall:    {rf_results['recall']:.3f}")
    print(f"F1 Score:  {rf_results['f1']:.3f}")
    print(f"AUC:       {rf_results['auc']:.3f}")
    
    # 4. XGBoost (optional)
    if XGBOOST_AVAILABLE:
        print("\n" + "=" * 60)
        print("4. XGBOOST")
        print("=" * 60)
        
        try:
            xgb_model = MLBaseline(model_type='xgboost')
            xgb_model.train(train_pairs)
            
            xgb_predictions = []
            xgb_probs = []
            
            for pair in test_pairs:
                pred, prob = xgb_model.predict(pair['person1_data'], pair['person2_data'])
                xgb_predictions.append(pred)
                xgb_probs.append(prob)
            
            xgb_results = {
                'model': 'XGBoost',
                'accuracy': accuracy_score(y_true, xgb_predictions),
                'precision': precision_score(y_true, xgb_predictions),
                'recall': recall_score(y_true, xgb_predictions),
                'f1': f1_score(y_true, xgb_predictions),
                'auc': roc_auc_score(y_true, xgb_probs),
                'confusion_matrix': confusion_matrix(y_true, xgb_predictions).tolist()
            }
            
            results['xgboost'] = xgb_results
            
            print(f"Accuracy:  {xgb_results['accuracy']:.3f}")
            print(f"Precision: {xgb_results['precision']:.3f}")
            print(f"Recall:    {xgb_results['recall']:.3f}")
            print(f"F1 Score:  {xgb_results['f1']:.3f}")
            print(f"AUC:       {xgb_results['auc']:.3f}")
        except Exception as e:
            print(f"XGBoost training failed: {e}")
    else:
        print("\n" + "=" * 60)
        print("4. XGBOOST - SKIPPED (not available)")
        print("=" * 60)
        print("To install XGBoost:")
        print("  1. brew install libomp")
        print("  2. pip install xgboost")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    
    comparison_df = pd.DataFrame([
        {
            'Model': r['model'],
            'Accuracy': f"{r['accuracy']:.3f}",
            'Precision': f"{r['precision']:.3f}",
            'Recall': f"{r['recall']:.3f}",
            'F1': f"{r['f1']:.3f}",
            'AUC': f"{r.get('auc', 0):.3f}" if 'auc' in r else 'N/A'
        }
        for r in results.values()
    ])
    
    print(comparison_df.to_string(index=False))
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"\nBest Model: {best_model[1]['model']} (F1 = {best_model[1]['f1']:.3f})")
    
    # Save results
    results['summary'] = {
        'train_size': len(train_pairs),
        'test_size': len(test_pairs),
        'match_rate': sum(p['match'] for p in unique_pairs) / len(unique_pairs),
        'best_model': best_model[1]['model'],
        'best_f1': best_model[1]['f1']
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == '__main__':
    import sys
    import os
    
    # Change to project root
    if os.path.basename(os.getcwd()) == 'experiments':
        os.chdir('..')
    
    # Check if results/personas.json exists
    if not os.path.exists('results/personas.json'):
        print("\n❌ ERROR: results/personas.json not found!")
        print("\nThis script needs persona data to run.")
        print("\nPossible solutions:")
        print("  1. Run from the correct directory (where results/personas.json exists)")
        print("  2. Generate personas first:")
        print("     cd experiments && python persona_generator.py")
        print("  3. Copy personas from another directory:")
        print("     mkdir -p results")
        print("     cp /path/to/personas.json results/")
        print("\nCurrent directory:", os.getcwd())
        print("Files in current directory:", os.listdir('.'))
        sys.exit(1)
    
    results = evaluate_baselines()
