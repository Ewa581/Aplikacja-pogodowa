import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

class AnomalyDetector:
    @staticmethod
    def detect_with_zscore(data, features, threshold=2.0):
        """Statistical Z-Score method"""
        scaler = StandardScaler()
        scores = np.abs(scaler.fit_transform(data[features]))
        anomalies = np.any(scores > threshold, axis=1)
        return anomalies

    @staticmethod
    def detect_with_isolation_forest(data, features, contamination=0.05):
        """Isolation Forest implementation"""
        clf = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        preds = clf.fit_predict(data[features])
        return preds == -1

    @staticmethod
    def detect_with_ocsvm(data, features, nu=0.05):
        """One-Class SVM implementation"""
        scaler = StandardScaler()
        X = scaler.fit_transform(data[features])
        clf = OneClassSVM(nu=nu, kernel="rbf", gamma='scale')
        preds = clf.fit_predict(X)
        return preds == -1

    @staticmethod
    def detect_with_lof(data, features, n_neighbors=20):
        """Local Outlier Factor implementation"""
        scaler = StandardScaler()
        X = scaler.fit_transform(data[features])
        clf = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=False,
            contamination='auto'
        )
        preds = clf.fit_predict(X)
        return preds == -1

    @staticmethod
    def detect_with_dbscan(data, features, eps=0.5, min_samples=5):
        """DBSCAN implementation"""
        scaler = StandardScaler()
        X = scaler.fit_transform(data[features])
        clusters = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='euclidean'
        ).fit_predict(X)
        return clusters == -1

class AnomalyEnsemble:
    @staticmethod
    def ensemble_detect(data, features, models_config, voting='soft'):
        """
        Combine multiple models' predictions
        
        Parameters:
        - data: Input DataFrame
        - features: List of features to use
        - models_config: Dictionary of {model_name: params}
        - voting: 'hard' (majority vote) or 'soft' (average probabilities)
        
        Returns:
        - ensemble_anomalies: Boolean array of anomalies
        - model_votes: DataFrame showing each model's prediction
        """
        predictions = {}
        detector = AnomalyDetector()
        
        for model_name, params in models_config.items():
            if model_name == "Z-Score":
                pred = detector.detect_with_zscore(data, features, params.get('threshold', 2.0))
            elif model_name == "Isolation Forest":
                pred = detector.detect_with_isolation_forest(
                    data, features, params.get('contamination', 0.05))
            elif model_name == "One-Class SVM":
                pred = detector.detect_with_ocsvm(
                    data, features, params.get('nu', 0.05))
            elif model_name == "LOF":
                pred = detector.detect_with_lof(
                    data, features, params.get('n_neighbors', 20))
            elif model_name == "DBSCAN":
                pred = detector.detect_with_dbscan(
                    data, features, 
                    params.get('eps', 0.5), 
                    params.get('min_samples', 5))
            
            predictions[model_name] = pred.astype(int)
        
        model_votes = pd.DataFrame(predictions)
        
        if voting == 'hard':
            ensemble_anomalies = (model_votes.mean(axis=1) > 0.5)
        elif voting == 'soft':
            ensemble_anomalies = (model_votes.sum(axis=1) >= 1)
        else:
            raise ValueError("Voting must be 'hard' or 'soft'")
        
        return ensemble_anomalies, model_votes