from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

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