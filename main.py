"""
Currency Fraud Detection using Classical Computer Vision and ML
Author: Zuhaib Iqbal
Description: End-to-end pipeline for feature extraction and ensemble ML classification
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import pickle
import warnings
warnings.filterwarnings("ignore")


# -------------------- PREPROCESSING -------------------- #
class ImagePreprocessor:
    def __init__(self):
        self.blur_kernel = (5, 5)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image not found: {img_path}")

        img = cv2.resize(img, (256, 256))
        img_blur = cv2.GaussianBlur(img, self.blur_kernel, 1.0)
        img_eq = self.clahe.apply(img_blur)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        img_morph = cv2.morphologyEx(img_eq, cv2.MORPH_CLOSE, kernel)

        edges = cv2.Canny(img_morph, 100, 200)
        laplacian = cv2.Laplacian(img_morph, cv2.CV_64F)
        laplacian = np.uint8(np.abs(laplacian))

        return img_morph, edges, laplacian


# -------------------- FEATURE EXTRACTION -------------------- #
class FeatureExtractor:
    @staticmethod
    def entropy(img):
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
        hist = hist / np.sum(hist)
        return -np.sum(hist * np.log2(hist + 1e-10))

    @staticmethod
    def extract_features(img, edges, laplacian):
        features = []

        # Statistical features
        features += [
            np.mean(img), np.std(img), np.min(img), np.max(img),
            np.median(img), np.max(img) - np.min(img),
            FeatureExtractor.entropy(img)
        ]

        # Edge features
        features += [
            np.sum(edges > 0) / edges.size,
            np.mean(edges),
            np.mean(np.abs(laplacian))
        ]

        # Contour-based features
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        features.append(len(contours))
        features.append(sum(cv2.contourArea(c) for c in contours))

        # Frequency domain features
        f = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f)
        mag = np.abs(f_shift)
        features += [np.mean(mag), np.std(mag), np.max(mag)]

        return np.array(features)


# -------------------- MODEL -------------------- #
class CurrencyDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = self._build_model()

    def _build_model(self):
        xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

        return VotingClassifier(
            estimators=[("xgb", xgb), ("svm", svm), ("rf", rf)],
            voting="soft"
        )

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X_scaled, y, cv=kfold, scoring="accuracy")
        print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

        self.model.fit(X_scaled, y)

    def predict(self, feature_vector):
        feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))
        pred = self.model.predict(feature_vector)[0]
        prob = self.model.predict_proba(feature_vector)[0].max()

        return pred, prob

    def save(self, path="model.pkl"):
        with open(path, "wb") as f:
            pickle.dump((self.model, self.scaler), f)


# -------------------- DATA LOADING -------------------- #
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    return df["image_path"].values, df["label"].values


# -------------------- MAIN -------------------- #
if __name__ == "__main__":
    print("Currency Fraud Detection System")
    print("Extracting features...")

    preprocessor = ImagePreprocessor()
    extractor = FeatureExtractor()
    detector = CurrencyDetector()

    # Load dataset CSV: image_path,label (1 = genuine, 0 = counterfeit)
    image_paths, labels = load_dataset("dataset.csv")

    feature_list = []
    for path in image_paths:
        img, edges, lap = preprocessor.preprocess(path)
        features = extractor.extract_features(img, edges, lap)
        feature_list.append(features)

    X = np.array(feature_list)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    detector.train(X_train, y_train)

    detector.save()
    print("Model saved as model.pkl")
