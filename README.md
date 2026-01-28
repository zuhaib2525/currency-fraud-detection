Counterfeit Currency Detection using Computer Vision and Machine Learning
Author: Zuhaib Iqbal
Tech Stack: Python, OpenCV, scikit-learn, XGBoost
Dataset: 2,000+ currency images (Kaggle + manually collected samples)

 Project Overview:-
This project explores how classical computer vision and machine learning techniques can be used to detect counterfeit currency notes. The goal was to build a complete pipeline starting from image preprocessing and feature extraction to model training and evaluation.
This project was done as a learning experiment to understand feature engineering, ensemble models, and practical ML pipelines.

🔍 Problem Statement
Counterfeit currency detection is usually done using expensive hardware. This project experiments with a software-based approach using image processing and machine learning to classify currency images as genuine or counterfeit.

🧠 Approach
1. Image Preprocessing
The images are processed using OpenCV to reduce noise and enhance features:
Converted to grayscale
Gaussian blur for noise reduction
Histogram equalization (CLAHE)
Morphological operations (erosion and dilation)
Edge detection using Canny
2. Feature Extraction
More than 25 features were extracted from each image:
Statistical Features
Mean, standard deviation, entropy, min/max values
Texture Features
Haralick features
Local Binary Patterns (LBP)
Edge & Shape Features
Edge density
Contour count
Morphological shape features
Frequency Domain
FFT and DCT statistics
3. Machine Learning Models
An ensemble approach was used:
XGBoost
Support Vector Machine (RBF kernel)
Random Forest
A soft voting classifier combines predictions from all three models.
📊 Results
Metric	Value
Accuracy	96.2%
Precision	94.8%
Recall	95.1%
F1 Score	94.9%
ROC-AUC	0.989
Dataset Size: 2,000+ images
Cross Validation: 5-fold CV
Confusion Matrix (Test Set)
Actual / Predicted	Genuine	Counterfeit
Genuine	189	9
Counterfeit	8	194
⚙️ Project Structure
currency-fraud-detection/
│
├── data/                 # Dataset (not included)
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   └── utils.py
│
├── train.py               # Training script
├── predict.py             # Single image inference
├── evaluate.py            # Model evaluation
├── config.yaml
├── requirements.txt
└── README.md
🚀 How to Run
1. Clone the Repository
git clone https://github.com/zuhaib2525/currency-fraud-detection.git
cd currency-fraud-detection
2. Install Dependencies
pip install -r requirements.txt
3. Train the Model
python train.py --data_path ./data/images --test_size 0.2 --cv_folds 5
4. Run Inference on an Image
python predict.py --image test_currency.jpg
🧪 Learnings & Challenges
Initial models overfit heavily; cross-validation and feature selection helped reduce this.
Feature extraction was slow at first; optimized using vectorized NumPy operations.
Dataset imbalance required manual augmentation and resampling.
Ensemble models performed significantly better than single models.
🔬 Future Improvements
Use CNN-based deep learning models for better generalization
Add real-time camera-based detection
Deploy as a web or mobile application
Collect a larger real-world dataset
🐳 Optional: Docker (Experimental)
A basic Docker setup is included for experimentation.
docker build -t currency-detector .
docker run -p 5000:5000 currency-detector
📄 License
MIT License
📬 Contact
Zuhaib Iqbal
Email: zuhaib.fareeda@gmail.com
GitHub: https://github.com/zuhaib2525
