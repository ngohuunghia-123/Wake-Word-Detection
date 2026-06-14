import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib


# ==============================
# LOAD DATA
# ==============================

X_wake = np.load("final_features/wake_word.npy")
X_noise = np.load("final_features/background.npy")

X = np.vstack([X_wake, X_noise])
y = np.array([1]*len(X_wake) + [0]*len(X_noise))

print("Dataset shape:", X.shape)
print("Wakeword samples:", len(X_wake))
print("Noise samples:", len(X_noise))


# ==============================
# TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# ==============================
# MODEL CONFIGURATION
# ==============================

models = {

    "SVM": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(class_weight="balanced", probability=True))
        ]),
        {
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__gamma": ["scale", "auto", 0.01, 0.001],
            "clf__kernel": ["rbf"]
        }
    ),

    "RandomForest": (
        RandomForestClassifier(
            class_weight="balanced",
            random_state=42
        ),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }
    ),

    "KNN": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier())
        ]),
        {
            "clf__n_neighbors": [3, 5, 7, 9],
            "clf__weights": ["uniform", "distance"]
        }
    )
}


best_model = None
best_score = 0
best_name = ""


# ==============================
# TRAIN + GRID SEARCH
# ==============================

for name, (model, params) in models.items():

    print("\n==========================")
    print("Training", name)

    grid = GridSearchCV(
        model,
        params,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Best params:", grid.best_params_)
    print("Accuracy:", acc)
    print("F1 score:", f1)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    if f1 > best_score:
        best_score = f1
        best_model = grid.best_estimator_
        best_name = name


# ==============================
# SAVE BEST MODEL
# ==============================

joblib.dump(best_model, "best_wakeword_model.pkl")

print("\n==========================")
print("Best Model:", best_name)
print("Best F1 Score:", best_score)
print("Model saved as best_wakeword_model.pkl")


# ==============================
# THRESHOLD GỢI Ý
# ==============================

print("\nSuggested detection threshold: 0.75 - 0.9")
print("Use model.predict_proba(features)[0][1] for wakeword probability")