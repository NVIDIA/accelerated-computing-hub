from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load

import time

# synthetic dataset dimensions
n_samples = 100_000
n_features = 10
n_classes = 2

# random forest depth and size
n_estimators = 25
max_depth = 10

start = time.time()

# generate synthetic data [ binary classification task ]
X, y = make_classification(
    n_classes=n_classes,
    n_features=n_features,
    n_samples=n_samples,
)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = RandomForestClassifier(
    max_depth=max_depth,
    n_estimators=n_estimators,
)

trained_RF = model.fit(X_train, y_train)

predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)

end = time.time()

print("sklearn accuracy:", score)
print(f"Total elapsed time for RF: {end - start:.4f} seconds")