import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
#from sklearn.metrics import accuracy_score
#Accuracy metrics
import pickle

df = pd.read_csv(r'finalDataset.csv')

print(df.tail())

#features
X = df.drop('class', axis=1)
#target value
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

print(type(X_test))
print(X_test.iloc[69])

pipelines = {
    'lr': make_pipeline(StandardScaler(), LogisticRegression(solver='newton-cg', max_iter=100)),
    'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

'''fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train.values, y_train.values)
    fit_models[algo] = model'''

fit_models = {}

# Train the models
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy manually
    correct_predictions = sum(y_pred == y_test)
    total_predictions = len(y_test)
    accuracy = correct_predictions / total_predictions

    print(f"Accuracy of {algo}: {accuracy}")


print(fit_models['lr'].predict(X_test))

print(fit_models['rc'].predict(X_test))
print(y_test)

with open('model_gb.pkl', 'wb') as f:
     pickle.dump(fit_models['gb'], f)

print("Model loaded")
