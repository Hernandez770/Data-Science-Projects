import preprocessing
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = preprocessing.load_data()

X = df[['NY.GDP.MKTP.KD','SP.POP.TOTL','AG.SRF.TOTL.K2']]
y = df['SP.DYN.LE00.IN']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

models = {"Random Forest": RandomForestRegressor()}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

# Train and save the model
model = RandomForestRegressor()
model.fit(X_train, y_train)
joblib.dump(model, 'new_model.joblib')