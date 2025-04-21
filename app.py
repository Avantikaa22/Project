from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Sample data (replace with actual dataset)
    df2 = pd.DataFrame({
        'Year': [2006, 2007, 2008, 2009, 2010],
        'Type': ['Imports', 'Imports', 'Exports', 'Exports', 'Imports'],
        'Automobile Volume': [100, 150, 120, 180, 160]
    })

    # Filter data for years 2006 to 2015
    df_years = df2[(df2['Year'] >= 2000) & (df2['Year'] <= 2015)]

    # Create a numerical column for the year
    df_years['Year Numeric'] = df_years['Year'] - 2006

    # Separate the data for imports and exports
    df_imports = df_years[df_years['Type'] == 'Imports']
    df_exports = df_years[df_years['Type'] == 'Exports']

    # Linear regression for Imports
    X_imports = df_imports[['Year Numeric']]
    y_imports = df_imports['Automobile Volume']

    model_imports = LinearRegression()
    model_imports.fit(X_imports, y_imports)

    years_future = np.array([2016, 2017]) - 2006
    predictions_imports = model_imports.predict(years_future.reshape(-1, 1))

    # Linear regression for Exports
    X_exports = df_exports[['Year Numeric']]
    y_exports = df_exports['Automobile Volume']

    model_exports = LinearRegression()
    model_exports.fit(X_exports, y_exports)

    predictions_exports = model_exports.predict(years_future.reshape(-1, 1))

    predictions = pd.DataFrame({
        'Year': [2016, 2017],
        'Predicted Imports': predictions_imports,
        'Predicted Exports': predictions_exports
    })

    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
