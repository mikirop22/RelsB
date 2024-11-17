from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
from interpret import show
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import mlflow
import warnings
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from torch import nn, optim

import torch

class Train:
    def train_model_ebm():
        # Cargar y preprocesar los datos
        train = pd.read_csv("C:/Users/FX517/OneDrive - Universitat Politècnica de Catalunya/Escritorio/Hackathons/RelsB/data/train_cleaned.csv")

        X_train = train.drop("Listing.Price.ClosePrice", axis=1)
        y_train = train["Listing.Price.ClosePrice"]

        valid = pd.read_csv("C:/Users/FX517/OneDrive - Universitat Politècnica de Catalunya/Escritorio/Hackathons/RelsB/data/valid_cleaned.csv")
        X_val = valid.drop("Listing.Price.ClosePrice", axis=1)
        y_val = valid["Listing.Price.ClosePrice"]

        # Iniciar el registro en MLflow
        mlflow.set_experiment("EBM House Price Prediction")
        mlflow.start_run()

        # Crear y entrenar el modelo EBM
        model = ExplainableBoostingRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Evaluar el modelo con rmse
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        print(f"RMSE: {rmse}")

        # Registrar métricas y el modelo en MLflow
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")
        print("Modelo EBM entrenado y registrado en MLflow")

        # Finalizar la ejecución del experimento
        mlflow.end_run()


    def train_linear_regression():
        # Load and preprocess the data
        train = pd.read_csv("C:/Users/FX517/OneDrive - Universitat Politècnica de Catalunya/Escritorio/Hackathons/RelsB/data/train_cleaned.csv")

        X_train = train.drop("Listing.Price.ClosePrice", axis=1)
        y_train = train["Listing.Price.ClosePrice"]

        valid = pd.read_csv("C:/Users/FX517/OneDrive - Universitat Politècnica de Catalunya/Escritorio/Hackathons/RelsB/data/valid_cleaned.csv")
        X_val = valid.drop("Listing.Price.ClosePrice", axis=1)
        y_val = valid["Listing.Price.ClosePrice"]

        # Get just the numeric features
        X_train = X_train.select_dtypes(exclude=['object'])
        X_val = X_val.select_dtypes(exclude=['object'])

        # Start an MLflow run
        mlflow.set_experiment("Linear Regression House Price Prediction")
        mlflow.start_run()

        columns_to_log = X_train.columns
        mlflow.log_param("columns", columns_to_log)

        # Create and train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate the model with RMSE
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        print(f"RMSE: {rmse}")
        
        # Calculate R^2
        r2 = model.score(X_val, y_val)
        print(f"R^2: {r2}")

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")

        # Generate a bar chart for the coefficients
        coef = model.coef_
        features = X_train.columns
        plt.figure(figsize=(10, 6))
        plt.barh(features, coef)
        plt.xlabel('Coefficient Value')
        plt.ylabel('Feature')
        plt.title('Linear Regression Coefficients')
        plt.tight_layout()

        # Generar gráfico con mejoras

        # Print the top 5 variables by coefficient magnitude
        coef_with_features = pd.DataFrame({
            "Feature": features,
            "Coefficient": coef
        })
        coef_with_features["AbsCoefficient"] = coef_with_features["Coefficient"].abs()
        top_features = coef_with_features.sort_values("AbsCoefficient", ascending=False).head(5)
        print("Top 5 variables by coefficient magnitude:")
        print(top_features[["Feature", "Coefficient"]])
        
        top_features = coef_with_features.sort_values("AbsCoefficient", ascending=False).head(15)
        
        plt.figure(figsize=(12, 8))  # Ajustar tamaño
        plt.barh(top_features["Feature"], top_features["Coefficient"], color='skyblue')
        plt.xlabel('Coefficient Value', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Top 15 Linear Regression Coefficients', fontsize=16)
        plt.gca().invert_yaxis()  # Invertir el eje Y para que la más importante esté arriba
        plt.tight_layout()
        
        # Guardar y mostrar el gráfico
        plt.savefig("improved_linear_regression_coefficients.png")
        plt.show()

        # Save the plot as an image
        plt.savefig("linear_regression_coefficients.png")

        # Log the image to MLflow
        mlflow.log_artifact("improved_linear_regression_coefficients.png")

        print("Linear regression model trained and logged to MLflow")
        
        # Finalize the experiment run
        mlflow.end_run()

    def lgbmregressor():
        # Load and preprocess the data
        train = pd.read_csv("C:/Users/FX517/OneDrive - Universitat Politècnica de Catalunya/Escritorio/Hackathons/RelsB/data/train_cleaned.csv")
        valid = pd.read_csv("C:/Users/FX517/OneDrive - Universitat Politècnica de Catalunya/Escritorio/Hackathons/RelsB/data/valid_cleaned.csv")

        X_train = train.drop("Listing.Price.ClosePrice", axis=1)
        y_train = train["Listing.Price.ClosePrice"]

        X_valid = valid.drop("Listing.Price.ClosePrice", axis=1)
        y_valid = valid["Listing.Price.ClosePrice"]

        mlflow.set_experiment("LightGBM House Price Prediction")
        mlflow.start_run()

        numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns

        # Define the model
        model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            metric='rmse',
            random_state=42,
        )

        # X_train = train.drop("Listing.Price.ClosePrice", axis=1)
        # y_train = train["Listing.Price.ClosePrice"]

        # X_val = valid.drop("Listing.Price.ClosePrice", axis=1)
        # y_val = valid["Listing.Price.ClosePrice"]

        # X_train = X_train_encoded
        # X_val = X_val_encoded
        # X_test = X_test_encoded

        # Train the model
        model.fit(X_train[numeric_features], y_train)

        # Calculate performance metrics
        train_pred = model.predict(X_train[numeric_features])
        val_pred = model.predict(X_valid[numeric_features])

        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_valid, val_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'val_r2': r2_score(y_valid, val_pred)
        }

        # Log the model and metrics
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metrics(metrics)

        # Calcular la importancia de las características
        feature_importance = model.feature_importances_

        # Crear un DataFrame para las características y su importancia
        feature_importance_df = pd.DataFrame({
            'Feature': numeric_features,
            'Importance': feature_importance
        })

        # Ordenar las características por su importancia
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Mostrar las variables más importantes
        top_features = feature_importance_df.head(10)
        print("Top 10 características más importantes:")
        print(top_features)

        # guardar las 30 variables mas importantes en MLflow
        top_features = feature_importance_df.head(30)
        plt.figure(figsize=(12, 8))
        plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Top 30 Feature Importance', fontsize=16)
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # Guardar y mostrar el gráfico
        plt.savefig("feature_importance.png")
        plt.show()

        # Log the feature importance plot to MLflow
        mlflow.log_artifact("feature_importance.png")


        # log the feature importance
        feature_importance = model.feature_importances_
        feature_importance = pd.Series(feature_importance, index=numeric_features)
        feature_importance = feature_importance.sort_values(ascending=False)

        # Plot the feature importance
        plt.figure(figsize=(10, 6))
        feature_importance.plot(kind='bar')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()

        # log the feature importance plot to MLflow
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")

        # End the MLflow run
        mlflow.end_run()

    def neural_network():
        # Load and preprocess the data
        train = pd.read_csv("C:/Users/FX517/OneDrive - Universitat Politècnica de Catalunya/Escritorio/Hackathons/RelsB/data/train_cleaned.csv")
        valid = pd.read_csv("C:/Users/FX517/OneDrive - Universitat Politècnica de Catalunya/Escritorio/Hackathons/RelsB/data/valid_cleaned.csv")

        y_train = train["Listing.Price.ClosePrice"]
        y_val = valid["Listing.Price.ClosePrice"]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        cols_categorical = train.select_dtypes(include=['object']).columns

        encoder = ce.BinaryEncoder(cols=cols_categorical)
        X_train_encoded = encoder.fit_transform(train)
        X_val_encoded = encoder.transform(valid)

        # Escalar las características numéricas (si es necesario)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_encoded)
        X_val_scaled = scaler.transform(X_val_encoded)

        # Definir el modelo de red neuronal
        class HousePricePredictor(nn.Module):
            def __init__(self):
                super(HousePricePredictor, self).__init__()
                self.model = nn.Sequential(
                    nn.Linear(555, 256),  
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.Dropout(0.2),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.Linear(32, 1)  # 1 salida (precio de la casa)
                )
            
            def forward(self, x):
                return self.model(x)

        # Crear el modelo
        model = HousePricePredictor().to(device)

        # Definir la función de pérdida y el optimizador
        criterion = nn.MSELoss()  # Para regresión utilizamos MSE
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Función para calcular RMSE y R²
        def calculate_metrics(y_true, y_pred):
            y_true = y_true.detach().numpy()
            y_pred = y_pred.detach().numpy()
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            return rmse, r2

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1).to(device)

        X_test_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_val.values, dtype=torch.float32).reshape(-1, 1).to(device)

        # Entrenamiento del modelo
        epochs = 10
        for epoch in tqdm(range(epochs)):
            # Entrenamiento
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            y_pred_train = model(X_train_tensor)
            
            # Calcular la pérdida
            loss = criterion(y_pred_train, y_train_tensor)
            
            # Backward pass y optimización
            loss.backward()
            optimizer.step()
            
            # Validación
            model.eval()
            with torch.no_grad():
                y_pred_test = model(X_test_tensor)
                rmse, r2 = calculate_metrics(y_test_tensor, y_pred_test)
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}')

            # Log the metrics to MLflow
            mlflow.log_metric("loss", loss.item(), step=epoch)
            mlflow.log_metric("rmse", rmse, step=epoch)
            mlflow.log_metric("r2", r2, step=epoch)

        # Evaluar el modelo con el conjunto de prueba
        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test_tensor)
            rmse, r2 = calculate_metrics(y_test_tensor, y_pred_test)

        print(f'Final RMSE: {rmse:.4f}')
        print(f'Final R2: {r2:.4f}')
