#!/usr/bin/env python3
# Neural Network for Nighttime Lights Analysis Using Extracted Features

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from skimage.transform import resize

# Configuration
IMAGE_DIR = "./Delhi_Nighttime_Lights"  # Change as needed
# Not used for CNN now; used for optional image resize if needed
TARGET_SIZE = (128, 128)
REGION = "Delhi"  # or "Singapore"

# Economic data for Delhi (example)
delhi_gdp = {
    'Year': [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'GDP': [343798, 391388, 443960, 494803, 550804, 616085, 677900, 738389, 792911, 763435, 904642, 1043759]
}


def extract_features_from_tif(image_path):
    """Extract a set of features from a TIF satellite image."""
    try:
        with rasterio.open(image_path) as src:
            img = src.read(1)
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None
    # Handle masked or NaN values
    if np.ma.is_masked(img):
        img = img.filled(fill_value=0)
    img = np.nan_to_num(img, nan=0)

    # Optionally, you can resize the image (if desired for consistency)
    # Here we simply use the full image for feature extraction.
    # img = resize(img, TARGET_SIZE, preserve_range=True)

    # Compute basic statistics
    light_mean = np.nanmean(img)
    light_std = np.nanstd(img)
    light_max = np.nanmax(img)
    threshold = 30
    urban_area_ratio = (img > threshold).sum() / img.size
    brightness_urban = light_mean * urban_area_ratio

    # Histogram-based entropy
    hist, _ = np.histogram(img.astype(np.uint8), bins=256, range=(0, 256))
    if hist.sum() > 0:
        hist = hist / hist.sum()
    non_zero = hist[hist > 0]
    entropy = -np.sum(non_zero * np.log2(non_zero)) if non_zero.size > 0 else 0

    # GLCM properties: contrast and energy
    try:
        img_uint8 = img.astype(np.uint8)
        if img_uint8.max() == 0:
            img_uint8[0, 0] = 1  # avoid division by zero in GLCM
        glcm = graycomatrix(img_uint8, distances=[1], angles=[0],
                            levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
    except Exception as e:
        print("Error computing GLCM features:", e)
        contrast = 0
        energy = 0

    features = {
        'light_mean': light_mean,
        'light_std': light_std,
        'light_max': light_max,
        'urban_area_ratio': urban_area_ratio,
        'brightness_urban': brightness_urban,
        'light_entropy': entropy,
        'light_contrast': contrast,
        'light_energy': energy
    }
    return features


def process_images_and_prepare_data():
    """Process images, extract features, and prepare a chronological dataset with economic data."""
    print("Processing images and preparing data...")

    # Create economic dataframe from provided GDP data
    econ_df = pd.DataFrame(delhi_gdp)

    # Create monthly economic data (each year has 12 months with same GDP value)
    monthly_data = []
    for year in econ_df['Year']:
        for month in range(1, 13):
            monthly_data.append({
                'year': year,
                'month': month,
                'GDP': econ_df[econ_df['Year'] == year]['GDP'].values[0]
            })
    monthly_econ = pd.DataFrame(monthly_data)

    # Collect image metadata from the directory
    valid_metadata = []
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Directory {IMAGE_DIR} does not exist")
        return None, None, None, None, None, None, None

    print("Collecting image metadata...")
    for fname in os.listdir(IMAGE_DIR):
        if fname.endswith('.tif'):
            parts = fname.split('_')
            if len(parts) >= 5:
                try:
                    year = int(parts[3])
                    month = int(parts[4].split('.')[0])
                    image_path = os.path.join(IMAGE_DIR, fname)
                    valid_metadata.append({
                        'year': year,
                        'month': month,
                        'path': image_path
                    })
                except ValueError:
                    continue

    meta_df = pd.DataFrame(valid_metadata)
    # Merge metadata with economic data based on year and month
    merged_df = pd.merge(meta_df, monthly_econ, on=[
                         'year', 'month'], how='inner')
    print(f"Found {len(merged_df)} images with matching economic data")

    # Process matched images: extract features and record corresponding GDP and date
    X_features = []
    y = []
    dates = []

    print("Extracting features from images...")
    for _, row in merged_df.iterrows():
        features = extract_features_from_tif(row['path'])
        if features is not None:
            # Define feature vector (order is important)
            feature_vector = [
                features['light_mean'],
                features['light_std'],
                features['light_max'],
                features['urban_area_ratio'],
                features['brightness_urban'],
                features['light_entropy'],
                features['light_contrast'],
                features['light_energy']
            ]
            X_features.append(feature_vector)
            y.append(row['GDP'])
            dates.append(pd.Timestamp(
                year=int(row['year']), month=int(row['month']), day=1))

    if not X_features:
        print("Error: No valid images processed")
        return None, None, None, None, None, None, None

    # Feature matrix shape: (n_samples, n_features)
    X_features = np.array(X_features)
    y = np.array(y)
    dates = np.array(dates)
    print(f"Final dataset: {len(X_features)} samples with extracted features")

    # Create a DataFrame for chronological sorting
    full_df = pd.DataFrame({
        'date': dates,
        'index': np.arange(len(X_features))
    }).sort_values('date')

    # Chronological train/test split (e.g., 80% training)
    train_size = int(len(full_df) * 0.8)
    indices_train = full_df['index'].values[:train_size]
    indices_test = full_df['index'].values[train_size:]
    X_train = X_features[indices_train]
    X_test = X_features[indices_test]
    y_train = y[indices_train]
    y_test = y[indices_test]
    train_dates = dates[indices_train]
    test_dates = dates[indices_test]

    print(f"Training on data from {min(train_dates)} to {max(train_dates)}")
    print(f"Testing on data from {min(test_dates)} to {max(test_dates)}")

    # Scale feature values (optional but recommended for neural networks)
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    # Scale target values
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # Create a DataFrame for test dates and actual GDP for plotting later
    test_df = pd.DataFrame({
        'date': test_dates,
        'actual_GDP': y_test
    })

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler, test_df, train_dates


def create_model(input_dim):
    """Create a simple dense neural network for regression using extracted features."""
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


def train_and_evaluate():
    """Main function to process data, train the model with extracted features using a dense network, and evaluate."""
    X_train, X_test, y_train, y_test, y_scaler, test_df, train_dates = process_images_and_prepare_data()

    if X_train is None:
        print("Failed to prepare data. Exiting.")
        return

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    print("Creating and training model with extracted features...")
    model = create_model(input_dim=X_train.shape[1])
    model.summary()

    # Optionally, you can further split the training set for validation (chronologically)
    # For simplicity, here we use 20% of training data as validation.
    val_size = int(X_train.shape[0] * 0.2)
    X_train_final = X_train[:-val_size]
    y_train_final = y_train[:-val_size]
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    print(
        f"Final training: {X_train_final.shape[0]} samples, Validation: {X_val.shape[0]} samples")

    history = model.fit(
        X_train_final, y_train_final,
        epochs=5000,
        batch_size=60,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )

    print("Evaluating model on test data...")
    y_pred = model.predict(X_test).flatten()
    y_pred_actual = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_actual = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)
    mape = np.mean(
        np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100

    print("Model Evaluation:")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # ------------------ PLOTTING ------------------ #

    # 1. Training History Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], marker='o', label='Training Loss')
    plt.plot(history.history['val_loss'], marker='o', label='Validation Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss over Epochs', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('training_history_features.png')
    plt.close()
    print("Training history plot saved as 'training_history_features.png'")

    # 2. Predictions vs Actual Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_actual, y_pred_actual, c='blue',
                alpha=0.6, edgecolors='k', label='Data Points')
    plt.plot([y_test_actual.min(), y_test_actual.max()],
             [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2, label='Ideal Fit')
    plt.xlabel('Actual GDP', fontsize=12)
    plt.ylabel('Predicted GDP', fontsize=12)
    plt.title('GDP Predictions vs Actual Values (Features)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('predictions_features.png')
    plt.close()
    print("Predictions plot saved as 'predictions_features.png'")

    # 3. Time Series Comparison Plot
    test_df['predicted_GDP'] = y_pred_actual
    plot_title = f'Time Series of Actual vs Predicted GDP\nR² = {r2:.4f}, MAE = {mae:.2f}, MAPE = {mape:.2f}%'
    plt.figure(figsize=(12, 6))
    plt.plot(test_df['date'], test_df['actual_GDP'],
             marker='o', linestyle='-', label='Actual GDP')
    plt.plot(test_df['date'], test_df['predicted_GDP'],
             marker='s', linestyle='--', label='Predicted GDP')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('GDP', fontsize=12)
    plt.title(plot_title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('time_series_comparison_features.png')
    plt.close()
    print("Time series comparison plot saved as 'time_series_comparison_features.png'")

    # 4. Prediction Error Over Time Plot
    test_df['error_pct'] = (test_df['predicted_GDP'] -
                            test_df['actual_GDP']) / test_df['actual_GDP'] * 100
    plt.figure(figsize=(12, 6))
    plt.bar(test_df['date'], test_df['error_pct'],
            alpha=0.7, color='skyblue', edgecolor='black')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.7)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Prediction Error (%)', fontsize=12)
    plt.title('Prediction Error Over Time (Walk-Forward Validation)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('prediction_error_over_time_features.png')
    plt.close()
    print("Prediction error over time plot saved as 'prediction_error_over_time_features.png'")

    # 5. Cumulative Forecast Error Plot
    test_df['abs_error'] = np.abs(
        test_df['predicted_GDP'] - test_df['actual_GDP'])
    plt.figure(figsize=(12, 6))
    plt.plot(test_df['date'], test_df['abs_error'].cumsum(),
             marker='o', linestyle='-', label='Cumulative Absolute Error')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Error', fontsize=12)
    plt.title('Cumulative Forecast Error Over Time', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('cumulative_error_features.png')
    plt.close()
    print("Cumulative error plot saved as 'cumulative_error_features.png'")

    # Save the trained model
    model.save('nightlights_nn_features.h5')
    print("Model saved as 'nightlights_nn_features.h5'")


if __name__ == "__main__":
    train_and_evaluate()
