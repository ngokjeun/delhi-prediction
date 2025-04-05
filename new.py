import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

# ----- Functions for Satellite Image Processing -----
def extract_features_from_tif(image_path):
    try:
        with rasterio.open(image_path) as src:
            img = src.read(1)
            profile = src.profile
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None, None, None
    
    # Handle masked or NaN values
    if np.ma.is_masked(img):
        img = img.filled(fill_value=0)
    img = np.nan_to_num(img, nan=0)

    light_mean = np.nanmean(img)
    light_std = np.nanstd(img)
    light_max = np.nanmax(img)
    threshold = 30
    urban_area_ratio = (img > threshold).sum() / img.size
    brightness_urban = light_mean * urban_area_ratio

    # Compute histogram-based entropy
    hist, _ = np.histogram(img.astype(np.uint8), bins=256, range=(0, 256))
    if hist.sum() > 0:
        hist = hist / hist.sum()
    non_zero = hist[hist > 0]
    entropy = -np.sum(non_zero * np.log2(non_zero)) if non_zero.size > 0 else 0

    # Compute GLCM features (contrast & energy)
    try:
        img_uint8 = img.astype(np.uint8)
        if img_uint8.max() == 0:
            img_uint8[0, 0] = 1  # avoid zero max value issues
        glcm = graycomatrix(img_uint8, distances=[1], angles=[0],
                            levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
    except Exception as e:
        print("Error computing GLCM properties:", e)
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
    return features, img, profile

def process_images(image_dir, region="Delhi"):
    features_list = []
    image_files = []

    if not os.path.exists(image_dir):
        print(f"Directory {image_dir} does not exist")
        return pd.DataFrame(), []

    file_count = 0
    for fname in os.listdir(image_dir):
        if fname.endswith('.tif'):
            # Example filename: prefix_anyinfo_year_month.tif
            parts = fname.split('_')
            if len(parts) >= 5:
                try:
                    year = int(parts[3])
                    month = int(parts[4].split('.')[0])
                except ValueError:
                    continue
            else:
                continue

            image_path = os.path.join(image_dir, fname)
            image_files.append({
                "path": image_path,
                "year": year,
                "month": month,
                "filename": fname
            })

            features, _, _ = extract_features_from_tif(image_path)
            if features:
                features['year'] = year
                features['month'] = month
                features['date'] = pd.Timestamp(year=year, month=month, day=1)
                features['image_path'] = image_path
                features_list.append(features)
                file_count += 1

    print(f"Processed {file_count} images from {image_dir}")
    return pd.DataFrame(features_list), image_files

# ----- Functions for Economic Data -----
def load_economic_data(region="Delhi"):
    if region == "Delhi":
        # Sample GDP data for Delhi
        sdp_data = {
            'Year': [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
            'GDP': [343798, 391388, 443960, 494803, 550804, 616085, 677900, 738389, 792911, 763435, 904642, 1043759]
        }
        econ_df = pd.DataFrame(sdp_data)
        econ_df['date'] = pd.to_datetime(econ_df['Year'].astype(str) + '-01-01')
        econ_df = econ_df.set_index('date')
        return econ_df, 'GDP'
    else:
        # Extend with other regions if needed
        return None, None

# ----- Build the MLP (TF Neural Network) Model -----
def build_tf_nn_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),  # Helps mitigate overfitting
        Dense(32, activation='relu'),
        Dense(1)       # Single output for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ----- Main Script -----
def main():
    # Settings
    region = "Delhi"
    image_dir = "./Delhi_Nighttime_Lights"  # Change to your satellite images directory

    # 1. Process Satellite Images to Extract Features
    features_df, image_files = process_images(image_dir, region)
    if features_df.empty:
        print("No image features extracted. Exiting.")
        return

    # 2. Load Economic Data
    econ_df, economic_variable = load_economic_data(region)
    if econ_df is None:
        print("Economic data not available. Exiting.")
        return

    # 3. Create a Monthly Economic DataFrame (Interpolated)
    monthly_dates = pd.date_range(start=econ_df.index.min(), end='2024-01-01', freq='MS')
    monthly_econ = econ_df.reindex(monthly_dates).interpolate(method='linear')
    monthly_econ = monthly_econ.reset_index().rename(columns={'index': 'date'})
    monthly_econ['year'] = monthly_econ['date'].dt.year
    monthly_econ['month'] = monthly_econ['date'].dt.month

    # 4. Merge Satellite Image Features with Economic Data (on year & month)
    merged_df = pd.merge(features_df, monthly_econ, on=['year', 'month'], how='inner')
    if merged_df.empty:
        print("Merged data is empty. Exiting.")
        return
    print("Merged dataset shape:", merged_df.shape)

    # 5. Define Feature Columns (from Satellite Images)
    feature_columns = [
        'light_mean', 'light_std', 'light_max',
        'urban_area_ratio', 'brightness_urban', 'light_entropy',
        'light_contrast', 'light_energy'
    ]
    
    # 6. Prepare Data for Modeling
    # Use the merged dataset sorted by date (time-series split)
    merged_df['date'] = merged_df['date_y']
    merged_df = merged_df.sort_values('date')
    X = merged_df[feature_columns].values
    y = merged_df[economic_variable].values

    # Time-based split: use the first 80% for training, the rest for testing
    train_size = int(len(merged_df) * 0.8)
    X_train = merged_df[feature_columns].values[:train_size]
    X_test = merged_df[feature_columns].values[train_size:]
    y_train = merged_df[economic_variable].values[:train_size]
    y_test = merged_df[economic_variable].values[train_size:]

    # 7. Scale the Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 8. Build and Train the MLP Model
    model = build_tf_nn_model(X_train_scaled.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100,
                        callbacks=[early_stopping], verbose=1)

    # 9. Evaluate the Model
    loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    print("Test Loss (MSE):", loss)
    print("Test MAE:", mae)

    # 10. Generate Predictions and Plot Results
    y_pred = model.predict(X_test_scaled).flatten()
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, label='Predictions')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2, label='Ideal')
    plt.xlabel("Actual GDP")
    plt.ylabel("Predicted GDP")
    plt.title("MLP Regression on Satellite Image Features")
    plt.legend()
    plt.show()

    # Print additional performance metrics
    mse = mean_squared_error(y_test, y_pred)
    mae_metric = mean_absolute_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae_metric:.2f}")

if __name__ == "__main__":
    main()
