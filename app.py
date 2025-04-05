import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error, r2_score

import os
import pickle
from datetime import datetime
from skimage.feature import graycomatrix, graycoprops
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(page_title="Delhi Nighttime Lights", layout="wide")

# Add a title
st.title("Delhi Nighttime Lights & SDP Analysis")

# Function to load data


@st.cache_data
def load_data():
    # Annual SDP data
    sdp_data = {
        'Year': [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
        'SDP': [343798, 391388, 443960, 494803, 550804, 616085, 677900, 738389, 792911, 763435, 904642, 1043759]
    }
    sdp_df = pd.DataFrame(sdp_data)
    sdp_df['date'] = pd.to_datetime(sdp_df['Year'].astype(str) + '-01-01')
    sdp_df = sdp_df.set_index('date')

    # Monthly interpolation
    monthly_dates = pd.date_range(
        start=sdp_df.index.min(), end='2024-01-01', freq='MS')
    monthly_sdp = sdp_df.reindex(monthly_dates).interpolate(method='linear')
    monthly_sdp = monthly_sdp.reset_index().rename(columns={'index': 'date'})
    monthly_sdp['year'] = monthly_sdp['date'].dt.year
    monthly_sdp['month'] = monthly_sdp['date'].dt.month

    return sdp_df, monthly_sdp

# Function to extract features from TIF images


def extract_features_from_tif(image_path):
    with rasterio.open(image_path) as src:
        img = src.read(1)
        profile = src.profile

    # Handle masked values and NaNs
    if np.ma.is_masked(img):
        img = img.filled(fill_value=0)
    img = np.nan_to_num(img, nan=0)

    # Basic statistics
    light_sum = np.nansum(img)
    light_mean = np.nanmean(img)
    light_std = np.nanstd(img)
    light_max = np.nanmax(img)

    # Urban area ratio
    threshold = 30
    urban_area_ratio = (img > threshold).sum() / img.size
    brightness_urban = light_mean * urban_area_ratio

    # Entropy calculation
    hist, _ = np.histogram(img.astype(np.uint8), bins=256, range=(0, 256))
    if hist.sum() > 0:
        hist = hist / hist.sum()
    non_zero = hist[hist > 0]
    entropy = -np.sum(non_zero * np.log2(non_zero)) if non_zero.size > 0 else 0

    # Texture features
    img_uint8 = img.astype(np.uint8)
    if img_uint8.max() == 0:
        img_uint8[0, 0] = 1
    glcm = graycomatrix(img_uint8, distances=[1], angles=[
                        0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]

    return {
        'light_sum': light_sum,
        'light_mean': light_mean,
        'light_std': light_std,
        'light_max': light_max,
        'urban_area_ratio': urban_area_ratio,
        'brightness_urban': brightness_urban,
        'light_entropy': entropy,
        'light_contrast': contrast,
        'light_energy': energy
    }, img, profile


@st.cache_data
def process_images(image_dir):
    features_list = []
    image_files = []

    for fname in os.listdir(image_dir):
        if fname.endswith('.tif'):
            parts = fname.split('_')
            if len(parts) >= 5:
                try:
                    year = int(parts[3])
                    month = int(parts[4].split('.')[0])
                except ValueError:
                    continue
                image_path = os.path.join(image_dir, fname)
                image_files.append(
                    {"path": image_path, "year": year, "month": month, "filename": fname})
                feats, _, _ = extract_features_from_tif(image_path)
                feats['year'] = year
                feats['month'] = month
                feats['date'] = pd.Timestamp(year=year, month=month, day=1)
                feats['image_path'] = image_path
                features_list.append(feats)

    return pd.DataFrame(features_list), image_files


# Load SDP data
sdp_df, monthly_sdp = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", [
                        "Data Explorer", "Image Viewer", "Feature Analysis", "Model Testing"])

# Set directory path
image_dir = st.sidebar.text_input(
    "TIF Images Directory", "./Delhi_Nighttime_Lights")

# # Data source selection
# data_source = st.sidebar.radio(
#     "Data Source", ["Pickle File", "Process Images"], index=0)

image_files = []


pickle_path = st.sidebar.text_input(
    "Path to Pickle File of Extracted Features DataFrame", "delhi_nightlights_features.pkl")

if os.path.exists(pickle_path):
    try:
        with open(pickle_path, 'rb') as f:
            features_df = pickle.load(f)
        st.sidebar.success(f"Loaded {len(features_df)} records")

        if 'image_path' in features_df.columns:
            image_files = [{"path": path, "year": year, "month": month, "filename": os.path.basename(path)}
                            for path, year, month in zip(features_df['image_path'], features_df['year'], features_df['month'])]
    except Exception as e:
        st.sidebar.error(f"Error loading pickle file: {e}")
        st.error("Failed to load pickle file.")
        st.stop()
else:
    st.sidebar.error(f"Pickle file not found.")
    st.error("Pickle file not found.")
    st.stop()

# Check for data
if features_df.empty:
    st.error("No data available.")
    st.stop()

# Merge image features with monthly SDP data
merged_df = pd.merge(features_df, monthly_sdp, on=[
                     'year', 'month'], how='inner')
merged_df['month_int'] = merged_df['month']

# Define available models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression(),
    "Support Vector Regression": SVR(kernel='linear')
}

# Define feature columns
feature_columns = [
    'light_sum', 'light_mean', 'light_std', 'light_max',
    'urban_area_ratio', 'brightness_urban', 'light_entropy',
    'light_contrast', 'light_energy', 'month_int'
]

# Page: Data Explorer
if page == "Data Explorer":
    st.header("Data Explorer")

    # SDP Data
    st.subheader("SDP Data")
    st.dataframe(sdp_df.reset_index())

    # SDP trend plot
    fig = px.line(sdp_df.reset_index(), x='date', y='SDP',
                  title="Delhi SDP Trend", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    # Monthly data
    st.subheader("Monthly Data")
    st.dataframe(monthly_sdp)

    # Features data
    st.subheader("Features Data")
    st.dataframe(features_df)



    # Merged data
    st.subheader("Merged Data")
    st.dataframe(merged_df)

# Page: Image Viewer
elif page == "Image Viewer":
    st.header("Nighttime Light Image Viewer")

    if 'image_path' not in merged_df.columns:
        st.error("Image paths not available.")
        st.stop()

    # Check for valid paths
    has_valid_paths = any(os.path.exists(path)
                          for path in merged_df['image_path'].unique())

    if not has_valid_paths:
        st.warning("Image paths don't exist on this system.")

        # Path updater
        st.subheader("Update Image Paths")
        new_image_dir = st.text_input("Image Directory", image_dir)

        if st.button("Update Paths") and os.path.exists(new_image_dir):
            merged_df_copy = merged_df.copy()
            available_images = {}

            for fname in os.listdir(new_image_dir):
                if fname.endswith('.tif'):
                    parts = fname.split('_')
                    if len(parts) >= 5:
                        try:
                            year = int(parts[3])
                            month = int(parts[4].split('.')[0])
                            available_images[(year, month)] = os.path.join(
                                new_image_dir, fname)
                        except ValueError:
                            continue

            updated_count = 0
            for i, row in merged_df_copy.iterrows():
                key = (row['year'], row['month'])
                if key in available_images:
                    merged_df_copy.at[i, 'image_path'] = available_images[key]
                    updated_count += 1

            if updated_count > 0:
                merged_df = merged_df_copy
                st.success(f"Updated {updated_count} image paths.")
            else:
                st.error("No matching images found.")

        st.error("No valid image paths found.")
    else:
        # Time slider implementation
        st.subheader("Explore Images Over Time")

        # Get valid dates and paths
        valid_dates = []
        date_to_path = {}

        for _, row in merged_df.iterrows():
            if os.path.exists(row['image_path']):
                date_obj = pd.Timestamp(
                    year=row['year'], month=row['month'], day=1)
                valid_dates.append(date_obj)
                date_to_path[date_obj] = row['image_path']

        valid_dates = sorted(valid_dates)

        if not valid_dates:
            st.error("No valid image files found.")
            st.stop()

        # Slider
        selected_index = st.slider(
            "Select Date", min_value=0, max_value=len(valid_dates)-1, value=0)
        selected_date = valid_dates[selected_index]
        st.info(f"Viewing: {selected_date.strftime('%B %Y')}")

        # Auto-play option
        auto_play = st.checkbox("Auto-play")
        if auto_play:
            play_speed = st.slider("Speed (seconds)", 0.2, 5.0, 0.3, 0.5)

        # Get image and display
        selected_image = date_to_path[selected_date]
        selected_year = selected_date.year
        selected_month = selected_date.month

        if auto_play:
            import time
            placeholder = st.empty()

            for i in range(selected_index, len(valid_dates)):
                with placeholder.container():
                    current_date = valid_dates[i]
                    current_image = date_to_path[current_date]

                    if os.path.exists(current_image):
                        feats, img, _ = extract_features_from_tif(
                            current_image)

                        # Display
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            fig, ax = plt.subplots(figsize=(10, 8))
                            show(img, ax=ax, cmap='viridis',
                                 title=f"Delhi - {current_date.strftime('%B %Y')}")
                            plt.colorbar(
                                ax.images[0], ax=ax, label="Light Intensity")
                            st.pyplot(fig)

                        with col2:
                            # Features
                            st.subheader("Features")
                            for key, value in feats.items():
                                if key not in ['year', 'month', 'date', 'image_path']:
                                    st.metric(key, f"{value:.2f}" if isinstance(
                                        value, float) else value)

                            # SDP value
                            sdp_value = merged_df[
                                (merged_df['year'] == current_date.year) &
                                (merged_df['month'] == current_date.month)
                            ]['SDP'].values[0]

                            st.subheader("SDP")
                            st.metric("SDP Value", f"{sdp_value:.2f}")

                time.sleep(play_speed)
        else:
            if os.path.exists(selected_image):
                feats, img, _ = extract_features_from_tif(selected_image)

                # Display layout
                col1, col2 = st.columns([3, 1])

                with col1:
                    # Main image
                    fig, ax = plt.subplots(figsize=(10, 8))
                    show(img, ax=ax, cmap='viridis',
                         title=f"Delhi - {selected_date.strftime('%B %Y')}")
                    plt.colorbar(ax.images[0], ax=ax, label="Light Intensity")
                    st.pyplot(fig)

                    # Feature trend
                    st.subheader("Feature Trend")
                    feature_options = [col for col in merged_df.columns if col in [
                        'light_mean', 'urban_area_ratio', 'brightness_urban', 'SDP'
                    ]]
                    selected_feature = st.selectbox(
                        "Select Feature", feature_options)

                    # Plot trend
                    time_data = merged_df.sort_values('date')
                    fig = px.line(time_data, x='date', y=selected_feature,
                                  title=f"{selected_feature} Over Time")
                    fig.add_vline(x=selected_date,
                                  line_color="red", line_width=2)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Features
                    st.subheader("Image Features")
                    for key, value in feats.items():
                        if key not in ['year', 'month', 'date', 'image_path']:
                            st.metric(key, f"{value:.2f}" if isinstance(
                                value, float) else value)

                    # SDP value
                    sdp_value = merged_df[
                        (merged_df['year'] == selected_year) &
                        (merged_df['month'] == selected_month)
                    ]['SDP'].values[0]

                    st.subheader("SDP Information")
                    st.metric("SDP Value", f"{sdp_value:.2f}")

                    # Month-over-month comparison
                    if selected_index > 0:
                        previous_date = valid_dates[selected_index - 1]
                        previous_sdp = merged_df[
                            (merged_df['year'] == previous_date.year) &
                            (merged_df['month'] == previous_date.month)
                        ]['SDP'].values[0]

                        sdp_change = (
                            (sdp_value - previous_sdp) / previous_sdp) * 100
                        st.metric("Monthly Change", f"{sdp_change:.2f}%")

                    # Histogram
                    st.subheader("Pixel Distribution")
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.hist(img.flatten(), bins=30, alpha=0.7, color='teal')
                    st.pyplot(fig)

                # Compare with previous/next
                st.subheader("Temporal Comparison")
                col1, col2, col3 = st.columns([1, 2, 1])

                with col1:
                    if selected_index > 0:
                        previous_date = valid_dates[selected_index - 1]
                        previous_image = date_to_path[previous_date]

                        if os.path.exists(previous_image):
                            _, prev_img, _ = extract_features_from_tif(
                                previous_image)

                            fig, ax = plt.subplots(figsize=(5, 4))
                            show(prev_img, ax=ax, cmap='viridis',
                                 title=f"Previous: {previous_date.strftime('%b %Y')}")
                            st.pyplot(fig)

                with col3:
                    if selected_index < len(valid_dates) - 1:
                        next_date = valid_dates[selected_index + 1]
                        next_image = date_to_path[next_date]

                        if os.path.exists(next_image):
                            _, next_img, _ = extract_features_from_tif(
                                next_image)

                            fig, ax = plt.subplots(figsize=(5, 4))
                            show(next_img, ax=ax, cmap='viridis',
                                 title=f"Next: {next_date.strftime('%b %Y')}")
                            st.pyplot(fig)

                with col2:
                    # Difference map
                    if selected_index > 0:
                        previous_date = valid_dates[selected_index - 1]
                        previous_image = date_to_path[previous_date]

                        if os.path.exists(previous_image):
                            _, prev_img, _ = extract_features_from_tif(
                                previous_image)

                            # Calculate difference
                            diff_img = img.astype(
                                float) - prev_img.astype(float)
                            vmax = max(abs(diff_img.min()),
                                       abs(diff_img.max()))
                            vmin = -vmax

                            # Show difference
                            fig, ax = plt.subplots(figsize=(8, 6))
                            im = ax.imshow(
                                diff_img, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                            ax.set_title(
                                f"Change: {previous_date.strftime('%b %Y')} to {selected_date.strftime('%b %Y')}")
                            plt.colorbar(im, ax=ax, label="Light Change")
                            st.pyplot(fig)

# Page: Feature Analysis
elif page == "Feature Analysis":
    st.header("Feature Analysis")

    # Feature selection
    selected_feature = st.selectbox("Select Feature", feature_columns)
    merged_df['date'] = merged_df['date_y']
    # Time series
    fig = px.line(merged_df.sort_values('date'), x='date',
                  y=selected_feature, title=f"{selected_feature} Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # Correlation with SDP
    correlation = merged_df[[selected_feature, 'SDP']].corr().iloc[0, 1]
    st.metric("Correlation with SDP", f"{correlation:.4f}")

    # Scatter plot
    fig = px.scatter(merged_df, x=selected_feature, y='SDP', trendline="ols",
                     title=f"SDP vs {selected_feature}")
    st.plotly_chart(fig, use_container_width=True)

    # Correlation matrix
    st.subheader("Correlation Matrix")
    correlation_matrix = merged_df[feature_columns + ['SDP']].corr()
    fig = px.imshow(correlation_matrix, text_auto='.2f', aspect="auto")
    st.plotly_chart(fig, use_container_width=True)


# Page: Model Testing
if page == "Model Testing":
    st.header("Model Testing")

    # Model selection: allow the user to choose between Random Forest and Linear Regression.
    model_options = {
        "Random Forest": lambda: RandomForestRegressor(n_estimators=500, random_state=42),
        "Linear Regression": lambda: LinearRegression()
    }
    selected_model_name = st.selectbox(
        "Select Model", list(model_options.keys()))

    # Ensure the 'date' column is in datetime format and sort the DataFrame.
    merged_df['date'] = pd.to_datetime(merged_df['date_y'])
    merged_df = merged_df.sort_values('date')

    # Create a 'year' column if not already present.
    if 'year' not in merged_df.columns:
        merged_df['year'] = merged_df['date'].dt.year

    # Create a numeric month column if not already present.
    merged_df['month_int'] = merged_df['month']

    # Define feature columns (including month_int as a seasonal feature)
    feature_columns = [
        'light_sum', 'light_mean', 'light_std', 'light_max',
        'urban_area_ratio', 'brightness_urban', 'light_entropy',
        'light_contrast', 'light_energy', 'month_int'
    ]

    results = []

    # Select test dates: use every month from 2011 to 2022.
    test_dates = merged_df[(merged_df['year'] >= 2011) & (
        merged_df['year'] <= 2022)]['date'].unique()
    test_dates = np.sort(test_dates)

    st.info(
        f"Running walk‐forward CV using {selected_model_name} over {len(test_dates)} test dates.")
    with st.spinner("Running walk‐forward cross-validation..."):
        for test_date in test_dates:
            # Training set: all data strictly before the test date.
            train_data = merged_df[merged_df['date'] < test_date]
            # Test set: data exactly on the test date.
            test_data = merged_df[merged_df['date'] == test_date]

            if train_data.empty or test_data.empty:
                st.write(
                    f"Insufficient data for test date {pd.to_datetime(test_date).date()}. Skipping.")
                continue

            # Prepare features and target (SDP is assumed to be your target variable)
            X_train = train_data[feature_columns].values
            y_train = train_data['SDP'].values
            X_test = test_data[feature_columns].values
            y_test = test_data['SDP'].values

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Instantiate the selected model.
            model = model_options[selected_model_name]()
            model.fit(X_train_scaled, y_train)

            # Make predictions on the test set for the current month
            y_pred = model.predict(X_test_scaled)
            # In case multiple samples exist on the same test date, average the predictions.
            predicted = np.mean(y_pred)
            actual = np.mean(y_test)
            error = (predicted - actual) / actual

            results.append({
                'test_date': test_date,
                'actual_sdp': actual,
                'predicted_sdp': predicted,
                'error': error,
                'num_train_samples': len(train_data),
                'num_test_samples': len(test_data)
            })

        # Convert results to a DataFrame and display.
        results_df = pd.DataFrame(results)
        st.subheader("Walk‐Forward Cross Validation Results")
        st.dataframe(results_df)


        # Plot predicted vs. actual SDP values over the test dates using Plotly.
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['test_date'],
            y=results_df['actual_sdp'],
            mode='lines+markers',
            name='Actual SDP'
        ))
        fig.add_trace(go.Scatter(
            x=results_df['test_date'],
            y=results_df['predicted_sdp'],
            mode='lines+markers',
            name='Predicted SDP'
        ))
        fig.update_layout(
            title='Delhi SDP Predictions (Walk‐Forward CV for Each Month)',
            xaxis_title='Test Date',
            yaxis_title='SDP',
            legend_title='Legend'
        )
        st.plotly_chart(fig)

        # Calculate additional metrics: MAPE and R².
        if not results_df.empty:
            mape = mean_absolute_percentage_error(
                results_df['actual_sdp'], results_df['predicted_sdp']) * 100
            r2 = r2_score(results_df['actual_sdp'],
                          results_df['predicted_sdp'])
            st.write(f"**MAPE:** {mape:.2f}%")
            st.write(f"**R²:** {r2:.2f}")

    X = merged_df[feature_columns].values
    y = merged_df['SDP'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    final_model = model_options[selected_model_name]()
    final_model.fit(X_scaled, y)

    # Check for feature importances (RandomForest) or coefficients (LinearRegression)
    if hasattr(final_model, 'feature_importances_'):
        importances = final_model.feature_importances_
    elif hasattr(final_model, 'coef_'):
        # For linear regression, take absolute values of coefficients to reflect importance
        importances = np.abs(final_model.coef_)
    else:
        st.write("The selected model does not support feature importance extraction.")
        importances = None

    # Plot the feature importances if available
    if importances is not None:
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        st.subheader("Feature Importances")
        st.bar_chart(importance_df.set_index('Feature'))
