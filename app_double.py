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
st.set_page_config(
    page_title="Nighttime Lights Economic Analysis", layout="wide")

# Add a title
st.title("Nighttime Lights & Economic Analysis")

# Function to load data


@st.cache_data
def load_data(region):
    if region == "Delhi":
        # Annual SDP data for Delhi
        sdp_data = {
            'Year': [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
            'GDP': [343798, 391388, 443960, 494803, 550804, 616085, 677900, 738389, 792911, 763435, 904642, 1043759]
        }
        econ_df = pd.DataFrame(sdp_data)
        econ_df['date'] = pd.to_datetime(
            econ_df['Year'].astype(str) + '-01-01')
        econ_df = econ_df.set_index('date')
        economic_variable = 'GDP'

    elif region == "Singapore":
        # Load Singapore GDP data from CSV
        try:
            # Try to read the CSV file
            sg_gdp_file = "Singapore_GDP_Singstat.csv"
            if os.path.exists(sg_gdp_file):
                gdp_data = pd.read_csv(sg_gdp_file)

                # Process the data (assuming structure from the prompt)
                # Extracting the GDP row
                gdp_row = gdp_data[gdp_data['Data Series']
                                   == 'GDP At Current Market Prices']

                if gdp_row.empty:
                    st.error("GDP data not found in the CSV file")
                    return None, None

                # Reshape the data from wide to long format
                years = []
                gdp_values = []

                for col in gdp_row.columns:
                    if col != 'Data Series':
                        try:
                            year = int(col.strip())
                            value = gdp_row[col].values[0]
                            # Handle special characters in the value
                            if isinstance(value, str):
                                value = value.replace(
                                    '*', '').replace('**', '')
                                try:
                                    value = float(value)
                                except:
                                    continue
                            years.append(year)
                            gdp_values.append(value)
                        except:
                            continue

                # Create a new DataFrame with the cleaned data
                gdp_cleaned = pd.DataFrame({
                    'Year': years,
                    'GDP': gdp_values
                })

                # Filter to only include years from 2012 to 2024 (matching nightlight data)
                gdp_cleaned = gdp_cleaned[(gdp_cleaned['Year'] >= 2012) & (
                    gdp_cleaned['Year'] <= 2024)]

                # Convert to datetime and set as index
                gdp_cleaned['date'] = pd.to_datetime(
                    gdp_cleaned['Year'].astype(str) + '-01-01')
                econ_df = gdp_cleaned.set_index('date')
                economic_variable = 'GDP'
        except Exception as e:
            st.error(f"Error loading Singapore GDP data: {e}")
            return None, None
    else:
        st.error(f"Unknown region: {region}")
        return None, None

    # Monthly interpolation
    monthly_dates = pd.date_range(
        start=econ_df.index.min(), end='2024-01-01', freq='MS')
    monthly_econ = econ_df.reindex(monthly_dates).interpolate(method='linear')
    monthly_econ = monthly_econ.reset_index().rename(columns={'index': 'date'})
    monthly_econ['year'] = monthly_econ['date'].dt.year
    monthly_econ['month'] = monthly_econ['date'].dt.month

    # Convert years to integers to remove decimal places
    monthly_econ['year'] = monthly_econ['year'].astype(int)

    return econ_df, monthly_econ, economic_variable

# Function to extract features from TIF images


def extract_features_from_tif(image_path):
    try:
        with rasterio.open(image_path) as src:
            img = src.read(1)
            profile = src.profile
    except Exception as e:
        st.error(f"Error reading image {image_path}: {e}")
        return None, None, None

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
    try:
        img_uint8 = img.astype(np.uint8)
        if img_uint8.max() == 0:
            img_uint8[0, 0] = 1
        glcm = graycomatrix(img_uint8, distances=[1], angles=[
                            0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
    except Exception as e:
        st.warning(f"Error calculating texture features: {e}")
        contrast = 0
        energy = 0

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
def process_images(image_dir, region="Delhi"):
    features_list = []
    image_files = []

    if not os.path.exists(image_dir):
        st.error(f"Directory {image_dir} does not exist")
        return pd.DataFrame(), []

    file_count = 0
    for fname in os.listdir(image_dir):
        if fname.endswith('.tif'):
            # Different filename parsing for different regions
            if region == "Delhi":
                parts = fname.split('_')
                if len(parts) >= 5:
                    try:
                        year = int(parts[3])
                        month = int(parts[4].split('.')[0])
                    except ValueError:
                        continue
            elif region == "Singapore":
                # Adapt the filename parsing for Singapore data
                # Assuming format similar to Delhi but might need adjustment
                parts = fname.split('_')
                if len(parts) >= 5:
                    try:
                        year = int(parts[3])
                        month = int(parts[4].split('.')[0])
                    except ValueError:
                        try:
                            # Alternative parsing if needed
                            # This is just an example - adjust based on actual filenames
                            date_part = parts[-2] + parts[-1].split('.')[0]
                            year = int(date_part[:4])
                            month = int(date_part[4:6])
                        except:
                            continue

            image_path = os.path.join(image_dir, fname)
            image_files.append(
                {"path": image_path, "year": year, "month": month, "filename": fname})

            feats, img, profile = extract_features_from_tif(image_path)
            if feats:
                feats['year'] = year
                feats['month'] = month
                feats['date'] = pd.Timestamp(year=year, month=month, day=1)
                feats['image_path'] = image_path
                features_list.append(feats)
                file_count += 1

    st.info(f"Processed {file_count} image files from {image_dir}")
    return pd.DataFrame(features_list), image_files


# Region selection
region = st.sidebar.radio("Select Region", ["Delhi", "Singapore"])

# Load economic data
econ_df, monthly_econ, economic_variable = load_data(region)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", [
                        "Data Explorer", "Image Viewer", "Feature Analysis", "Model Testing"])

# Set directory path
if region == "Delhi":
    default_image_dir = "./Delhi_Nighttime_Lights"
else:
    default_image_dir = "./Singapore_Nighttime_Lights"

image_dir = st.sidebar.text_input(
    "TIF Images Directory", default_image_dir)

# Data source options
data_source = st.sidebar.radio(
    "Data Source", ["Process Images Directly", "Load from Pickle File"], index=0)

if data_source == "Load from Pickle File":
    pickle_path = st.sidebar.text_input(
        f"Path to Pickle File of Extracted Features DataFrame",
        f"{region.lower()}_nightlights_features.pkl")

    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                features_df = pickle.load(f)
            st.sidebar.success(
                f"Loaded {len(features_df)} records from pickle file")

            if 'image_path' in features_df.columns:
                image_files = [
                    {"path": path, "year": year, "month": month,
                        "filename": os.path.basename(path)}
                    for path, year, month in zip(features_df['image_path'], features_df['year'], features_df['month'])
                ]
        except Exception as e:
            st.sidebar.error(f"Error loading pickle file: {e}")
            st.error("Failed to load pickle file. Processing images instead.")
            features_df, image_files = process_images(image_dir, region)
    else:
        st.sidebar.warning(
            f"Pickle file not found. Processing images instead.")
        features_df, image_files = process_images(image_dir, region)
else:
    features_df, image_files = process_images(image_dir, region)

    # Save to pickle for future use
    save_pickle = st.sidebar.checkbox(
        "Save features to pickle file", value=True)
    if save_pickle and not features_df.empty:
        pickle_name = f"{region.lower()}_nightlights_features.pkl"
        with open(pickle_name, 'wb') as f:
            pickle.dump(features_df, f)
        st.sidebar.success(f"Saved features to {pickle_name}")

# Check for data
if features_df.empty:
    st.error(
        f"No data available for {region}. Please check the image directory.")
    st.stop()

if econ_df is None or monthly_econ is None:
    st.error(f"No economic data available for {region}.")
    st.stop()

# Merge image features with monthly economic data
merged_df = pd.merge(features_df, monthly_econ, on=[
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

    # Economic Data
    st.subheader(f"{region} {economic_variable} Data")
    econ_df_display = econ_df.reset_index()
    st.dataframe(econ_df_display)

    # Add "lazy student" analytics on GDP data
    st.subheader("GDP Data Analysis - Quick Stats")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Annual Growth Rate",
                  f"{((econ_df['GDP'].iloc[-1] / econ_df['GDP'].iloc[0]) ** (1 / (len(econ_df) - 1)) - 1) * 100:.2f}%")
    with col2:
        st.metric("Min GDP", f"{econ_df['GDP'].min()}")
    with col3:
        st.metric("Max GDP", f"{econ_df['GDP'].max()}")
    with col4:
        st.metric("Avg GDP", f"{econ_df['GDP'].mean():.0f}")
    
    # Economic trend plot
    st.subheader(f"Interactive {economic_variable} Trend")
    fig = px.line(econ_df.reset_index(), x='date', y=economic_variable,
                  title=f"{region} {economic_variable} Trend", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    # Monthly data
    st.subheader("Monthly Data (Interpolated)")
    st.dataframe(monthly_econ)

    # Download button for monthly data
    csv = monthly_econ.to_csv(index=False)
    st.download_button(
        label="Download Monthly Data as CSV",
        data=csv,
        file_name=f"{region.lower()}_monthly_data.csv",
        mime="text/csv"
    )

    # Box plot for brightness over years
    st.subheader("Light Brightness Distribution by Year")
    if not features_df.empty and 'light_mean' in features_df.columns and 'year' in features_df.columns:
        try:
            fig = px.box(features_df, x='year', y='light_mean',
                         title=f"Change in Brightness Over Years ({region})")
            fig.update_layout(xaxis_title="Year",
                              yaxis_title="Mean Light Brightness")
            st.plotly_chart(fig, use_container_width=True)

            # Add a text box explaining what the boxplot shows
            st.info("This boxplot shows how the distribution of nighttime light brightness has changed over the years. The box represents the middle 50% of values, the line in the box is the median, and the whiskers show the range of non-outlier values.")
        except Exception as e:
            st.error(f"Error creating boxplot: {e}")

    # Features data
    st.subheader("Features Data")
    st.dataframe(features_df)

    # Merged data
    st.subheader("Merged Data")
    st.dataframe(merged_df)

# Page: Image Viewer
elif page == "Image Viewer":
    st.header(f"{region} Nighttime Light Image Viewer")

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
                                 title=f"{region} - {current_date.strftime('%B %Y')}")
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

                            # Economic value
                            econ_value = merged_df[
                                (merged_df['year'] == current_date.year) &
                                (merged_df['month'] == current_date.month)
                            ][economic_variable].values[0]

                            st.subheader(economic_variable)
                            st.metric(f"{economic_variable} Value",
                                      f"{econ_value:.2f}")

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
                         title=f"{region} - {selected_date.strftime('%B %Y')}")
                    plt.colorbar(ax.images[0], ax=ax, label="Light Intensity")
                    st.pyplot(fig)

                    # Feature trend
                    st.subheader("Feature Trend")
                    feature_options = [col for col in merged_df.columns if col in [
                        'light_mean', 'urban_area_ratio', 'brightness_urban', economic_variable
                    ]]
                    selected_feature = st.selectbox(
                        "Select Feature", feature_options)
                    merged_df['date'] = merged_df['date_y']
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

                    # Economic value
                    econ_value = merged_df[
                        (merged_df['year'] == selected_year) &
                        (merged_df['month'] == selected_month)
                    ][economic_variable].values[0]

                    st.subheader(f"{economic_variable} Information")
                    st.metric(f"{economic_variable} Value",
                              f"{econ_value:.2f}")

                    # Month-over-month comparison
                    if selected_index > 0:
                        previous_date = valid_dates[selected_index - 1]
                        previous_econ = merged_df[
                            (merged_df['year'] == previous_date.year) &
                            (merged_df['month'] == previous_date.month)
                        ][economic_variable].values[0]

                        econ_change = (
                            (econ_value - previous_econ) / previous_econ) * 100
                        st.metric("Monthly Change", f"{econ_change:.2f}%")

                    # Histogram
                    st.subheader("Pixel Distribution")
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.hist(img.flatten(), bins=30, alpha=0.7, color='teal')
                    st.pyplot(fig)

                # Compare with previous/next
                st.subheader("Previous/Next Comparison")
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

    # Ensure date column is available
    if 'date_y' in merged_df.columns:
        merged_df['date'] = merged_df['date_y']

    # Time series
    fig = px.line(merged_df.sort_values('date'), x='date',
                  y=selected_feature, title=f"{selected_feature} Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # Correlation with economic variable
    correlation = merged_df[[selected_feature,
                             economic_variable]].corr().iloc[0, 1]
    st.metric(f"Correlation with {economic_variable}", f"{correlation:.4f}")

    # Scatter plot
    fig = px.scatter(merged_df, x=selected_feature, y=economic_variable, trendline="ols",
                     title=f"{economic_variable} vs {selected_feature}")
    st.plotly_chart(fig, use_container_width=True)

    # Correlation matrix
    st.subheader("Correlation Matrix")
    correlation_matrix = merged_df[feature_columns +
                                   [economic_variable]].corr()
    fig = px.imshow(correlation_matrix, text_auto='.2f', aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

# Page: Model Testing
elif page == "Model Testing":
    st.header("Model Testing")

    # Model selection
    model_options = {
        "Random Forest": lambda: RandomForestRegressor(n_estimators=500, random_state=42),
        "Linear Regression": lambda: LinearRegression()
    }
    selected_model_name = st.selectbox(
        "Select Model", list(model_options.keys()))

    # Ensure the 'date' column is in datetime format and sort the DataFrame
    if 'date_y' in merged_df.columns:
        merged_df['date'] = pd.to_datetime(merged_df['date_y'])
    merged_df = merged_df.sort_values('date')

    # Create a 'year' column if not already present
    if 'year' not in merged_df.columns:
        merged_df['year'] = merged_df['date'].dt.year

    # Create a numeric month column if not already present
    merged_df['month_int'] = merged_df['month']

    # Define feature columns (including month_int as a seasonal feature)
    feature_columns = [
        'light_sum', 'light_mean', 'light_std', 'light_max',
        'urban_area_ratio', 'brightness_urban', 'light_entropy',
        'light_contrast', 'light_energy', 'month_int'
    ]

    results = []

    # Select test dates: use all available dates for the region
    min_year = merged_df['year'].min()
    max_year = merged_df['year'].max()
    test_dates = merged_df[(merged_df['year'] >= min_year) & (
        merged_df['year'] <= max_year)]['date'].unique()
    test_dates = np.sort(test_dates)

    st.info(
        f"Running walk‐forward CV using {selected_model_name} over {len(test_dates)} test dates for {region}.")

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
            y_train = train_data['GDP'].values
            X_test = test_data[feature_columns].values
            y_test = test_data['GDP'].values

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
                'actual_gdp': actual,
                'predicted_gdp': predicted,
                'error': error,
                'num_train_samples': len(train_data),
                'num_test_samples': len(test_data)
            })

        # Convert results to a DataFrame and display.
        results_df = pd.DataFrame(results)
        st.subheader("Walk‐Forward Cross Validation Results")
        st.dataframe(results_df)

        # Plot predicted vs. actual GDP values over the test dates using Plotly.
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['test_date'],
            y=results_df['actual_gdp'],
            mode='lines+markers',
            name='Actual GDP'
        ))
        fig.add_trace(go.Scatter(
            x=results_df['test_date'],
            y=results_df['predicted_gdp'],
            mode='lines+markers',
            name='Predicted GDP'
        ))
        fig.update_layout(
            title='GDP Predictions (Walk‐Forward CV for Each Month)',
            xaxis_title='Test Date',
            yaxis_title='GDP',
            legend_title='Legend'
        )
        st.plotly_chart(fig)

        # Calculate additional metrics: MAPE and R².
        if not results_df.empty:
            mape = mean_absolute_percentage_error(
                results_df['actual_gdp'], results_df['predicted_gdp']) * 100
            r2 = r2_score(results_df['actual_gdp'],
                          results_df['predicted_gdp'])
            st.write(f"**MAPE:** {mape:.2f}%")
            st.write(f"**R²:** {r2:.2f}")

    X = merged_df[feature_columns].values
    y = merged_df['GDP'].values
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
