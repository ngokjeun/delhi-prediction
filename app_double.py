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
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import ProbPlot
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import rasterio
from skimage.transform import resize
import numpy as np
import pandas as pd
import os
import pickle
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats


from sklearn.linear_model import LinearRegression
import streamlit as st


import os
import pickle
from datetime import datetime
from skimage.feature import graycomatrix, graycoprops
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



st.set_page_config(
    page_title="Nighttime Lights Analysis", layout="wide")

st.title("Nighttime Lights Analysis")



@st.cache_data
def load_data(region):
    if region == "Delhi":
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
        sg_gdp_file = "Singapore_GDP_Singstat.csv"
        if os.path.exists(sg_gdp_file):
            gdp_data = pd.read_csv(sg_gdp_file)





            gdp_row = gdp_data[gdp_data['Data Series']
                                == 'GDP At Current Market Prices']

            if gdp_row.empty:
                st.error("GDP data not found in the CSV file")
                return None, None
            years = []
            gdp_values = []

            for col in gdp_row.columns:
                if col != 'Data Series':
                    try:
                        year = int(col.strip())
                        value = gdp_row[col].values[0]
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
            gdp_cleaned = pd.DataFrame({
                'Year': years,
                'GDP': gdp_values
            })

            gdp_cleaned = gdp_cleaned[(gdp_cleaned['Year'] >= 2012) & (
                gdp_cleaned['Year'] <= 2024)]

            gdp_cleaned['date'] = pd.to_datetime(
                gdp_cleaned['Year'].astype(str) + '-01-01')
            econ_df = gdp_cleaned.set_index('date')
            economic_variable = 'GDP'

    monthly_dates = pd.date_range(
        start=econ_df.index.min(), end='2024-01-01', freq='MS')
    monthly_econ = econ_df.reindex(monthly_dates).interpolate(method='linear')
    monthly_econ = monthly_econ.reset_index().rename(columns={'index': 'date'})
    monthly_econ['year'] = monthly_econ['date'].dt.year
    monthly_econ['month'] = monthly_econ['date'].dt.month


    monthly_econ['year'] = monthly_econ['year'].astype(int)

    return econ_df, monthly_econ, economic_variable

# raster
def extract_features_from_tif(image_path):
    try:
        with rasterio.open(image_path) as src:
            img = src.read(1)
            profile = src.profile
    except Exception as e:
        st.error(f"Error reading image {image_path}: {e}")
        return None, None, None
    
    # mask bug fixes dont remove
    if np.ma.is_masked(img):
        img = img.filled(fill_value=0)
    img = np.nan_to_num(img, nan=0)

    light_mean = np.nanmean(img)
    light_std = np.nanstd(img)
    light_max = np.nanmax(img)
    threshold = 30 # maybe make configured by user
    urban_area_ratio = (img > threshold).sum() / img.size
    brightness_urban = light_mean * urban_area_ratio


    hist, _ = np.histogram(img.astype(np.uint8), bins=256, range=(0, 256))
    if hist.sum() > 0:
        hist = hist / hist.sum()
    non_zero = hist[hist > 0]
    entropy = -np.sum(non_zero * np.log2(non_zero)) if non_zero.size > 0 else 0
    try:
        img_uint8 = img.astype(np.uint8)
        if img_uint8.max() == 0:
            img_uint8[0, 0] = 1
        glcm = graycomatrix(img_uint8, distances=[1], angles=[
                            0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
    except Exception as e:
        st.write("error", e)
        contrast = 0
        energy = 0

    return {
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
            if region == "Delhi":
                parts = fname.split('_')
                if len(parts) >= 5:
                    try:
                        year = int(parts[3])
                        month = int(parts[4].split('.')[0])
                    except ValueError:
                        continue
            elif region == "Singapore":
                parts = fname.split('_')
                if len(parts) >= 5:
                    try:
                        year = int(parts[3])
                        month = int(parts[4].split('.')[0])
                    except ValueError:
                        try:
                            date_part = parts[-2] + parts[-1].split('.')[0]
                            year = int(date_part[:4])
                            month = int(date_part[4:6])
                        except:
                            continue

            image_path = os.path.join(image_dir, fname)
            image_files.append(
                {"path": image_path, "year": year, "month": month, "filename": fname})

            f, _ , _ = extract_features_from_tif(image_path)
            if f:
                f['year'] = year
                f['month'] = month
                f['date'] = pd.Timestamp(year=year, month=month, day=1)
                f['image_path'] = image_path
                features_list.append(f)
                file_count += 1

    st.write(f"using {file_count} images at {image_dir}")
    return pd.DataFrame(features_list), image_files



region = st.sidebar.radio("Select Region", ["Delhi", "Singapore"])
econ_df, monthly_econ, economic_variable = load_data(region) #GDP
st.sidebar.title("Dashboard")
page = st.sidebar.radio("Choose", [
                        "Data Explorer", "Image Viewer", "Extracted features", "Model Testing"])

if region == "Delhi":
    default_image_dir = "./Delhi_Nighttime_Lights"
else:
    default_image_dir = "./Singapore_Nighttime_Lights"

image_dir = st.sidebar.text_input(
    "TIF Images Directory", default_image_dir)

data_source = st.sidebar.radio(
    "Data Source", ["Process Images", "Pickle"], index=1)

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
    save_pickle = st.sidebar.checkbox(
        "Save features to pickle file", value=True)
    if save_pickle and not features_df.empty:
        pickle_name = f"{region.lower()}_nightlights_features.pkl"
        with open(pickle_name, 'wb') as f:
            pickle.dump(features_df, f)
        st.sidebar.success(f"Saved features to {pickle_name}")


if features_df.empty:
    st.error(
        f"No data available for {region}. Please check the image directory.")
    st.stop()

if econ_df is None or monthly_econ is None:
    st.error(f"No economic data available for {region}.")
    st.stop()
merged_df = pd.merge(features_df, monthly_econ, on=[
                     'year', 'month'], how='inner')
merged_df['month_int'] = merged_df['month']


month_dummies = pd.get_dummies(
    merged_df['month'], prefix='month', drop_first=True)
merged_df = pd.concat([merged_df, month_dummies], axis=1)
month_cols = [col for col in merged_df.columns if col.startswith('month_')]
feature_columns = [
    'light_mean', 'light_std', 'light_max',
    'urban_area_ratio', 'brightness_urban', 'light_entropy',
    'light_contrast', 'light_energy'
] + month_cols


df_temp = merged_df.copy()
df_temp['year_prev'] = df_temp['year'] - 1
df_temp['key'] = df_temp['year'].astype(
    str) + '-' + df_temp['month'].astype(str)
df_temp['key_prev'] = df_temp['year_prev'].astype(
    str) + '-' + df_temp['month'].astype(str)

prev_vals = {}
for _, row in df_temp.iterrows():
    k = row['key']
    for feat in ['light_mean', 'light_std', 'light_max', 'urban_area_ratio', 'brightness_urban']:
        prev_vals[f"{k}_{feat}"] = row[feat]

for idx, row in df_temp.iterrows():
    k_prev = row['key_prev']
    for feat in ['light_mean', 'light_std', 'light_max', 'urban_area_ratio', 'brightness_urban']:
        prev_key = f"{k_prev}_{feat}"
        if prev_key in prev_vals:
            yoy_val = ((row[feat] - prev_vals[prev_key]) / prev_vals[prev_key] * 100
                       if prev_vals[prev_key] != 0 else np.nan)
            df_temp.loc[idx, f"{feat}_yoy"] = yoy_val

yoy_feats = [f"{feat}_yoy" for feat in [
    'light_mean', 'light_std', 'urban_area_ratio']]
feature_columns.extend([f for f in yoy_feats if f in df_temp.columns])
merged_df = df_temp.copy()
merged_df = merged_df.fillna(method="ffill")

if page == "Data Explorer":
    st.header("Data Explorer")
    st.subheader(f"{region} {economic_variable} Data")
    econ_df_display = econ_df.reset_index()
    st.dataframe(econ_df_display)
    st.subheader("GDP Data Analysis")
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
    
    st.subheader(f"Interactive {economic_variable} Trend")
    fig = px.line(econ_df.reset_index(), x='date', y=economic_variable,
                  title=f"{region} {economic_variable} Trend", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Data (Interpolated)")
    st.dataframe(monthly_econ[['date', 'GDP', 'year', 'month']])
    st.subheader("Light Brightness Distribution by Year")
    if not features_df.empty and 'light_mean' in features_df.columns and 'year' in features_df.columns:
        try:
            fig = px.box(features_df, x='year', y='light_mean',
                         title=f"Change in Brightness Over Years ({region})")
            fig.update_layout(xaxis_title="Year",
                              yaxis_title="Mean Light Brightness")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error creating boxplot: {e}")

    st.subheader("Features Data")
    st.dataframe(features_df)
    st.subheader("Merged Data")
    st.dataframe(merged_df)

elif page == "Image Viewer":
    st.header(f"{region} Nighttime Light Image Viewer")

    if 'image_path' not in merged_df.columns:
        st.error("Image paths not available.")
        st.stop()
    has_valid_paths = any(os.path.exists(path)
                          for path in merged_df['image_path'].unique())

    if not has_valid_paths:
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
        st.subheader("Explore Images Over Time")
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
        selected_index = st.slider(
            "Select Date", min_value=0, max_value=len(valid_dates)-1, value=0)
        selected_date = valid_dates[selected_index]
        st.info(f"Viewing: {selected_date.strftime('%B %Y')}")

        auto_play = st.checkbox("Auto-play")
        if auto_play:
            play_speed = st.slider("Speed (seconds)", 0.2, 5.0, 0.3, 0.5)

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
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            fig, ax = plt.subplots(figsize=(10, 8))
                            show(img, ax=ax, cmap='viridis',
                                 title=f"{region} - {current_date.strftime('%B %Y')}")
                            plt.colorbar(
                                ax.images[0], ax=ax, label="Light Intensity")
                            st.pyplot(fig)

                        with col2:
                            st.subheader("Features")
                            for key, value in feats.items():
                                if key not in ['year', 'month', 'date', 'image_path']:
                                    st.metric(key, f"{value:.2f}" if isinstance(
                                        value, float) else value)
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
                col1, col2 = st.columns([3, 1])

                with col1:

                    fig, ax = plt.subplots(figsize=(10, 8))
                    show(img, ax=ax, cmap='viridis',
                         title=f"{region} - {selected_date.strftime('%B %Y')}")
                    plt.colorbar(ax.images[0], ax=ax, label="Light Intensity")
                    st.pyplot(fig)


                    st.subheader("Feature Trend")
                    feature_options = [col for col in merged_df.columns]
                    selected_feature = st.selectbox(
                        "Select Feature", feature_options)
                    merged_df['date'] = merged_df['date_y']
                    time_data = merged_df.sort_values('date')
                    fig = px.line(time_data, x='date', y=selected_feature,
                                  title=f"{selected_feature} Over Time")
                    fig.add_vline(x=selected_date,
                                  line_color="red", line_width=2)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Image Features")
                    for key, value in feats.items():
                        if key not in ['year', 'month', 'date', 'image_path']:
                            st.metric(key, f"{value:.2f}" if isinstance(
                                value, float) else value)


                    econ_value = merged_df[
                        (merged_df['year'] == selected_year) &
                        (merged_df['month'] == selected_month)
                    ][economic_variable].values[0]

                    st.subheader(f"{economic_variable} Information")
                    st.metric(f"{economic_variable} Value",
                              f"{econ_value:.2f}")

                    if selected_index > 0:
                        previous_date = valid_dates[selected_index - 1]
                        previous_econ = merged_df[
                            (merged_df['year'] == previous_date.year) &
                            (merged_df['month'] == previous_date.month)
                        ][economic_variable].values[0]

                        econ_change = (
                            (econ_value - previous_econ) / previous_econ) * 100
                        st.metric("Monthly Change", f"{econ_change:.2f}%")

                st.subheader("Previous/Next Comparison")
                n_months = st.radio("Choose date comparison:", ["MoM", "YoY"])
                n_months = 1 if n_months == "MoM" else 12
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if selected_index > 0:
                        previous_date = valid_dates[selected_index - n_months]
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
                        next_date = valid_dates[selected_index + n_months]
                        next_image = date_to_path[next_date]

                        if os.path.exists(next_image):
                            _, next_img, _ = extract_features_from_tif(
                                next_image)

                            fig, ax = plt.subplots(figsize=(5, 4))
                            show(next_img, ax=ax, cmap='viridis',
                                 title=f"Next: {next_date.strftime('%b %Y')}")
                            st.pyplot(fig)

                with col2:
                    if selected_index > 0:
                        previous_date = valid_dates[selected_index - n_months]
                        previous_image = date_to_path[previous_date]

                        if os.path.exists(previous_image):
                            _, prev_img, _ = extract_features_from_tif(
                                previous_image)
                            diff_img = img.astype(
                                float) - prev_img.astype(float)
                            vmax = max(abs(diff_img.min()),
                                       abs(diff_img.max()))
                            vmin = -vmax
                            fig, ax = plt.subplots(figsize=(8, 6))
                            im = ax.imshow(
                                diff_img, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                            ax.set_title(
                                f"Change: {previous_date.strftime('%b %Y')} to {selected_date.strftime('%b %Y')}")
                            plt.colorbar(im, ax=ax, label="Light Change")
                            st.pyplot(fig)

elif page == "Extracted features":
    st.header("Extracted features")

    log_transform = True


    selected_feature = st.selectbox("Select Feature", [feature for feature in feature_columns if feature not in month_cols])
    if 'date_y' in merged_df.columns:
        merged_df['date'] = merged_df['date_y']

    df_plot = merged_df.copy()
    if log_transform:
        df_plot[economic_variable] = np.log(df_plot[economic_variable])
        y_label = f"Log({economic_variable})"
    else:
        y_label = economic_variable

    st.subheader(f"{selected_feature} Over Time")
    time_data = df_plot.sort_values('date')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_data['date'],
        y=time_data[selected_feature],
        mode='lines+markers',
        name=selected_feature
    ))

    window_size = max(3, len(time_data) // 4)
    rolling_mean = time_data[selected_feature].rolling(
        window=window_size, center=True).mean()
    rolling_std = time_data[selected_feature].rolling(
        window=window_size, center=True).std()

    fig.add_trace(go.Scatter(
        x=time_data['date'],
        y=rolling_mean + rolling_std,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=time_data['date'],
        y=rolling_mean - rolling_std,
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(68, 68, 168, 0.3)',
        fill='tonexty',
        name='± 1 Std Dev'
    ))

    fig.update_layout(
        title=f"{selected_feature} Over Time with Confidence Bands",
        xaxis_title="Date",
        yaxis_title=selected_feature,
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    correlation = df_plot[[selected_feature,
                           economic_variable]].corr().iloc[0, 1]
    st.metric(f"Correlation with {y_label}", f"{correlation:.4f}")

    corr_pvalue = stats.pearsonr(
        df_plot[selected_feature].values, df_plot[economic_variable].values)[1]
    st.metric("P-value", f"{corr_pvalue:.4f}",
              delta="Statistically significant" if corr_pvalue < 0.05 else "Not statistically significant")
    st.subheader(f"Regression Analysis")

    @st.cache_data
    def fit_multiple_regression(data, features, target):
        X_multi = data[features].copy()
        X_multi = sm.add_constant(X_multi)
        X_multi = X_multi.apply(pd.to_numeric, errors='coerce')
        X_multi.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_multi.dropna(inplace=True)
        y_multi = data.loc[X_multi.index, target]

        model_multi = sm.OLS(y_multi, X_multi).fit()
        return model_multi, X_multi, y_multi

    model_multi, X_multi, y_multi = fit_multiple_regression(
        merged_df, [
            col for col in feature_columns if col not in month_cols], economic_variable)
    st.subheader("Multiple Regression Model Summary")
    st.code(model_multi.summary().as_text())
    st.subheader("Multicollinearity Analysis")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_multi.columns
    vif_data["VIF"] = [variance_inflation_factor(X_multi.values, i)
                    for i in range(X_multi.shape[1])]

    vif_data.sort_values(by='VIF', ascending=False, inplace=True)
    fig_vif = go.Figure(go.Bar(
        x=vif_data['Feature'],
        y=vif_data['VIF'],
        marker_color=[('red' if vif > 10 else 'orange' if vif >
                    5 else 'green') for vif in vif_data['VIF']]
    ))

    fig_vif.update_layout(
        title="Variance Inflation Factor (VIF)",
        xaxis_title="Features",
        yaxis_title="VIF (log scale)",
        yaxis_type="log",
        shapes=[
            dict(type='line', y0=5, y1=5, x0=-0.5, x1=len(vif_data['Feature'])-0.5,
                line=dict(color='orange', dash='dash')),
            dict(type='line', y0=10, y1=10, x0=-0.5, x1=len(vif_data['Feature'])-0.5,
                line=dict(color='red', dash='dash'))
        ]
    )

    st.subheader("Correlation Matrix")
    corr_features = ['GDP'] + [col for col in feature_columns if col != 'GDP']
    corr_matrix = merged_df[corr_features].corr()

    # Extract only the GDP row
    gdp_corr_row = corr_matrix.loc['GDP']

    fig_corr = go.Figure(data=go.Heatmap(
        z=[gdp_corr_row.values],
        x=gdp_corr_row.index,
        y=['GDP'],
        colorscale='Viridis',
        colorbar=dict(title="Correlation"),
        text=[gdp_corr_row.values],
        texttemplate="%{text:.2f}"  # Show data labels
    ))

    fig_corr.update_layout(
        xaxis_nticks=len(corr_features),
        title="Correlation of Features with GDP"
    )

    st.plotly_chart(fig_corr, use_container_width=True)


elif page == "Model Testing":
    st.header("Model Testing")
    model_options = {
        "Random Forest": lambda: RandomForestRegressor(n_estimators=500, random_state=42, max_depth=10, min_samples_split=2),
        "Linear Regression": lambda: LinearRegression()
    }
    selected_model_name = st.selectbox(
        "Select Model", list(model_options.keys()))

    if 'date_y' in merged_df.columns:
        merged_df['date'] = pd.to_datetime(merged_df['date_y'])
    merged_df = merged_df.sort_values('date')
    if 'year' not in merged_df.columns:
        merged_df['year'] = merged_df['date'].dt.year
    merged_df['month_int'] = merged_df['month']

    feature_columns = [
        'light_mean', 'light_std', 'light_max',
        'urban_area_ratio', 'brightness_urban', 'light_entropy',
        'light_contrast', 'light_energy'
    ] + month_cols

    results = []
    min_year = merged_df['year'].min()
    max_year = merged_df['year'].max()
    test_dates = merged_df[(merged_df['year'] >= min_year) & (
        merged_df['year'] <= max_year)]['date'].unique()
    test_dates = np.sort(test_dates)
    log_transform = True

    with st.spinner("Running CV..."):
        for test_date in test_dates:
            # st.write(test_date)
            train_data = merged_df[merged_df['date'] < test_date]
            test_data = merged_df[merged_df['date'] == test_date]

            if train_data.empty or test_data.empty:
                continue
            # st.write(train_data.shape, test_data.shape)
            X_train = train_data[feature_columns].values
            X_test = test_data[feature_columns].values

            if log_transform:
                y_train = np.log(train_data['GDP'].values)
                y_test = np.log(test_data['GDP'].values)
            else:
                y_train = train_data['GDP'].values
                y_test = test_data['GDP'].values

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = model_options[selected_model_name]()
            model.fit(X_train_scaled, y_train)

            y_pred_transformed = model.predict(X_test_scaled)

            if log_transform:
                y_pred = np.exp(y_pred_transformed)
                actual = np.exp(np.mean(y_test))
            else:
                y_pred = y_pred_transformed
                actual = np.mean(y_test)

            predicted = np.mean(y_pred)
            error = (predicted - actual) / actual

            results.append({
                'test_date': test_date,
                'actual_gdp': actual,
                'predicted_gdp': predicted,
                'error': error,
                'samples': len(train_data),
                'actual_t': np.mean(y_test),
                'predicted_t': np.mean(y_pred_transformed),
            })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        t_val = stats.t.ppf(0.975, df=len(results_df)-2)

        if len(results_df) > 2:
            if log_transform:
                t_std = np.std(results_df['actual_t'] - results_df['predicted_t'])
                results_df['ci_l'] = np.exp(
                    results_df['predicted_t'] - t_val * t_std)
                results_df['ci_u'] = np.exp(
                    results_df['predicted_t'] + t_val * t_std)
            else:
                pred_std = np.std(
                    results_df['actual_gdp'] - results_df['predicted_gdp'])
                results_df['ci_l'] = results_df['predicted_gdp'] - t_val * pred_std
                results_df['ci_u'] = results_df['predicted_gdp'] + t_val * pred_std

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['test_date'], y=results_df['actual_gdp'], mode='lines+markers', name='Actual'))
        fig.add_trace(go.Scatter(x=results_df['test_date'], y=results_df['predicted_gdp'],
                    mode='lines+markers', name='Predicted', line=dict(color='red')))

        if 'ci_l' in results_df.columns:
            fig.add_trace(go.Scatter(x=results_df['test_date'], y=results_df['ci_u'], mode='lines', line=dict(
                width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=results_df['test_date'], y=results_df['ci_l'], mode='lines', line=dict(
                width=0), fillcolor='rgba(255,0,0,0.2)', fill='tonexty', name='95% CI'))

        fig.update_layout(title='GDP Predictions',
                        xaxis_title='Date', yaxis_title='GDP')
        st.plotly_chart(fig, use_container_width=True)

        if not results_df.empty:
            results_df['resid'] = results_df['actual_gdp'] - \
                results_df['predicted_gdp']
            results_df['pct_err'] = results_df['resid'] / \
                results_df['actual_gdp'] * 100

            r_fig = px.scatter(results_df, x='test_date',
                            y='resid', trendline='lowess')
            r_fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(r_fig, use_container_width=True)

            mape = np.mean(np.abs(results_df['pct_err']))
            r2 = 1 - np.sum(results_df['resid']**2) / np.sum(
                (results_df['actual_gdp'] - results_df['actual_gdp'].mean())**2)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Error", f"{results_df['pct_err'].mean():.1f}%")
            with col2:
                st.metric("StdDev", f"{results_df['resid'].std():.1f}")
            with col3:
                st.metric("MAPE", f"{mape:.2f}%")
            with col4:
                st.metric("R²", f"{r2:.3f}")

        st.subheader("Feature Importances")
        if selected_model_name == "Random Forest":
            feature_importances = model.feature_importances_
            importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)

            fig = px.bar(importance_df, x='Feature', y='Importance',
                 title="Feature Importances (Random Forest)", labels={'Importance': 'Importance Score'})
            st.plotly_chart(fig, use_container_width=True)

        elif selected_model_name == "Linear Regression":
            coefficients = model.coef_
            importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Coefficient': coefficients
            }).sort_values(by='Coefficient', ascending=False)

            fig = px.bar(importance_df, x='Feature', y='Coefficient',
                 title="Feature Coefficients (Linear Regression)", labels={'Coefficient': 'Coefficient Value'})
            st.plotly_chart(fig, use_container_width=True)

