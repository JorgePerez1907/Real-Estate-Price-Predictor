import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

def build_ultimate_model():
    print("=" * 80)
    print("USA HOUSING PRICE PREDICTOR")
    print("=" * 80)
    
    # Load data
    print("\n[1/7] Loading data...")
    df = pd.read_csv('data/USA Housing Dataset.csv')
    print(f"Initial shape: {df.shape}")
    
    # Parse date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # === IMPROVEMENT 1: Add Temporal Features ===
    print("\n[2/7] Adding temporal features...")
    df['sale_year'] = df['date'].dt.year
    df['sale_month'] = df['date'].dt.month
    df['sale_quarter'] = df['date'].dt.quarter
    df['is_summer'] = df['sale_month'].isin([6, 7, 8]).astype(int)
    df['is_spring'] = df['sale_month'].isin([3, 4, 5]).astype(int)
    
    # Calculate house age at sale time
    df['age_at_sale'] = df['sale_year'] - df['yr_built']
    df['age_at_sale'] = df['age_at_sale'].clip(lower=0)
    
    # Years since renovation (0 if never renovated)
    df['years_since_reno'] = np.where(
        df['yr_renovated'] > 0,
        df['sale_year'] - df['yr_renovated'],
        df['age_at_sale']
    )
    
    print(f"Added temporal features: sale_year, sale_month, sale_quarter, age_at_sale, years_since_reno")
    
    # === IMPROVEMENT 2: Add Geographic Features ===
    print("\n[3/7] Adding geographic features...")
    
    # One-hot encode top 15 cities
    top_cities = df['city'].value_counts().head(15).index
    print(f"Top cities: {list(top_cities[:5])}... ({len(top_cities)} total)")
    
    for city in top_cities:
        df[f'city_{city}'] = (df['city'] == city).astype(int)
    
    # Extract zip code from statezip
    df['zipcode'] = df['statezip'].astype(str).str.extract(r'(\d{5})')[0]
    
    # Group zipcodes by average price (proxy for neighborhood quality)
    zip_prices = df.groupby('zipcode')['price'].mean()
    df['zip_avg_price'] = df['zipcode'].map(zip_prices)
    df['zip_avg_price'] = df['zip_avg_price'].fillna(df['price'].median())
    
    print(f"Added {len(top_cities)} city dummy variables + zip price encoding")
    
    # === IMPROVEMENT 3: Enhanced Feature Engineering ===
    print("\n[4/7] Creating engineered features...")
    
    # Basic features
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['bath_bed_ratio'] = df['bathrooms'] / (df['bedrooms'] + 1)
    df['has_basement'] = (df['sqft_basement'] > 0).astype(int)
    df['basement_ratio'] = df['sqft_basement'] / (df['sqft_living'] + 1)
    df['lot_living_ratio'] = df['sqft_lot'] / (df['sqft_living'] + 1)
    
    # Quality score
    df['quality_score'] = (
        df['waterfront'] * 10 +
        df['view'] * 2 +
        df['condition']
    )
    
    # Age categories
    df['is_new'] = (df['age_at_sale'] <= 5).astype(int)
    df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)
    
    # === IMPROVEMENT 4: Polynomial/Interaction Features ===
    print("\n[5/7] Creating interaction features...")
    
    # Size interactions
    df['sqft_squared'] = df['sqft_living'] ** 2
    df['sqft_log'] = np.log1p(df['sqft_living'])
    
    # Quality × Size interactions
    df['quality_sqft'] = df['quality_score'] * df['sqft_living']
    df['waterfront_sqft'] = df['waterfront'] * df['sqft_living']
    df['view_sqft'] = df['view'] * df['sqft_living']
    df['condition_sqft'] = df['condition'] * df['sqft_living']
    
    # Age interactions
    df['age_sqft'] = df['age_at_sale'] * df['sqft_living']
    df['age_squared'] = df['age_at_sale'] ** 2
    
    # Location × Features
    df['floors_sqft'] = df['floors'] * df['sqft_living']
    
    # Bedrooms per sqft (density)
    df['bed_density'] = df['bedrooms'] / (df['sqft_living'] / 1000)
    
    # Luxury score (waterfront + view + large size)
    df['luxury_score'] = (
        df['waterfront'] * 5 +
        df['view'] * 2 +
        (df['sqft_living'] > 3000).astype(int) * 3
    )
    
    print("Created 15+ interaction features")
    
    # === IMPROVEMENT 5: Remove Outliers ===
    print("\n[6/7] Removing outliers...")
    
    initial_len = len(df)
    
    # Remove rows with missing critical values
    critical_cols = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot']
    df = df.dropna(subset=critical_cols)
    
    # Remove invalid values
    df = df[df['price'] > 0]
    df = df[df['bedrooms'] > 0]
    df = df[df['bedrooms'] <= 10]  # No mega-mansions
    df = df[df['bathrooms'] > 0]
    df = df[df['sqft_living'] > 0]
    
    # Remove price outliers using IQR method
    Q1 = df['price'].quantile(0.05)
    Q3 = df['price'].quantile(0.95)
    df = df[(df['price'] >= Q1) & (df['price'] <= Q3)]
    
    # Remove price per sqft outliers (catches data errors)
    df['price_per_sqft'] = df['price'] / df['sqft_living']
    ppsf_q1 = df['price_per_sqft'].quantile(0.01)
    ppsf_q3 = df['price_per_sqft'].quantile(0.99)
    df = df[(df['price_per_sqft'] >= ppsf_q1) & (df['price_per_sqft'] <= ppsf_q3)]
    
    print(f"Removed {initial_len - len(df)} outliers ({((initial_len - len(df))/initial_len)*100:.1f}%)")
    print(f"Final dataset: {len(df)} samples")
    
    # Define feature columns
    numeric_features = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
        'age_at_sale', 'years_since_reno', 'sale_year', 'sale_month', 'sale_quarter',
        'is_summer', 'is_spring', 'total_rooms', 'bath_bed_ratio',
        'has_basement', 'basement_ratio', 'lot_living_ratio', 'quality_score',
        'is_new', 'is_renovated', 'sqft_squared', 'sqft_log',
        'quality_sqft', 'waterfront_sqft', 'view_sqft', 'condition_sqft',
        'age_sqft', 'age_squared', 'floors_sqft', 'bed_density', 'luxury_score',
        'zip_avg_price'
    ]
    
    # Add city dummy variables
    city_features = [col for col in df.columns if col.startswith('city_')]
    all_features = numeric_features + city_features
    
    print(f"\nTotal features: {len(all_features)} ({len(numeric_features)} numeric + {len(city_features)} city dummies)")
    
    # Check correlations
    print("\n=== TOP 15 FEATURES BY CORRELATION WITH PRICE ===")
    correlations = df[numeric_features + ['price']].corr()['price'].sort_values(ascending=False)
    for i, (feat, corr) in enumerate(list(correlations.items())[:16], 1):
        if feat != 'price':
            print(f"{i:2}. {feat:25} {corr:+.4f}")
    
    # Extract features and target
    X = df[all_features].values
    y = df['price'].values
    
    print(f"\n=== DATA SUMMARY ===")
    print(f"Samples: {len(X):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Price range: ${y.min():,.0f} - ${y.max():,.0f}")
    print(f"Price mean: ${y.mean():,.0f}")
    print(f"Price median: ${np.median(y):,.0f}")
    
    # Split with stratification
    price_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=price_bins
    )
    
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Log transform target
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_log.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test_log.reshape(-1, 1)).ravel()
    
    # === IMPROVEMENT 6: Ensemble Models ===
    print("\n[7/7] Training ensemble models...")
    print("=" * 80)
    
    # Model 1: Neural Network
    print("\n[Model 1/4] Training Deep Neural Network...")
    nn_model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=[len(all_features)]),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    nn_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=0.00001, verbose=0)
    
    nn_model.fit(
        X_train_scaled, y_train_scaled,
        epochs=500,
        batch_size=32,
        validation_data=(X_test_scaled, y_test_scaled),
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    # Predictions
    nn_pred_scaled = nn_model.predict(X_test_scaled, verbose=0)
    nn_pred_log = y_scaler.inverse_transform(nn_pred_scaled).flatten()
    nn_pred = np.expm1(nn_pred_log)
    
    nn_mae = np.mean(np.abs(y_test - nn_pred))
    nn_r2 = 1 - (np.mean((y_test - nn_pred)**2) / np.var(y_test))
    print(f"✓ Neural Network: MAE=${nn_mae:,.0f}, R²={nn_r2:.4f}")
    
    # Model 2: Random Forest
    print("\n[Model 2/4] Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=30, min_samples_split=5, 
                                     random_state=42, n_jobs=-1)
    rf_model.fit(X_train, np.log1p(y_train))
    rf_pred = np.expm1(rf_model.predict(X_test))
    
    rf_mae = np.mean(np.abs(y_test - rf_pred))
    rf_r2 = 1 - (np.mean((y_test - rf_pred)**2) / np.var(y_test))
    print(f"✓ Random Forest: MAE=${rf_mae:,.0f}, R²={rf_r2:.4f}")
    
    # Model 3: Gradient Boosting
    print("\n[Model 3/4] Training Gradient Boosting...")
    gb_model = GradientBoostingRegressor(n_estimators=200, max_depth=7, learning_rate=0.1,
                                        random_state=42)
    gb_model.fit(X_train, np.log1p(y_train))
    gb_pred = np.expm1(gb_model.predict(X_test))
    
    gb_mae = np.mean(np.abs(y_test - gb_pred))
    gb_r2 = 1 - (np.mean((y_test - gb_pred)**2) / np.var(y_test))
    print(f"✓ Gradient Boosting: MAE=${gb_mae:,.0f}, R²={gb_r2:.4f}")
    
    # Model 4: Ridge Regression
    print("\n[Model 4/4] Training Ridge Regression...")
    ridge_model = Ridge(alpha=10.0)
    ridge_model.fit(X_train_scaled, y_train_log)
    ridge_pred = np.expm1(ridge_model.predict(X_test_scaled))
    
    ridge_mae = np.mean(np.abs(y_test - ridge_pred))
    ridge_r2 = 1 - (np.mean((y_test - ridge_pred)**2) / np.var(y_test))
    print(f"✓ Ridge Regression: MAE=${ridge_mae:,.0f}, R²={ridge_r2:.4f}")
    
    # Ensemble: Average predictions
    print("\n[ENSEMBLE] Combining all models...")
    ensemble_pred = (nn_pred + rf_pred + gb_pred + ridge_pred) / 4
    
    ensemble_mae = np.mean(np.abs(y_test - ensemble_pred))
    ensemble_rmse = np.sqrt(np.mean((y_test - ensemble_pred)**2))
    ensemble_r2 = 1 - (np.mean((y_test - ensemble_pred)**2) / np.var(y_test))
    ensemble_mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
    ensemble_median_ape = np.median(np.abs((y_test - ensemble_pred) / y_test)) * 100
    
    print("\n" + "=" * 80)
    print("FINAL ENSEMBLE RESULTS")
    print("=" * 80)
    
    print(f'\nTest MAE: ${ensemble_mae:,.2f}')
    print(f'Test RMSE: ${ensemble_rmse:,.2f}')
    print(f'Test R²: {ensemble_r2:.4f}')
    print(f'Mean Absolute % Error: {ensemble_mape:.2f}%')
    print(f'Median Absolute % Error: {ensemble_median_ape:.2f}%')
    
    # Sample predictions
    print('\n=== SAMPLE PREDICTIONS ===')
    indices = np.random.choice(len(y_test), min(20, len(y_test)), replace=False)
    for i, idx in enumerate(sorted(indices), 1):
        error = abs(ensemble_pred[idx] - y_test[idx])
        error_pct = (error / y_test[idx]) * 100
        status = '✓' if error_pct < 15 else '•' if error_pct < 30 else '✗'
        print(f'{status} House {i:2}: Predicted ${ensemble_pred[idx]:>10,.0f} | Actual ${y_test[idx]:>10,.0f} | Error ${error:>8,.0f} ({error_pct:5.1f}%)')
    
    # Baseline comparison
    baseline_mae = np.mean(np.abs(y_test - y_train.mean()))
    improvement = ((baseline_mae - ensemble_mae) / baseline_mae) * 100
    
    print(f'\n=== COMPARISON ===')
    print(f'Baseline (mean) MAE: ${baseline_mae:,.2f}')
    print(f'Ensemble MAE: ${ensemble_mae:,.2f}')
    print(f'Improvement: {improvement:.1f}%')
    
    # Performance rating
    print(f'\n=== PERFORMANCE RATING ===')
    if ensemble_r2 > 0.80:
        print('★★★★★ EXCELLENT: R² > 0.80')
    elif ensemble_r2 > 0.70:
        print('★★★★☆ VERY GOOD: R² > 0.70')
    elif ensemble_r2 > 0.60:
        print('★★★☆☆ GOOD: R² > 0.60')
    elif ensemble_r2 > 0.50:
        print('★★☆☆☆ FAIR: R² > 0.50')
    else:
        print('★☆☆☆☆ POOR: R² < 0.50')
    
    if ensemble_mape < 10:
        print('★★★★★ EXCELLENT: MAPE < 10%')
    elif ensemble_mape < 15:
        print('★★★★☆ VERY GOOD: MAPE < 15%')
    elif ensemble_mape < 25:
        print('★★★☆☆ GOOD: MAPE < 25%')
    elif ensemble_mape < 35:
        print('★★☆☆☆ FAIR: MAPE < 35%')
    else:
        print('★☆☆☆☆ POOR: MAPE > 35%')
    
    print("\n" + "=" * 80)
    
    return nn_model, rf_model, gb_model, ridge_model

if __name__ == '__main__':
    models = build_ultimate_model()
    print("\n✓ All models trained successfully!")