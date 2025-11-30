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

def build_merged_dataset_model():
    """
    Train model on MERGED dataset (2.2M samples, 6 basic features)
    
    NOTE: This performs MUCH WORSE than USA Housing dataset due to weak features!
    Expected R²: 0.15-0.30 (vs 0.78 with USA Housing)
    """
    print("=" * 80)
    print("REAL ESTATE PREDICTOR - MERGED DATASET VERSION")
    print("WARNING: Using basic features - expect lower performance")
    print("=" * 80)
    
    # Load merged data (from preprocessing)
    print("\n[1/6] Loading merged dataset...")
    df = pd.read_csv('data/merged_real_estate.csv')
    print(f"Initial shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Available features (only 6 basic ones)
    base_features = ['bedrooms', 'bathrooms', 'sqft', 'lot_size', 'age', 'location_score']
    target_col = 'price'
    
    # Check what columns actually exist
    available_features = [f for f in base_features if f in df.columns]
    print(f"Available features: {available_features}")
    
    # Clean data
    print("\n[2/6] Cleaning data...")
    df_clean = df[available_features + [target_col]].dropna()
    
    initial_len = len(df_clean)
    
    # Remove invalid values
    df_clean = df_clean[df_clean['price'] > 0]
    df_clean = df_clean[df_clean['bedrooms'] > 0]
    df_clean = df_clean[df_clean['bedrooms'] <= 10]
    df_clean = df_clean[df_clean['bathrooms'] > 0]
    df_clean = df_clean[df_clean['sqft'] > 0]
    
    # Remove extreme outliers
    Q1 = df_clean['price'].quantile(0.05)
    Q3 = df_clean['price'].quantile(0.95)
    df_clean = df_clean[(df_clean['price'] >= Q1) & (df_clean['price'] <= Q3)]
    
    # Price per sqft outliers
    df_clean['price_per_sqft'] = df_clean['price'] / df_clean['sqft']
    ppsf_q1 = df_clean['price_per_sqft'].quantile(0.01)
    ppsf_q3 = df_clean['price_per_sqft'].quantile(0.99)
    df_clean = df_clean[(df_clean['price_per_sqft'] >= ppsf_q1) & 
                        (df_clean['price_per_sqft'] <= ppsf_q3)]
    
    print(f"Removed {initial_len - len(df_clean)} outliers ({((initial_len - len(df_clean))/initial_len)*100:.1f}%)")
    print(f"Final dataset: {len(df_clean):,} samples")
    
    # Feature engineering (limited by available features)
    print("\n[3/6] Creating engineered features...")
    
    df_clean['total_rooms'] = df_clean['bedrooms'] + df_clean['bathrooms']
    df_clean['bath_bed_ratio'] = df_clean['bathrooms'] / (df_clean['bedrooms'] + 1)
    df_clean['sqft_log'] = np.log1p(df_clean['sqft'])
    df_clean['sqft_squared'] = df_clean['sqft'] ** 2
    df_clean['age_squared'] = df_clean['age'] ** 2
    df_clean['location_sqft'] = df_clean['location_score'] * df_clean['sqft']
    df_clean['lot_per_sqft'] = df_clean['lot_size'] / (df_clean['sqft'] + 1)
    
    # Feature list
    feature_cols = available_features + [
        'total_rooms', 'bath_bed_ratio', 'sqft_log', 'sqft_squared',
        'age_squared', 'location_sqft', 'lot_per_sqft'
    ]
    
    print(f"Total features: {len(feature_cols)}")
    
    # Correlations
    print("\n=== FEATURE CORRELATIONS WITH PRICE ===")
    correlations = df_clean[feature_cols + [target_col]].corr()[target_col].sort_values(ascending=False)
    for i, (feat, corr) in enumerate(list(correlations.items())[:11], 1):
        if feat != target_col:
            print(f"{i:2}. {feat:25} {corr:+.4f}")
    
    # Extract features and target
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    
    print(f"\n=== DATA SUMMARY ===")
    print(f"Samples: {len(X):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Price range: ${y.min():,.0f} - ${y.max():,.0f}")
    print(f"Price mean: ${y.mean():,.0f}")
    
    # Split (use smaller test set due to large data size)
    price_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=price_bins
    )
    
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Log transform target
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_log.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test_log.reshape(-1, 1)).ravel()
    
    # Train ensemble
    print("\n[4/6] Training ensemble models...")
    print("=" * 80)
    
    # Model 1: Neural Network (smaller due to weak features)
    print("\n[Model 1/4] Training Neural Network...")
    nn_model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=[len(feature_cols)]),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    nn_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=0.00001, verbose=0)
    
    nn_model.fit(
        X_train_scaled, y_train_scaled,
        epochs=200,
        batch_size=128,
        validation_data=(X_test_scaled, y_test_scaled),
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    nn_pred_scaled = nn_model.predict(X_test_scaled, verbose=0)
    nn_pred_log = y_scaler.inverse_transform(nn_pred_scaled).flatten()
    nn_pred = np.expm1(nn_pred_log)
    
    nn_mae = np.mean(np.abs(y_test - nn_pred))
    nn_r2 = 1 - (np.mean((y_test - nn_pred)**2) / np.var(y_test))
    print(f"✓ Neural Network: MAE=${nn_mae:,.0f}, R²={nn_r2:.4f}")
    
    # Model 2: Random Forest (subsample for speed)
    print("\n[Model 2/4] Training Random Forest...")
    sample_size = min(100000, len(X_train))
    sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
    
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=25, min_samples_split=10,
                                     random_state=42, n_jobs=-1)
    rf_model.fit(X_train[sample_idx], np.log1p(y_train[sample_idx]))
    rf_pred = np.expm1(rf_model.predict(X_test))
    
    rf_mae = np.mean(np.abs(y_test - rf_pred))
    rf_r2 = 1 - (np.mean((y_test - rf_pred)**2) / np.var(y_test))
    print(f"✓ Random Forest: MAE=${rf_mae:,.0f}, R²={rf_r2:.4f}")
    
    # Model 3: Gradient Boosting (subsample)
    print("\n[Model 3/4] Training Gradient Boosting...")
    gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                        random_state=42, subsample=0.8)
    gb_model.fit(X_train[sample_idx], np.log1p(y_train[sample_idx]))
    gb_pred = np.expm1(gb_model.predict(X_test))
    
    gb_mae = np.mean(np.abs(y_test - gb_pred))
    gb_r2 = 1 - (np.mean((y_test - gb_pred)**2) / np.var(y_test))
    print(f"✓ Gradient Boosting: MAE=${gb_mae:,.0f}, R²={gb_r2:.4f}")
    
    # Model 4: Ridge
    print("\n[Model 4/4] Training Ridge Regression...")
    ridge_model = Ridge(alpha=10.0)
    ridge_model.fit(X_train_scaled, y_train_log)
    ridge_pred = np.expm1(ridge_model.predict(X_test_scaled))
    
    ridge_mae = np.mean(np.abs(y_test - ridge_pred))
    ridge_r2 = 1 - (np.mean((y_test - ridge_pred)**2) / np.var(y_test))
    print(f"✓ Ridge Regression: MAE=${ridge_mae:,.0f}, R²={ridge_r2:.4f}")
    
    # Ensemble
    print("\n[5/6] Combining ensemble...")
    ensemble_pred = (nn_pred + rf_pred + gb_pred + ridge_pred) / 4
    
    ensemble_mae = np.mean(np.abs(y_test - ensemble_pred))
    ensemble_rmse = np.sqrt(np.mean((y_test - ensemble_pred)**2))
    ensemble_r2 = 1 - (np.mean((y_test - ensemble_pred)**2) / np.var(y_test))
    ensemble_mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
    ensemble_median_ape = np.median(np.abs((y_test - ensemble_pred) / y_test)) * 100
    
    print("\n" + "=" * 80)
    print("FINAL ENSEMBLE RESULTS (MERGED DATASET)")
    print("=" * 80)
    
    print(f'\nTest MAE: ${ensemble_mae:,.2f}')
    print(f'Test RMSE: ${ensemble_rmse:,.2f}')
    print(f'Test R²: {ensemble_r2:.4f}')
    print(f'Mean Absolute % Error: {ensemble_mape:.2f}%')
    print(f'Median Absolute % Error: {ensemble_median_ape:.2f}%')
    
    # Sample predictions
    print('\n=== SAMPLE PREDICTIONS ===')
    indices = np.random.choice(len(y_test), min(15, len(y_test)), replace=False)
    for i, idx in enumerate(sorted(indices), 1):
        error = abs(ensemble_pred[idx] - y_test[idx])
        error_pct = (error / y_test[idx]) * 100
        status = '✓' if error_pct < 15 else '•' if error_pct < 30 else '✗'
        print(f'{status} House {i:2}: Predicted ${ensemble_pred[idx]:>10,.0f} | Actual ${y_test[idx]:>10,.0f} | Error ${error:>8,.0f} ({error_pct:5.1f}%)')
    
    # Comparison
    baseline_mae = np.mean(np.abs(y_test - y_train.mean()))
    improvement = ((baseline_mae - ensemble_mae) / baseline_mae) * 100
    
    print(f'\n=== COMPARISON ===')
    print(f'Baseline MAE: ${baseline_mae:,.2f}')
    print(f'Ensemble MAE: ${ensemble_mae:,.2f}')
    print(f'Improvement: {improvement:.1f}%')
    
    # Rating
    print(f'\n=== PERFORMANCE RATING ===')
    if ensemble_r2 > 0.80:
        print('★★★★★ EXCELLENT: R² > 0.80')
    elif ensemble_r2 > 0.70:
        print('★★★★☆ VERY GOOD: R² > 0.70')
    if ensemble_r2 > 0.60:
        print('★★★☆☆ GOOD: R² > 0.60')
    elif ensemble_r2 > 0.50:
        print('★★☆☆☆ FAIR: R² > 0.50')
    else:
        print('★☆☆☆☆ POOR: R² > 0.50')
    
    if ensemble_mape < 10:
        print('★★★★★ EXCELLENT: MAPE < 10%')
    elif ensemble_mape < 15:
        print('★★★★☆ VERY GOOD: MAPE < 15%')
    if ensemble_mape < 25:
        print('★★★☆☆ GOOD: MAPE < 25%')
    elif ensemble_mape < 35:
        print('★★☆☆☆ FAIR: MAPE < 35%')
    else:
        print('★☆☆☆☆ POOR: MAPE > 35%')
    
    print("\n" + "=" * 80)
    print("NOTE: Performance is limited by basic features in merged dataset")
    print("For better results, use USA Housing dataset with rich features (R² = 0.78)")
    print("=" * 80)
    
    return nn_model, rf_model, gb_model, ridge_model

if __name__ == '__main__':
    models = build_merged_dataset_model()
    print("\n✓ Training complete!")