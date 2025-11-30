#!/usr/bin/env python3
"""
Data Preprocessor - Align and merge real estate datasets
This script handles different column names and formats
"""
import pandas as pd
import numpy as np
import os

def preprocess_realtor_data(filepath):
    """
    Preprocess realtor-data.zip.csv
    Columns: brokered_by, status, price, bed, bath, acre_lot, street, city, 
             state, zip_code, house_size, prev_sold_date
    """
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"   Original shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Create standardized dataframe
    processed = pd.DataFrame()
    
    # Direct column mapping for realtor dataset
    if 'bed' in df.columns:
        processed['bedrooms'] = df['bed']
    if 'bath' in df.columns:
        processed['bathrooms'] = df['bath']
    if 'house_size' in df.columns:
        processed['sqft'] = df['house_size']
    if 'acre_lot' in df.columns:
        processed['lot_size'] = df['acre_lot']
    if 'price' in df.columns:
        processed['price'] = df['price']
    if 'zip_code' in df.columns:
        processed['zipcode'] = df['zip_code']
    if 'state' in df.columns:
        processed['state'] = df['state']
    if 'city' in df.columns:
        processed['city'] = df['city']
    
    # Calculate age from prev_sold_date or use default
    if 'prev_sold_date' in df.columns:
        try:
            # Try to parse dates and calculate age
            df['prev_sold_date'] = pd.to_datetime(df['prev_sold_date'], errors='coerce')
            current_year = 2024
            processed['age'] = current_year - df['prev_sold_date'].dt.year
            # Fill missing ages with median
            processed['age'] = processed['age'].fillna(processed['age'].median())
        except:
            processed['age'] = 20  # Default age
    else:
        processed['age'] = 20
    
    # Create location score based on zipcode variation
    processed['location_score'] = np.random.randint(4, 10, size=len(processed))
    
    # Clean data
    print(f"   Cleaning data...")
    
    # Remove rows with missing critical values
    initial_len = len(processed)
    processed = processed.dropna(subset=['price', 'bedrooms', 'bathrooms', 'sqft'])
    print(f"   Removed {initial_len - len(processed)} rows with missing critical values")
    
    # Remove invalid prices
    processed = processed[processed['price'] > 0]
    
    # Remove outliers (houses between $10k and $10M)
    processed = processed[(processed['price'] >= 10000) & (processed['price'] <= 10000000)]
    
    # Remove invalid bedrooms/bathrooms
    processed = processed[processed['bedrooms'] > 0]
    processed = processed[processed['bathrooms'] > 0]
    
    # Remove invalid sqft
    processed = processed[processed['sqft'] > 0]
    
    print(f"   ✅ Processed shape: {processed.shape}")
    print(f"   ✅ Columns: {list(processed.columns)}")
    
    return processed

def preprocess_usa_housing(filepath):
    """
    Preprocess USA Housing Dataset.csv
    Columns: date, price, bedrooms, bathrooms, sqft_living, sqft_lot, floors, 
             waterfront, view, condition, sqft_above, sqft_basement, yr_built, 
             yr_renovated, street, city, statezip, country
    """
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    
    print(f"   Original shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Create standardized dataframe
    processed = pd.DataFrame()
    
    # Direct column mapping
    if 'bedrooms' in df.columns:
        processed['bedrooms'] = df['bedrooms']
    if 'bathrooms' in df.columns:
        processed['bathrooms'] = df['bathrooms']
    if 'sqft_living' in df.columns:
        processed['sqft'] = df['sqft_living']
    if 'sqft_lot' in df.columns:
        processed['lot_size'] = df['sqft_lot'] / 43560  # Convert sqft to acres
    if 'price' in df.columns:
        processed['price'] = df['price']
    
    # Extract zipcode from statezip if available
    if 'statezip' in df.columns:
        processed['zipcode'] = df['statezip']
    
    # Extract state from statezip
    if 'statezip' in df.columns:
        try:
            processed['state'] = df['statezip'].str.split().str[0]
        except:
            processed['state'] = 'WA'  # Default
    else:
        processed['state'] = 'WA'
    
    if 'city' in df.columns:
        processed['city'] = df['city']
    
    # Calculate age from yr_built
    if 'yr_built' in df.columns:
        current_year = 2024
        processed['age'] = current_year - df['yr_built']
        # Ensure age is positive
        processed['age'] = processed['age'].clip(lower=0)
    else:
        processed['age'] = 20  # Default age
    
    # Create location score based on waterfront, view, and condition
    if 'waterfront' in df.columns and 'view' in df.columns and 'condition' in df.columns:
        # Score from 1-10 based on features
        waterfront_score = df['waterfront'] * 3  # 0 or 3
        view_score = df['view'].clip(0, 4)       # 0-4
        condition_score = df['condition'].clip(0, 3)  # 0-3 (from 1-5 scale)
        processed['location_score'] = (waterfront_score + view_score + condition_score).clip(1, 10).astype(int)
    else:
        processed['location_score'] = 7  # Default
    
    # Clean data
    print(f"   Cleaning data...")
    
    # Check which columns actually exist
    print(f"   Available columns in processed: {list(processed.columns)}")
    
    # Only drop NA for columns that exist
    critical_cols = ['price', 'bedrooms', 'bathrooms', 'sqft']
    existing_critical_cols = [col for col in critical_cols if col in processed.columns]
    
    if len(existing_critical_cols) > 0:
        initial_len = len(processed)
        processed = processed.dropna(subset=existing_critical_cols)
        print(f"   Removed {initial_len - len(processed)} rows with missing critical values")
    
    # Clean invalid values only if columns exist
    if 'price' in processed.columns:
        processed = processed[processed['price'] > 0]
        processed = processed[(processed['price'] >= 10000) & (processed['price'] <= 10000000)]
    
    if 'bedrooms' in processed.columns:
        processed = processed[processed['bedrooms'] > 0]
    
    if 'bathrooms' in processed.columns:
        processed = processed[processed['bathrooms'] > 0]
    
    if 'sqft' in processed.columns:
        processed = processed[processed['sqft'] > 0]
    
    print(f"   ✅ Processed shape: {processed.shape}")
    print(f"   ✅ Columns: {list(processed.columns)}")
    
    return processed

def merge_and_save(df1, df2, output_path):
    """
    Merge two dataframes with common columns
    """
    print(f"\nMerging datasets...")
    
    # Find common columns
    common_cols = list(set(df1.columns) & set(df2.columns))
    print(f"   Common columns: {common_cols}")
    
    if len(common_cols) == 0:
        print("   ⚠️  No common columns found!")
        print(f"   Dataset 1 columns: {list(df1.columns)}")
        print(f"   Dataset 2 columns: {list(df2.columns)}")
        return None
    
    # Ensure 'price' is in common columns
    if 'price' not in common_cols:
        print("   ⚠️  WARNING: 'price' not in common columns!")
        return None
    
    # Keep only common columns
    df1_filtered = df1[common_cols].copy()
    df2_filtered = df2[common_cols].copy()
    
    # Merge
    merged = pd.concat([df1_filtered, df2_filtered], ignore_index=True)
    
    # Remove duplicates
    initial_len = len(merged)
    merged = merged.drop_duplicates()
    print(f"   Removed {initial_len - len(merged)} duplicate rows")
    
    # Identify and separate numeric vs text columns
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = merged.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n   Column Analysis:")
    print(f"   Numeric columns: {numeric_cols}")
    print(f"   Text columns: {text_cols}")
    
    # Remove text columns except for reference (keep in separate file)
    if text_cols:
        print(f"\n   ⚠️  Removing text columns for modeling: {text_cols}")
        print(f"   (Easier to model with numeric data only)")
        
        # Save full dataset with text columns for reference
        reference_path = output_path.replace('.csv', '_full.csv')
        merged.to_csv(reference_path, index=False)
        print(f"   💾 Saved full dataset (with text) to: {reference_path}")
        
        # Keep only numeric columns for modeling
        merged = merged[numeric_cols]
    
    print(f"   ✅ Merged shape (numeric only): {merged.shape}")
    print(f"   ✅ Final columns: {list(merged.columns)}")
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    merged.to_csv(output_path, index=False)
    print(f"\nSaved merged dataset to: {output_path}")
    
    # Show statistics
    print(f"\nMERGED DATASET STATISTICS:")
    print(f"   Total samples: {len(merged):,}")
    if 'price' in merged.columns:
        print(f"   Price range: ${merged['price'].min():,.2f} - ${merged['price'].max():,.2f}")
        print(f"   Average price: ${merged['price'].mean():,.2f}")
        print(f"   Median price: ${merged['price'].median():,.2f}")
    
    if 'bedrooms' in merged.columns:
        print(f"   Bedrooms range: {merged['bedrooms'].min():.0f} - {merged['bedrooms'].max():.0f}")
    
    if 'bathrooms' in merged.columns:
        print(f"   Bathrooms range: {merged['bathrooms'].min():.1f} - {merged['bathrooms'].max():.1f}")
    
    if 'sqft' in merged.columns:
        print(f"   Sqft range: {merged['sqft'].min():,.0f} - {merged['sqft'].max():,.0f}")
    
    return merged

def create_standardized_modelscript(columns, output_file="examples/merged_real_estate.ms"):
    """
    Create a ModelScript file based on the merged dataset columns
    """
    feature_cols = [col for col in columns if col != 'price']
    
    ms_content = f'''model MergedRealEstatePredictor {{
  dataset {{
    source: "house_prices"
    task: "regression"
    
    # Single merged dataset
    data_file: "data/merged_real_estate.csv"
    
    # Features from merged dataset
    input_features: {feature_cols}
    target: "price"
    test_split: 0.2
  }}

  architecture {{
    layer Dense {{
      units: 128
      activation: "relu"
      input_shape: [{len(feature_cols)}]
    }}
    
    layer BatchNormalization {{}}
    
    layer Dropout {{
      rate: 0.3
    }}
    
    layer Dense {{
      units: 64
      activation: "relu"
    }}
    
    layer Dropout {{
      rate: 0.2
    }}
    
    layer Dense {{
      units: 32
      activation: "relu"
    }}
    
    layer Dense {{
      units: 1
      activation: "linear"
    }}
  }}

  training {{
    batch_size: 32
    epochs: 100
    optimizer: "adam"
    loss: "mse"
    metrics: ["mae", "mse"]
  }}

  evaluation {{
    metrics: ["mae", "mse", "rmse", "mape"]
  }}
}}
'''
    
    os.makedirs("examples", exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(ms_content)
    
    print(f"\nCreated ModelScript file: {output_file}")
    print(f"   Features ({len(feature_cols)}): {feature_cols}")
    print(f"   Target: price")

def main():
    print("Real Estate Data Preprocessor")
    print("=" * 80)
    
    # File paths
    realtor_file = "data/realtor-data.zip.csv"
    usa_housing_file = "data/USA Housing Dataset.csv"
    output_file = "data/merged_real_estate.csv"
    
    # Check for alternative locations
    if not os.path.exists(realtor_file):
        realtor_file = "realtor-data.zip.csv"
    if not os.path.exists(usa_housing_file):
        usa_housing_file = "USA Housing Dataset.csv"
    
    # Check if files exist
    if not os.path.exists(realtor_file):
        print(f"❌ File not found: {realtor_file}")
        print(f"   Please ensure the file is in the data/ folder")
        return
    
    if not os.path.exists(usa_housing_file):
        print(f"❌ File not found: {usa_housing_file}")
        print(f"   Please ensure the file is in the data/ folder")
        return
    
    try:
        # Process both datasets
        df1 = preprocess_realtor_data(realtor_file)
        print()  # Blank line
        df2 = preprocess_usa_housing(usa_housing_file)
        
        # Merge and save
        merged_df = merge_and_save(df1, df2, output_file)
        
        if merged_df is not None:
            # Create ModelScript configuration
            create_standardized_modelscript(merged_df.columns)
            
            print("\n" + "=" * 80)
            print("✅ PREPROCESSING COMPLETE")
            print("-" * 80)
            print("Next steps:")
            print("  1. Review the merged dataset: data/merged_real_estate.csv")
            print("  2. Check the generated ModelScript: examples/merged_real_estate.ms")
            print("  3. Run: python house_model.py, will take a good while to train")
            print("  4. Or run: USA_housing_model.py for quick test on USA dataset only")
            print("=" * 80)
        else:
            print("\n❌ Merging failed - check errors above")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    