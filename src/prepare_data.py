"""
Data preparation script for admission prediction model.
Loads, cleans, and splits the admission dataset into training and test sets.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

def load_data(file_path):
    """Load the admission dataset from CSV file."""
    try:
        df = pd.read_csv(file_path)
        log.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        log.error(f"File {file_path} not found.")
        return None
    except Exception as e:
        log.error(f"Error loading data: {e}")
        return None

def clean_data(df):
    """Clean the dataset by removing unnecessary columns and handling missing values."""
    log.info("=== Data Cleaning ===")
    
    # Display basic info about the dataset
    log.info(f"Original dataset shape: {df.shape}")
    log.info(f"Columns: {df.columns.tolist()}")
    
    # Remove Serial No. column as it's just an identifier and not useful for modeling
    if 'Serial No.' in df.columns:
        df = df.drop('Serial No.', axis=1)
        log.info("Removed 'Serial No.' column (not relevant for modeling)")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    log.info(f"Missing values per column:\n{missing_values}")
    
    if missing_values.sum() > 0:
        log.warning("Handling missing values...")
        # For numerical columns, fill with median
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
                log.info(f"Filled missing values in {col} with median")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        log.warning(f"Found {duplicates} duplicate rows. Removing duplicates...")
        df = df.drop_duplicates()
    else:
        log.info("No duplicate rows found")
    
    # Display data types and basic statistics
    log.info(f"Cleaned dataset shape: {df.shape}")
    log.info(f"Data types:\n{df.dtypes}")
    log.info(f"Basic statistics:\n{df.describe()}")
    
    return df

def prepare_features_target(df):
    """Separate features and target variable."""
    log.info("=== Feature and Target Preparation ===")
    
    # Define target variable
    target_column = 'Chance of Admit '  # Note the space at the end
    if target_column not in df.columns:
        # Try without space
        target_column = 'Chance of Admit'
        if target_column not in df.columns:
            log.error("Target variable 'Chance of Admit' not found in columns")
            log.error(f"Available columns: {df.columns.tolist()}")
            return None, None
    
    # Separate features (X) and target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    log.info(f"Features shape: {X.shape}")
    log.info(f"Target shape: {y.shape}")
    log.info(f"Feature columns: {X.columns.tolist()}")
    log.info(f"Target variable: {target_column}")
    log.info(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    log.info("=== Data Splitting ===")
    log.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    
    log.info(f"Training set: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    log.info(f"Test set: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, output_dir):
    """Save the processed datasets to the specified directory."""
    log.info("=== Saving Processed Data ===")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the datasets
    file_paths = {
        'X_train': os.path.join(output_dir, 'X_train.csv'),
        'X_test': os.path.join(output_dir, 'X_test.csv'),
        'y_train': os.path.join(output_dir, 'y_train.csv'),
        'y_test': os.path.join(output_dir, 'y_test.csv')
    }
    
    try:
        # Save as CSV files
        X_train.to_csv(file_paths['X_train'], index=False)
        X_test.to_csv(file_paths['X_test'], index=False)
        y_train.to_csv(file_paths['y_train'], index=False)
        y_test.to_csv(file_paths['y_test'], index=False)
        
        log.info("Files saved successfully:")
        for name, path in file_paths.items():
            log.info(f"  {name}: {path}")
    except Exception as e:
        log.error(f"Error saving processed data: {e}")
        return None
    
    return file_paths

def main():
    """Main function to execute the data preparation pipeline."""
    log.info("=== ADMISSION DATA PREPARATION PIPELINE ===")
    
    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    input_file = os.path.join(project_root, 'data', 'raw', 'admission.csv')
    output_dir = os.path.join(project_root, 'data', 'processed')
    
    log.info(f"Input file: {input_file}")
    log.info(f"Output directory: {output_dir}")
    
    # Step 1: Load data
    df = load_data(input_file)
    if df is None:
        log.fatal("Failed to load data. Exiting pipeline.")
        return
    
    # Step 2: Clean data
    df_clean = clean_data(df)
    
    # Step 3: Prepare features and target
    X, y = prepare_features_target(df_clean)
    if X is None or y is None:
        log.fatal("Failed to prepare features and target. Exiting pipeline.")
        return
    
    # Step 4: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 5: Save processed data
    file_paths = save_processed_data(X_train, X_test, y_train, y_test, output_dir)
    if file_paths is None:
        log.fatal("Failed to save processed data. Exiting pipeline.")
        return
    
    log.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
    log.info("Dataset Summary:")
    log.info(f"  Total samples: {len(df_clean)}")
    log.info(f"  Features: {len(X.columns)}")
    log.info(f"  Training samples: {len(X_train)}")
    log.info(f"  Test samples: {len(X_test)}")
    log.info(f"  Train/Test split: {len(X_train)/(len(X_train)+len(X_test))*100:.1f}%/{len(X_test)/(len(X_train)+len(X_test))*100:.1f}%")

if __name__ == "__main__":
    main()
