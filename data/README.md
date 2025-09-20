# Data Directory Structure

This directory contains all data files used by the PStock AI system.

## Subdirectories:

- **stock/**: Individual stock data files (CSV format)
- **index/**: Market index data files 
- **model/**: Trained model files (.h5, .pkl format)
- **scaler/**: Data normalization scaler files (.save format)
- **bins/**: Data binning configuration files
- **global/**: Global market data and macroeconomic indicators
- **temp/**: Temporary files during processing

## Note:
All data files are excluded from version control via .gitignore for performance and privacy reasons.
The directory structure is created automatically when needed by the application.