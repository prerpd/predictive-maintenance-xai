# predictive-maintenance-xai
An end-to-end ML project predicting machine failure with XAI (SHAP).


## Structure
predictive_maintenance_xai/  
├── data/               # Download ai4i2020.csv here 
├── models/             # Saved .json or .pkl models 
├── notebooks/          # For initial EDA only  
├── src/                # The "Engine" of your project  
│   ├── __init__.py 
│   ├── preprocess.py   # Data cleaning & Engineering  
│   ├── train.py        # Model training logic  
│   └── explain.py      # SHAP visualization logic 
├── requirements.txt  
└── README.md  
