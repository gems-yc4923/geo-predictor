#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:46:21 2024

@author: Li Junyi, Miao zhibo, Wu Bin
"""
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from xgboost.sklearn import XGBRegressor
import lasio


def log_transformer(x):

    return np.log(x)

log_trans = FunctionTransformer(log_transformer)

def find_closest_depth(las_depth, csv_depths):
    
    closest_depth = csv_depths.iloc[(csv_depths - las_depth).abs().argsort()[:1]]
    
    return closest_depth.values[0]

def preprocessing_data(las_data, porosity_data):
    try:
        pd.to_numeric(las_data.index)
    except:
        las_data = las_data.loc[pd.to_numeric(las_data.index, errors = 'coerce').notna()]
    new_name = {}
    for col in porosity_data.columns:
        if col == 'POROSITY\n':
            new_name[col] = 'POROSITY\n(HELIUM)'
    porosity_data.rename(columns = new_name, inplace = True)
    porosity_data['DEPTH\n'] = porosity_data['DEPTH\n'].astype(float)
    las_data = las_data[(las_data.index<=porosity_data['DEPTH\n'].max()) & (las_data.index>=porosity_data['DEPTH\n'].min())]
    porosity_data['true depth'] = porosity_data['DEPTH\n'].apply(lambda d: find_closest_depth(d, pd.Series(las_data.index)))    
    las_data = pd.merge(porosity_data,las_data, left_on='true depth', right_index = True )
    
    return las_data

class permeability_predict:
    
    def __init__(self, folder_path:str='./Sources/data/labels/permeability/preprocessed_data') -> None:
        """
        A class used to predict permeability using well-log data.
        
        This class encapsulates the processes of loading data, splitting it into training and testing sets,
        defining and training a model using XGBRegressor, and making predictions on new data.
        
        Parameters
        ----------
        folder_path : str, optional
        The path to the folder containing the training data CSV files.
        The default path is './Sources/data/labels/permeability/preprocessed_data'.

        Methods
        -------
        load_and_split(folder_name):
        Loads data from the specified folder, splits it into training and testing sets, and returns them.
        
        model():
        Processes the training data, defines the XGBRegressor model, trains it, and stores the trained model.
        
        predict(folder_name):
        Predicts permeability for new data located in the specified folder, returning a Series of predictions.
        """
        
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_and_split(folder_path)
        self.model()
        
        
    def load_and_split(self, folder_path):
        """
        Loads data from a specified folder, preprocesses it, splits it into training and test sets, and returns them.
        
        This method handles reading multiple CSV files from the given folder, combining them into a single DataFrame,
        cleaning and preprocessing the data, and finally splitting the data into training and testing sets.
        
        Parameters
        ----------
        folder_path : str
        The path to the folder containing the training data CSV files.
        
        Returns
        -------
        X_train, X_test, y_train, y_test : tuple
        The split training and testing datasets, as pandas DataFrame or numpy array.
        """
        
        csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]
        
        combined_df = pd.DataFrame()
        
        # Read each CSV file and append it into combined_df
        for file in csv_files:
            temp_df = pd.read_csv(file)
            combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
        
        space_index = combined_df[pd.to_numeric(combined_df['PERMEABILITY (HORIZONTAL)\nKair\nmd'],errors='coerce').notnull()].index
        df = combined_df.loc[space_index]
        
        porosity = list(df['POROSITY\n(HELIUM)'])
        for i, value in enumerate(porosity):
            if not isinstance(value, float):
                result = ''.join(['0' if c.isalpha() else c for c in value])
                result = result.replace(',','.')
                porosity[i] = float(result)
        
        df['POROSITY\n(HELIUM)'] = porosity
        df = df[df['POROSITY\n(HELIUM)']<50]
        
        features_columns = ['true depth', 'DENS', 'DTC', 'GR', 'NEUT', 'PEF', 'RDEP', 'POROSITY\n(HELIUM)'] # 
        X = df[features_columns]
        y = df['PERMEABILITY (HORIZONTAL)\nKair\nmd']
        y = y.replace('<.01', '0.0001')
        y = y.replace('< .01', '0.0001')
        y = y.astype(float)
        
        return train_test_split(X,y,train_size=0.8,random_state=42)
    
    def model(self):
        """
        Defines, trains, and stores an XGBRegressor model using the class's training data.
    
        This method is responsible for processing the training data, defining the XGBRegressor model,
        training it with the processed data, and storing the trained model for future predictions.
        """
        
        num_data = self.X_train.select_dtypes(include=['int64','float64']).columns
        num_pipe = make_pipeline(SimpleImputer(), StandardScaler())
        data_pipe = ColumnTransformer([
            (['log_transformer',log_trans,['RDEP']]),
            ('num', num_pipe, num_data)])
        
        final_pipeline = Pipeline([
            ('data_pipe', data_pipe),
            ('Regressor', XGBRegressor(max_depth=10, learning_rate=0.01, n_estimators=300, reg_alpha = 0.5, reg_lambda = 1, subsample = 0.5, gamma = 0.1))])
        
        final_pipeline.fit(self.X_train, self.y_train)
        
        self.pre_model = final_pipeline
        self.y_pred = final_pipeline.predict(self.X_test)


    def predict(self, path_to_new_las_file:str='data/new_las_data.las', path_to_new_por_file:str='data/new_por_data.csv') -> np.array:
        """
        Predicts permeability for new data located in the specified folder.
    
        Parameters
        ----------
        folder_name : str
            The name of the folder containing the unlabeled data CSV files.
    
        Returns
        -------
        predictions : pd.Series
            A Series containing the predicted permeability values for the test data.
        """
        well_las = lasio.read(path_to_new_las_file)
        
        las_data = well_las.df()
        por_data = pd.read_csv(path_to_new_por_file)
        
        well_df = preprocessing_data(las_data, por_data)
        
        
        features_columns = ['true depth', 'DENS', 'DTC', 'GR', 'NEUT', 'PEF', 'RDEP', 'POROSITY\n(HELIUM)']  #
        
        X = well_df[features_columns]
        
        y_pred = pd.Series(self.pre_model.predict(X))
        
        well_df['Prediction'] = y_pred
        
        self.predicted_result = well_df
        
        return y_pred


