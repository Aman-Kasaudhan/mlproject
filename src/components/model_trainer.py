import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_filr_path=os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def inintiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                'Random Forest Regressor':RandomForestRegressor(),
                'Decision Tree':DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regressor":LinearRegression(),
                'k-neightbour Regressor':KNeighborsRegressor(),
                 'XGBRegressor':XGBRegressor(),
                'Adaboost Regressor':AdaBoostRegressor()
            }

            logging.info("Hyperparameter tunning started")

            params = {

                 "Random Forest Regressor": {
                     "n_estimators": [100, 200, 300],
                     "max_depth": [None, 10, 20, 30],
                     "min_samples_split": [2, 5, 10],
                     "min_samples_leaf": [1, 2, 4],
                     "max_features": ["sqrt", "log2"]
                 },

                 "Decision Tree": {
                     "max_depth": [None, 10, 20, 30],
                     "min_samples_split": [2, 5, 10],
                     "min_samples_leaf": [1, 2, 4],
                     "criterion": ["squared_error", "friedman_mse"]
                 },

                 "Gradient Boosting": {
                     "n_estimators": [100, 200, 300],
                     "learning_rate": [0.01, 0.05, 0.1],
                     "max_depth": [3, 5, 7],
                     "subsample": [0.8, 1.0],
                     "min_samples_split": [2, 5]
                 },

                 "Linear Regressor": {
                     "fit_intercept": [True, False],
                     "positive": [False, True]
                     },
             
                 "k-neightbour Regressor": {
                     "n_neighbors": [3, 5, 7, 9],
                     "weights": ["uniform", "distance"],
                     "algorithm": ["auto", "ball_tree", "kd_tree"],
                     "p": [1, 2]   # Manhattan vs Euclidean
                 },

                 "XGBRegressor": {
                     "n_estimators": [100, 200, 300],
                     "learning_rate": [0.01, 0.05, 0.1],
                     "max_depth": [3, 5, 7],
                     "subsample": [0.7, 0.8, 1.0],
                     "colsample_bytree": [0.7, 0.8, 1.0],
                     "gamma": [0, 0.1, 0.2],
                     "reg_alpha": [0, 0.1],
                     "reg_lambda": [1, 1.5]
                 },
             
                 "Adaboost Regressor": {
                     "n_estimators": [50, 100, 200],
                     "learning_rate": [0.01, 0.05, 0.1, 1.0],
                     "loss": ["linear", "square", "exponential"]
                 }
                     }


           

            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            # model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,param=params)

            print("Hyperparameter tunning completed")
            logging.info("Hyperparameter tunning completed")

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                return CustomException("No best model found")
            
            logging.info(f"Best model found for both training and test dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_filr_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)

            r2_square=r2_score(y_test,predicted)


            return r2_square



        except Exception as e:
            raise CustomException(e,sys)