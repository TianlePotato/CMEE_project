############################ MODULES ##################################
# Data Processing
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Modelling
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from scipy.stats import randint
import xgboost as xgb

# Plotting
import matplotlib.pyplot as plt


########################### DATA #######################################
# Store final accuracy for all models
final_data = pd.DataFrame(columns=["accuracy", "datatype", "frames", "reduced_to", "model"])

# List to store the classification reports
reports = []


folders = ["KP3D"]
frames = [30]

# 1. #####################################
# Loop through all data types
for folder in folders:
    print(folder)

    # 2. #############################################
    # Loop through all frames
    for num_frames in frames:
        print(num_frames)

        # How many frames to count?
        num_frames_str = str(num_frames)

        reduced_frames = [30]

        # 3. #######################################################################
        # Loop through reduced frames:
        for num_frames_reduced in reduced_frames:

            rf_str = str(num_frames_reduced)

            # Define the base directory
            base_dir = "/media/tianle/Elements/Dataset-3DPOP/Dataset/ML_data_backup/" + folder + "_" + num_frames_str + "_" + rf_str

            # Dataset
            kp_train = pd.read_csv(os.path.join(base_dir, "Train_kp_"+num_frames_str+"_"+rf_str+".csv"), index_col=0)
            gt_train = pd.read_csv(os.path.join(base_dir, "Train_gt_"+num_frames_str+"_"+rf_str+".csv"), index_col=0)
            move_train = pd.read_csv(os.path.join(base_dir, "Train_movement_"+num_frames_str+"_"+rf_str+".csv"), index_col=0)

            # # Dataset val
            # kp_val = pd.read_csv(os.path.join(base_dir, "Val_kp_"+num_frames_str+".csv"), index_col=0)
            # gt_val = pd.read_csv(os.path.join(base_dir, "Val_gt_"+num_frames_str+".csv"), index_col=0)
            # move_val = pd.read_csv(os.path.join(base_dir, "Val_movement_"+num_frames_str+".csv"), index_col=0)

            # Dataset test
            kp_test = pd.read_csv(os.path.join(base_dir, "Test_kp_"+num_frames_str+"_"+rf_str+".csv"), index_col=0)
            gt_test = pd.read_csv(os.path.join(base_dir, "Test_gt_"+num_frames_str+"_"+rf_str+".csv"), index_col=0)
            move_test = pd.read_csv(os.path.join(base_dir, "Test_movement_"+num_frames_str+"_"+rf_str+".csv"), index_col=0)


            # Function to flatten data &
            # Combine main and movement data
            def PrepData(main_data, move_data, rows_to_keep):
                """
                Takes in raw pd.dataframe data
                Outputs data coordinates (x) and corresponding behaviour (y) as array
                """


                total_rows = main_data.shape[0]
                total_cols = main_data.shape[1]
                behav_len = move_data.shape[1]

                # Create empty array based on flattened array
                Dataset_array = np.empty((int(total_rows/rows_to_keep), 
                                        rows_to_keep*total_cols+behav_len))
                Dataset_array[:] = np.nan

                # Loop to flatten array
                j = 0
                for i in range(0, len(main_data), rows_to_keep):
                    # Flatten all rows of data in a frame into one row
                    temp1 = main_data.iloc[i:i+rows_to_keep].to_numpy().flatten()

                    # Add the movement data to the row 
                    temp2 = [move_data.iloc[j]]

                    # Combine the two
                    temp = np.append(temp1, temp2)

                    # Add to overall array
                    Dataset_array[j,:] = temp
                    j += 1

                return Dataset_array

            
            # Wrangle keypoint data into usable format
            x_train = PrepData(kp_train, move_train, num_frames_reduced)
            #x_val =   PrepData(kp_val,   move_val,   num_frames)
            x_test =  PrepData(kp_test,  move_test,  num_frames_reduced)

            # Groundtruth data
            y_train = np.array(gt_train["behaviour"])
            #y_val = np.array(gt_val["behaviour"])
            y_test = np.array(gt_test["behaviour"])


            ############################### Testing random forest #################################################
            # Create empty model
            model = RandomForestClassifier(random_state=69)

            # Hyperparameter tuning parameters            
            param_grid = {
                'max_depth': [5, 7, 10, 20, 50, None], # default: None
                'min_samples_split': [2, 3, 4, 5], # default: 2
                'max_leaf_nodes': [90, 105, 120, 135, 150], # default: None
                'min_samples_leaf': [1, 2, 3, 4, 5] # default: 
                }
            
            # Initialize GridSearchCV
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, refit='accuracy', cv=5, n_jobs=-1, verbose=2)

            # Fit the grid search to the data
            grid_search.fit(x_train, y_train)

            # find best parameters
            best_params = grid_search.best_params_
            print(f"Best Parameters: {best_params}")

            # create best model
            best_model = grid_search.best_estimator_

            # predict on test set
            y_pred = best_model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Test Accuracy: {accuracy}")

            # predict on training set
            y_pred = best_model.predict(x_train)
            accuracy = accuracy_score(y_train, y_pred)
            print(f"Train Accuracy: {accuracy}")

            # get cross val score
            score = cross_val_score(best_model, x_train, y_train, cv=5)
            std_score = np.std(cross_val_score)
            print(f"Standard deviation for cross validation: {std_score}")

