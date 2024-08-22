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


folders = ["KP2D", "KP3D", "Angle"]
frames = [1, 15, 30, 60, 90]

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

        # Reduce frame count?
        if num_frames == 1:
            reduced_frames = [num_frames]

        elif num_frames == 15:
            reduced_frames = [num_frames, 5]

        else:
            reduced_frames = [num_frames, 15, 10, 5]

        # 3. #######################################################################
        # Loop through reduced frames:
        for num_frames_reduced in reduced_frames:

            rf_str = str(num_frames_reduced)

            # Define the base directory
            base_dir = "/media/tianle/Elements/Dataset-3DPOP/Dataset/ML_data/" + folder + "_" + num_frames_str + "_" + rf_str

            # Dataset
            kp_train = pd.read_csv(os.path.join(base_dir, "Train_kp_"+num_frames_str+"_"+rf_str+".csv"), index_col=0)
            gt_train = pd.read_csv(os.path.join(base_dir, "Train_gt_"+num_frames_str+"_"+rf_str+".csv"), index_col=0)
            move_train = pd.read_csv(os.path.join(base_dir, "Train_move_"+num_frames_str+"_"+rf_str+".csv"), index_col=0)

            # # Dataset val
            # kp_val = pd.read_csv(os.path.join(base_dir, "Val_kp_"+num_frames_str+".csv"), index_col=0)
            # gt_val = pd.read_csv(os.path.join(base_dir, "Val_gt_"+num_frames_str+".csv"), index_col=0)
            # move_val = pd.read_csv(os.path.join(base_dir, "Val_movement_"+num_frames_str+".csv"), index_col=0)

            # Dataset test
            kp_test = pd.read_csv(os.path.join(base_dir, "Test_kp_"+num_frames_str+"_"+rf_str+".csv"), index_col=0)
            gt_test = pd.read_csv(os.path.join(base_dir, "Test_gt_"+num_frames_str+"_"+rf_str+".csv"), index_col=0)
            move_test = pd.read_csv(os.path.join(base_dir, "Test_move_"+num_frames_str+"_"+rf_str+".csv"), index_col=0)


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

            # Fit model
            model = model.fit(x_train, y_train)

            # Hyperparameter tuning parameters
            param_grid = {
                'max_depth': [25, 50, None], # default: None
                'min_samples_split': [2, 6, 12], # default: 2
                'max_leaf_nodes': [40, 80, None], # default: None
                'min_samples_leaf': [1, 3, 9] # default: 1
                }

            # Model test
            prediction = model.predict(x_test)

            # Get accuracy score
            val_accuracy = accuracy_score(y_test, prediction)

            model = "RF"
            #results
            results = pd.DataFrame({"accuracy": val_accuracy, "datatype": folder, "frames": num_frames, "reduced_to": num_frames_reduced, "model": model}, index=[0])

            final_data = pd.concat([final_data, results])


            #### Plot & save confusion matrix
            cm = confusion_matrix(y_test, prediction, normalize="true")
            labels = gt_train["behaviour"].unique().tolist()
            fig, ax = plt.subplots(figsize=(12, 10))
            cm_plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            cm_plot.plot(ax=ax) # ax changes axis sizes for figure

            figname = "/media/tianle/Elements/Dataset-3DPOP/Dataset/confusion_matrix/" +folder+ "_" +num_frames_str+"_"+rf_str+ "_rf.pdf"
            cm_plot.figure_.savefig(figname)


            # Get classification report - including precision metrics etc.
            classification_dict = classification_report(y_test, prediction, output_dict=True)

            # Add additional information to the classification report
            classification_dict['model'] = 'rf'
            classification_dict['folder'] = folder
            classification_dict['num_frames'] = num_frames
            classification_dict['num_frames_reduced'] = num_frames_reduced
            classification_dict['accuracy'] = val_accuracy

            # Flatten the classification report for easier DataFrame conversion
            flat_report = {}
            for label, metrics in classification_dict.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        flat_report[f"{label}_{metric}"] = value
                else:
                    flat_report[label] = metrics

            # Append the flattened report to the list
            reports.append(flat_report)


            ############################### Testing XGBoost #####################################################
            # change y labels into integers for XGBoost model

            # Initialize an empty list to hold the values
            y_train_int = []
            y_test_int = []
            # Loop through the range 0 to 6
            for i in range(7):
                # Extend the list with the current value repeated 35 times
                y_train_int.extend([i] * 210)
                y_test_int.extend([i] * 35)
            # Convert the list to a NumPy array
            y_train_int = np.array(y_train_int)
            y_test_int = np.array(y_test_int)



            # Model itself
            xgb_model = xgb.XGBClassifier(random_state = 69) 

            # Fit model
            xgb_model.fit(x_train, y_train_int)

            # Model test
            prediction = xgb_model.predict(x_test)

            # Get accuracy score
            val_accuracy = accuracy_score(y_test_int, prediction) 

            model = "XG"
            #results
            results = pd.DataFrame({"accuracy": val_accuracy, "datatype": folder, "frames": num_frames, "reduced_to": num_frames_reduced, "model": model}, index=[0])

            final_data = pd.concat([final_data, results])


            #### Plot & save confusion matrix
            cm = confusion_matrix(y_test_int, prediction, normalize="true")
            labels = gt_train["behaviour"].unique().tolist()
            fig, ax = plt.subplots(figsize=(12, 10))
            cm_plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            cm_plot.plot(ax=ax) # ax changes axis sizes for figure

            figname = "/media/tianle/Elements/Dataset-3DPOP/Dataset/confusion_matrix/" +folder+ "_" +num_frames_str+"_"+rf_str+ "_xg.pdf"
            cm_plot.figure_.savefig(figname)
            plt.close(fig)

            # translate integer classes into named classes
            translate = {0:"grooming",1:"courting_status",2:"bowing_status",
                         3:"headdown",4:"feeding",5:"walking",6:"vigilance_status"}
            trans_prediction = np.vectorize(translate.get)(prediction)

            # Get classification report - including precision metrics etc.
            classification_dict = classification_report(y_test, trans_prediction, output_dict=True)

            # Add additional information to the classification report
            classification_dict['model'] = 'xg'
            classification_dict['folder'] = folder
            classification_dict['num_frames'] = num_frames
            classification_dict['num_frames_reduced'] = num_frames_reduced
            classification_dict['accuracy'] = val_accuracy

            # Flatten the classification report for easier DataFrame conversion
            flat_report = {}
            for label, metrics in classification_dict.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        flat_report[f"{label}_{metric}"] = value
                else:
                    flat_report[label] = metrics

            # Append the flattened report to the list
            reports.append(flat_report)



# Final data export
df_reports = pd.DataFrame(reports)
df_reports.to_csv("/media/tianle/Elements/Dataset-3DPOP/Dataset/classification_reports.csv")


# #################################### MAKING THE RANDOM FOREST MODEL ########################################
# model = RandomForestClassifier(random_state=69)

# model = model.fit(x_train, y_train)

# # prediction = model.predict(x_train)
# # accuracy_score(y_train, prediction)
# # cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')


# # # HYPERPARAMETER TUNING
# # param_dist = {'n_estimators': randint(50,500),
# #               'max_depth': randint(1,20)}

# # # Create a random forest classifier
# # rf = RandomForestClassifier()

# # # Use random search to find the best hyperparameters
# # rand_search = RandomizedSearchCV(rf, 
# #                                  param_distributions = param_dist, 
# #                                  n_iter=5, 
# #                                  cv=5)

# # # Fit the random search object to the data
# # rand_search.fit(x_train, y_train)

# # # Create a variable for the best model
# # best_rf = rand_search.best_estimator_

# # # Print the best hyperparameters
# # print('Best hyperparameters:',  rand_search.best_params_)


# Model validation
# prediction = model.predict(x_val)

# val_accuracy = accuracy_score(y_val, prediction) 
# print(val_accuracy, "validation")

# cm = confusion_matrix(y_test, prediction, normalize="true")

# labels = gt_train["behaviour"].unique().tolist()
# fig, ax = plt.subplots(figsize=(12, 10))
# cm_plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
# cm_plot.plot(ax=ax)

# cm_plot.figure_.savefig("Dataset/confusion_matrix/cm.pdf")

# # Model test
# prediction = model.predict(x_test)

# val_accuracy = accuracy_score(y_test, prediction) 
# print(val_accuracy, "test")

# cm = confusion_matrix(y_test, prediction, normalize="true")

# labels = gt_train["behaviour"].unique().tolist()
# ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot()
# plt.show() 



# ################################ XGBoost model ##############################

# # change y labels into integers for XGBoost model
# label_encoder = LabelEncoder()
# y_train_int = label_encoder.fit_transform(y_train)
# y_val_int = label_encoder.fit_transform(y_val)
# y_test_int = label_encoder.fit_transform(y_test)

# # Model itself
# xgb_model = xgb.XGBClassifier(objective='multi:softprob', # Loss function for multi-classification
#                               verbosity = 2,
#                               random_state = 69,
#                               colsample_bytree = 0.5) 

# # Fit model
# xgb_model.fit(x_train,
#               y_train_int,
#               eval_set=[(x_val, y_val_int)], 
#               verbose=True)


# # # Model validation
# # prediction = xgb_model.predict(x_val)

# # val_accuracy = accuracy_score(y_val_int, prediction) 
# # print(val_accuracy, "validation")

# # cm = confusion_matrix(y_val_int, prediction, normalize="true")

# # labels = gt_train["behaviour"].unique().tolist()
# # labels = ['grooming',
# #  'courting',
# #  'bowing',
# #  'headdown',
# #  'feeding',
# #  'walking',
# #  'headup']
# # ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot()
# # plt.show() 


# # Hyperparameter tuning
# param_dist = {
#     # 'n_estimators': [70, 100, 150, 200],
#     # 'learning_rate': [0.01, 0.05, 0.1],
#     'max_depth': [3, 6, 9],
# #     'min_child_weight': [0.5, 1, 3],
# #     'subsample': [0.8, 0.9, 1.0],
# #     'colsample_bytree': [0.8, 0.9, 1.0],

# #     'gamma': [0, 0.1, 0.2],
# #     'reg_alpha': [0, 0.01, 0.1],
# #     'reg_lambda': [1, 1.5, 2]
# }

# optimal_params = GridSearchCV(xgb_model,
#                                  param_grid=param_dist,
#                                  scoring="roc_auc",
#                                  n_jobs=10,
#                                  cv = 3,
#                                  verbose=1)

# optimal_params.fit(x_train, 
#                    y_train_int,
#                    eval_set = [(x_test, y_test_int)],
#                    verbose = True)

# best_estimator = optimal_params.best_score_


# # Refit with updated parameters
# xgb_model = xgb.XGBClassifier(random_state = 69) 

# xgb_model.fit(x_train, y_train_int)

# prediction = xgb_model.predict(x_val)

# val_accuracy = accuracy_score(y_val_int, prediction) 
# print(val_accuracy, "validation")

# cm = confusion_matrix(y_val_int, prediction, normalize="true")

# labels = gt_train["behaviour"].unique().tolist()
# ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot()
# plt.show() 


# # Model test
# prediction = xgb_model.predict(x_test,
#                                )

# val_accuracy = accuracy_score(y_test_int, prediction) 
# print(val_accuracy, "test")

# cm = confusion_matrix(y_test_int, prediction, normalize="true")

# labels = gt_train["behaviour"].unique().tolist()
# ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot()
# plt.show() 



            # # Get num row and column of main_data
            # total_rows = main_data.shape[0]
            # total_cols = main_data.shape[1]


            # # Filter 
            # if rows_to_keep == 0:
            #     rows_to_keep = frame_size

            # else:
            #     # Calculate the distance between rows to keep
            #     step_size = frame_size / rows_to_keep

            #     # Get the row numbers of rows to keep
            #     selected_indices = [int(i * step_size) for i in range(rows_to_keep)]

            #     # Empty df to store data 
            #     modified_data = pd.DataFrame(np.nan, 
            #                                 index=range(int(total_rows/frame_size*rows_to_keep)), 
            #                                 columns=range(total_cols))

            #     j = 0
            #     for i in range(0, len(main_data), frame_size):

            #         tempdata = main_data.iloc[i:i+frame_size]
            #         tempdata = tempdata.iloc[selected_indices]
            #         modified_data.iloc[j:j+rows_to_keep] = tempdata
            #         j+=rows_to_keep


            #     main_data = modified_data

