############################ MODULES ##################################
# Data Processing
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Modelling
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV,cross_val_score
from scipy.stats import randint
import xgboost as xgb

# Plotting
import matplotlib.pyplot as plt


########################### DATA #######################################
final_data = pd.DataFrame(columns=["accuracy", "sample_size", "model"])


folders = ["Best_KP3D_"]
frames = [30]
samplesize = [70] + [90] + list(range(120,199,20)) + list(range(250,1151,100))

for folder in folders:
    print(folder)

    for num_frames in frames:
        print(num_frames)

        for ss in samplesize:
            ss_str = str(ss)
            print(ss)

            # How many frames to count?
            num_frames_str = str(num_frames)

            # Define the base directory
            base_dir = "/media/tianle/Elements/Dataset-3DPOP/Dataset/ML_data/" + folder + num_frames_str + "/" + ss_str

            # Reduce frame count?
            reduced_frames = 30
            rf_int = str(reduced_frames)

            # Dataset
            kp_train = pd.read_csv(os.path.join(base_dir, "Train_kp_"+num_frames_str+".csv"), index_col=0)
            gt_train = pd.read_csv(os.path.join(base_dir, "Train_gt_"+num_frames_str+".csv"), index_col=0)
            move_train = pd.read_csv(os.path.join(base_dir, "Train_movement_"+num_frames_str+".csv"), index_col=0)

            # # Dataset val
            # kp_val = pd.read_csv(os.path.join(base_dir, "Val_kp_"+num_frames_str+".csv"), index_col=0)
            # gt_val = pd.read_csv(os.path.join(base_dir, "Val_gt_"+num_frames_str+".csv"), index_col=0)
            # move_val = pd.read_csv(os.path.join(base_dir, "Val_movement_"+num_frames_str+".csv"), index_col=0)

            # Dataset test
            kp_test = pd.read_csv(os.path.join(base_dir, "Test_kp_"+num_frames_str+".csv"), index_col=0)
            gt_test = pd.read_csv(os.path.join(base_dir, "Test_gt_"+num_frames_str+".csv"), index_col=0)
            move_test = pd.read_csv(os.path.join(base_dir, "Test_movement_"+num_frames_str+".csv"), index_col=0)


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
            x_train = PrepData(kp_train, move_train, reduced_frames)
            #x_val =   PrepData(kp_val,   move_val,   num_frames)
            x_test =  PrepData(kp_test,  move_test,  reduced_frames)

            # Groundtruth data
            y_train = np.array(gt_train["behaviour"])
            #y_val = np.array(gt_val["behaviour"])
            y_test = np.array(gt_test["behaviour"])


            # ############################### Testing random forest #################################################
            # Create empty model
            # model = RandomForestClassifier(random_state=69)

            # # Fit model
            # model = model.fit(x_train, y_train)

            # # Model test
            # prediction = model.predict(x_test)

            # # Get accuracy score
            # val_accuracy = accuracy_score(y_test, prediction) 

            # model = "RF"
            # #results
            # results = pd.DataFrame({"accuracy": val_accuracy, "sample_size": ss_str, "model": model}, index=[0])

            # final_data = pd.concat([final_data, results])

            ############################# Testing XGBoost #####################################################
            # change y labels into integers for XGBoost model
            label_encoder = LabelEncoder()
            y_train_int = label_encoder.fit_transform(y_train)
            #y_val_int = label_encoder.fit_transform(y_val)
            y_test_int = label_encoder.fit_transform(y_test)

            # Model itself
            xgb_model = xgb.XGBClassifier() 

            # Fit model
            xgb_model.fit(x_train, y_train_int)

            # Model test
            prediction = xgb_model.predict(x_test)

            # Get accuracy score
            val_accuracy = accuracy_score(y_test_int, prediction) 

            model = "XG"
            #results
            results = pd.DataFrame({"accuracy": val_accuracy, "sample_size": ss, "model": model}, index=[0])

            final_data = pd.concat([final_data, results])


# colors = {'RF': 'red', 'XG': 'blue'}
# color_values = [colors[cat] for cat in final_data["model"]]


# plt.scatter(final_data["sample_size"], final_data["accuracy"], c=color_values)
# plt.show()