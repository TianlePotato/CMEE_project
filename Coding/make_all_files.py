######################################## MODULES ####################################
# Data Processing
import pandas as pd
import numpy as np
import os
import math
from POP3D_Reader.Trial import Trial

# Data Visualisation
import matplotlib.pyplot as plt


# for f in ["Angle", "KP3D", "KP2D"]:

#     for i in [90, 60, 30, 15, 1]:

#         if i == 1:
#             jlist = [1]

#         elif i == 15:
#             jlist = [15, 5]

#         else:
#             jlist = [i, 15, 10, 5]

#         for j in jlist:

#             f_str = str(f)
#             i_str = str(i)
#             j_str = str(j)

#             os.mkdir("Dataset/ML_data/" + f_str + "_" + i_str + "_" + j_str)


""" 
First, get groundtruth behaviour IDs from sampling the 2D keypoint dataset.

This is because the 2D keypoint dataset has the fewest datapoints, as occlusions
cause some keypoints present in 3D dataset to be "NaN" in 2D dataset.

The selected IDs (210 for train, 90 for val, 35 for test) are used for all other datasets.
"""


#################################### PARAMETERS ######################################


all_datasets = ["Train", "Val", "Test"]


# To loop entire thing:
for dataset_type in all_datasets:

    # How many frames to count?
    num_frames = 90
    num_frames_str = str(num_frames)

    # How many to sample in total?
    if dataset_type == "Train":
        sample_num = 210
    elif dataset_type == "Val":
        sample_num = 90
    elif dataset_type == "Test":
        sample_num = 35

    ############################## 0. DATA FILE PATHS #######################################

    # Define the base directory
    base_dir = "/media/tianle/Elements/Dataset-3DPOP"

    # Import Metadata
    metadata = pd.read_csv(
        os.path.join(base_dir, "Dataset/Pop3DMetadata.csv"),
        dtype = str
    )

    # Remove files that are only 1 pigeon, and reset row numbers
    metadata = metadata[metadata['IndividualNum'] != "1"]
    metadata = metadata.reset_index(drop = True)


    # Create empty lists to store file path names for all .csv files
    paths_gt = []
    paths_kp = []

    # Get paths to all behaviour (gt) and keypoint (kp) files
    for i in range(0, len(metadata)):

        seq = str(metadata.at[i, "Sequence"])
        num = str(metadata.at[i, "IndividualNum"])
        date = str(metadata.at[i, "Date"])

        if len(num) != 2:
            num = "0" + num

        # Groundtruth
        path_gt = "Dataset/Pigeon" + num + "/Sequence" + seq + "_n" + num + "_" + date + "/TrainingSplit/" + dataset_type +"/Sequence" + seq + "_n" + num + "_" + date + "-Cam1-Behaviour.csv"
        # 2D keypoints
        path_kp = "Dataset/Pigeon" + num + "/Sequence" + seq + "_n" + num + "_" + date + "/TrainingSplit/" + dataset_type +"/Sequence" + seq + "_n" + num + "_" + date + "-Cam1-Keypoint2DFiltered.csv"

        paths_gt.append(path_gt)
        paths_kp.append(path_kp)


    ##############################################################################################
    ################################ 1. KEYPOINT DATA WRANGLING ##################################
    ##############################################################################################


    # Overall dataframe to store ALL ground truth from files, 18 columns
    data_kp = pd.DataFrame(columns=["leftShoulder_x", "leftShoulder_y",    
                                    "rightShoulder_x", "rightShoulder_y",    
                                    "topKeel_x", "topKeel_y",    
                                    "bottomKeel_x", "bottomKeel_y",    
                                    "tail_x", "tail_y",    
                                    "beak_x", "beak_y",    
                                    "nose_x", "nose_y",    
                                    "leftEye_x", "leftEye_y",    
                                    "rightEye_x", "rightEye_y"])

    ####~~~~ MAIN LOOP ~~~~####
    for i in range(0,len(paths_kp)):

        print("Now processing...", paths_kp[i])

        file_kp = pd.read_csv(
            os.path.join(base_dir, paths_kp[i])
        )


        # Extract metadata associated with file
        seq_num = int(metadata.loc[i,"Sequence"])
        data = Trial("/media/tianle/Elements/Dataset-3DPOP/Dataset", seq_num)

        # Extract pigeon subjects present in metadata, 
        # and filter data to remove non-subjects
        subjects = tuple(data.Subjects)
        file_kp = file_kp.loc[:, file_kp.columns.str.startswith(subjects)]



        # 2. 
        # Loop through each set of 34 columns (all keypoints), 
        # and append every 18/34 columns to the bottom of the overall dataframe.
        for j in range(0, (file_kp.shape[1]), 34):

            # Store 18/34 columns in a temporary dataframe
            # This removes backpack & head markerpoints
            tempdata = file_kp.iloc[:, list(range(j+8, j+18)) + list(range(j+26, j+34))]
        
            # Rename columns
            tempdata.columns = ["leftShoulder_x", "leftShoulder_y",    
                                    "rightShoulder_x", "rightShoulder_y",    
                                    "topKeel_x", "topKeel_y",    
                                    "bottomKeel_x", "bottomKeel_y",    
                                    "tail_x", "tail_y",    
                                    "beak_x", "beak_y",    
                                    "nose_x", "nose_y",    
                                    "leftEye_x", "leftEye_y",    
                                    "rightEye_x", "rightEye_y"]

            # Combine (concatenate) overall dataframe with temporary dataframe
            # pd.dataframe with all kp data (1208728 x 18)
            data_kp = pd.concat([data_kp, tempdata])



    ##################################################################################
    ############################ 2. GROUNDTRUTH DATA WRANGLING #######################
    ##################################################################################


    ##~~~~ Function to FIND CONTINUOUS FRAMES OF BEHAVIOUR ~~~~##

    def find_continuous_frames    (df_behaviours, chunk_len, IDs_removed, num_previous_rows, df_kp):
        """
        Function to find frames of length "chunk_len"
        where the behaviour is continuous
        at least 80% have to be of that behaviour
        """

        # list to store the OVERALL RESULTS
        final = []

        df_length = len(df_behaviours)

        IDs_removed = [ID - num_previous_rows for ID in IDs_removed]
        used_IDs = IDs_removed + list(range((df_length-chunk_len+1), df_length))


        ############################################################################
        # Loop 2 to see which chunks of behaviours are present for continuous frames
        # frames that are continuous for duration are assigned value of 1

        # Keep track of how many rows have been looped through
        row = 0

        while row < df_length:
            # 1. ###########################################
            # Check if row is already used up
            if row in used_IDs:
                row += chunk_len
                final = final + ([0]*chunk_len)

            else:

                # 2. ###########################################
                # Check if behaviour is present
                if df_behaviours.iloc[row] == 0:
                    final.append(0)
                    row += 1

                else:

                    # 3. ###########################################
                    # Check if behaviour is continuous for X frames
                    # Extract X frames as a dataframe
                    behaviours = df_behaviours.iloc[row:row+chunk_len]

                    # Count number of instances where behaviour is == "1", which indicates a behaviour is present
                    num_1 = (behaviours == 1).sum()

                    # If less than 80% are a behaviour, move on
                    if num_1 < (chunk_len*0.7):
                        final.append(0)
                        row += 1

                    # Allow if 80% of chunks contain a given behaviour
                    else:
            
                        # 4. ################################################
                        # Check for excessive NaN in the data
                        # If less than 1/3 of kp datapoints are NaN, proceed
                        # Following code checks NaN for head kp columns and body kp columns separately, as sometimes only one of them has NaN values
                        NaN_sum_body = df_kp.iloc[(num_previous_rows + row):(num_previous_rows + row + chunk_len), 0].isna().sum()
                        NaN_sum_head = df_kp.iloc[(num_previous_rows + row):(num_previous_rows + row + chunk_len), 10].isna().sum()

                        if NaN_sum_body <= (chunk_len/3) and NaN_sum_head <= (chunk_len/3):
                        
                            # Append "1" to list indicating beginning of sample chunk
                            # Append "0"s for the rest of the chunk length, as so future chunks don't overlap with current chunk
                            final = final + [1] + ([0]*(chunk_len-1))

                            # list of rows to be marked as 'used', including all of the current chunk, and before the current chunk as to prevent overlap
                            remove_rows = list(range(row-chunk_len+1, row+chunk_len))
                            used_IDs = used_IDs + remove_rows
                            row += chunk_len

                        # If more than 1/3 of kp datapoints are NaN, move on
                        else:
                            final.append(0)
                            row += 1

                    
        # Trim off any overshoots
        final = final[0:df_length]

        return final


    ##~~~~ Before starting main loop ~~~~##
    # Overall dataframe to store ALL ground truth from files
    data_gt = pd.DataFrame(columns=["ID", "behaviour"])

    # Select which columns to use
    columns=["grooming","courting_status","bowing_status","headdown","feeding","walking","vigilance_status"]

    # Order from fewest to most
    column_order = list(range(0,len(columns)))

    # list to save which IDs have been sampled, and therefore used up in the loop
    Sampled_IDs = []



    ####~~~~ MAIN LOOP ~~~~####
    # 1. ######################################################
    # Focus on 1 column at a time by looping through each behaviour
    for col in column_order:
        print("Now processing column", col)

        # Keep track of num rows (ID)
        rows_looped_thru = 0

        # Overall list to record continuity of a behaviour as "1" or "0"
        full_column = []

        # 2. ######################################################
        # Loop through each file, and extract columns with the behaviour specified in loop 1
        for i in range(0,len(paths_gt)):

            # Read in file
            file_gt = pd.read_csv(
                os.path.join(base_dir, paths_gt[i])
            )

            # Extract metadata associated with file
            seq_num = int(metadata.loc[i,"Sequence"])
            data = Trial("/media/tianle/Elements/Dataset-3DPOP/Dataset", seq_num)

            # Extract pigeon subjects present in metadata, 
            # and filter data to remove non-subjects
            subjects = tuple(data.Subjects)
            file_gt = file_gt.loc[:, file_gt.columns.str.startswith(subjects)]

            # Filter data to contain only desired behaviour
            file_gt = file_gt.loc[:, file_gt.columns.str.endswith(columns[col])]


            # 3. ######################################################
            # Loop through each set of columns,
            # Find continuous frames,
            # And sample.
            # Append sampled columns to the bottom of the overall dataframe.
            for j in range(0, (file_gt.shape[1])):

                # Store every 11 columns in a temporary dataframe
                tempdata = file_gt.iloc[:, j]
                
                # Apply function, which returns list of "1" & "0"
                full_column_len = len(full_column)
                tempdata_filtered = find_continuous_frames(tempdata, num_frames, Sampled_IDs, full_column_len, data_kp)

                # Append list to overall list
                full_column += tempdata_filtered


        # 4. ######################################################
        # Final dataframe for a behaviour
        full_df = pd.DataFrame({
                                "ID": list(range(0, len(full_column))), 
                                "behaviour": [columns[col]] * len(full_column),
                                "continuous": full_column
                                })
        
        # Filter to keep only suitable sample IDs (discard "0")
        full_df_filtered = full_df[full_df["continuous"] == 1]
        print(columns[col], len(full_df_filtered))
        
        # SAMPLING
        sampled_df = full_df_filtered.sample(n=sample_num, random_state=69)

        # now the column with "1" and "0" is not needed as it only contains "1"
        sampled_df = sampled_df.drop("continuous", axis = 1)

        # Append sampled IDs to overall list, so they are not used again
        Sampled_IDs += list(sampled_df["ID"])

        # Combine (concatenate) overall dataframe with temporary dataframe
        data_gt = pd.concat([data_gt, sampled_df], axis=0)


    # Save gt data as csv
    filename = "Dataset/ML_data/KP2D_" + num_frames_str + "_" + num_frames_str + "/" + dataset_type + "_gt_" + num_frames_str + "_" + num_frames_str + ".csv"
    data_gt.to_csv(filename)


    #################################################################################
    ########################## 3. KP DATA INTERPOLATION #############################
    #################################################################################

    # empty dataframe to save final results
    df_interpolated = pd.DataFrame(columns=["leftShoulder_x", "leftShoulder_y",    
                                    "rightShoulder_x", "rightShoulder_y",    
                                    "topKeel_x", "topKeel_y",    
                                    "bottomKeel_x", "bottomKeel_y",    
                                    "tail_x", "tail_y",    
                                    "beak_x", "beak_y",    
                                    "nose_x", "nose_y",    
                                    "leftEye_x", "leftEye_y",    
                                    "rightEye_x", "rightEye_y"])

    # Interpolation loop
    for row in range(0, len(data_gt)):

        # Extract each chunk of behaviour
        row_id = data_gt.iloc[row, 0]
        row_behaviour = data_gt.iloc[row,1]

        ids = list(range(row_id, row_id + num_frames))

        rows_kp = data_kp.iloc[ids, 0:10]
        df_temp_1 = rows_kp.interpolate(method='linear', limit_direction="both", axis=0)

        rows_kp = data_kp.iloc[ids, 10:18]
        df_temp_2 = rows_kp.interpolate(method='linear', limit_direction="both", axis=0)

        df_temp = pd.concat([df_temp_1, df_temp_2], axis = 1)

        df_interpolated = pd.concat([df_interpolated, df_temp])



    ################################## 4. ADDING MOVEMENT #####################################

    # Create dataframe to store movement values
    move_between_frames = pd.DataFrame(np.nan, index=range(int(len(df_interpolated)/num_frames)), columns=range(num_frames-1))

    # keep track of loop iterations
    loop_num = 0

    # Main loop
    for row in range(0, len(df_interpolated),num_frames):

        # Extract columns and rows for bottomKeel (x, y)
        movement_kp = df_interpolated.iloc[row:row+num_frames, 6:8]

        # list to store movement
        temp_df = []

        # Calculate movement of bottomKeel between each frame
        for frame in range(0, num_frames-1):
            pt1 = np.array(movement_kp.iloc[frame])
            pt2 = np.array(movement_kp.iloc[frame+1])
            dist = np.linalg.norm(pt1-pt2)

            temp_df.append(dist)

        # move_avg = np.mean(temp_df)
        # temp_df.append(move_avg)

        move_between_frames.iloc[loop_num] = temp_df
        loop_num += 1


    # Save movement data as csv
    filename = "Dataset/ML_data/KP2D_" + num_frames_str + "_" + num_frames_str + "/" + dataset_type + "_movement_" + num_frames_str + "_" + num_frames_str + ".csv"
    move_between_frames.to_csv(filename)



    ############################ 5. NORMALISATION OF COORDINATES #########################################

    def GetMagnitude(point):
        """Return magnitude of vector"""
        return math.sqrt(point[0]**2 + point[1]**2)

    def GetMidPoint(p1, p2):
        return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

    def GetEucDist(Point1, Point2):
        """Get Euclidean distance, both 2D and 3D"""
        if len(Point1) == 3 and len(Point2) == 3:
            EucDist = math.sqrt((Point1[0] - Point2[0])**2 + (Point1[1] - Point2[1])**2 + (Point1[2] - Point2[2])**2)
        elif len(Point1) == 2 and len(Point2) == 2:
            EucDist = math.sqrt((Point1[0] - Point2[0])**2 + (Point1[1] - Point2[1])**2)
        else:
            raise Exception("point input size error")
        
        return EucDist

    def DefineObj(beak, pt1, pt2):
        """
        Define head object coordinate from 3 points: beak and 2 eyes
        Input: 2D Points, order matters. Beak, Eye1, Eye2; 
        Or Tail, ShoulderR, ShoulderL for body coordinate system!!!
        pt1 = ShoulderR, pt2 = ShoulderL always.
        """
        
        # Get vector from eye to beak
        Vec1 = np.array(pt1) - np.array(beak)
        Vec2 = np.array(pt2) - np.array(beak)
        
        # Get vector between eyes to beak (as the y axis)
        BetweenEye = np.array(GetMidPoint(pt1, pt2))
        ForwardVec = BetweenEye - np.array(beak)
        ForwardUnit = ForwardVec / GetMagnitude(ForwardVec)
        
        # Get horizontal axis (x), normal is (z), mid eye to beak is y
        HorizontalAxis = np.array([ForwardVec[1], -ForwardVec[0]])
        HorizontalUnit = HorizontalAxis / GetMagnitude(HorizontalAxis)
        
        # Calc rotation matrix against principle axes
        Xaxis = np.array([1, 0])
        Yaxis = np.array([0, 1])
        
        # Rotate matrix:
        R = np.array([[np.dot(Xaxis, HorizontalUnit), np.dot(Xaxis, ForwardUnit)],
                    [np.dot(Yaxis, HorizontalUnit), np.dot(Yaxis, ForwardUnit)]])
        
        T = BetweenEye

        return R, T  # returns rotation translation array

    def GetObjCoords(Point2D, R, T):
        """Given R and T and 2D points in world coordinate, get object coordinate
        
        Point2D: 2D points in world coordinates (as an array)

        input must be an array
        """

        # Get points in object coordinate system
        Point2D = np.array(Point2D).reshape(2, 1)
        T = np.array(T).reshape(2, 1)

        Translated = Point2D - T

        # test:
        ObjPointsObj = np.dot(np.linalg.inv(R), Translated).T

        return ObjPointsObj


    # Convert the tail and shoulder coordinates into numpy arrays for processing
    # The normalisation and transformation of data is based on the tail and shoulders
    coord_tail = df_interpolated.loc[:,"tail_x":"tail_y"].to_numpy()
    coord_rightShoulder = df_interpolated.loc[:,"rightShoulder_x":"rightShoulder_y"].to_numpy()
    coord_leftShoulder = df_interpolated.loc[:,"leftShoulder_x":"leftShoulder_y"].to_numpy()


    # Empty lists to store the Rotation[R] and Translation[T] of coordinates
    all_R = []
    all_T = []

    # Loop through each row of the coordinate data, 
    # and apply the DefineObj function to obtain R & T values 
    for i in range(0, len(coord_tail)):

        R, T = DefineObj(beak=coord_tail[i], 
                        pt1=coord_rightShoulder[i], 
                        pt2=coord_leftShoulder[i])
        
        all_R.append(R)
        all_T.append(T)

    # Empty dataframe to store rotated and transformed coordinates
    Dataset_trans = pd.DataFrame(np.nan, index=range(len(df_interpolated)), 
                                columns=["leftShoulder_x", "leftShoulder_y",    
                                    "rightShoulder_x", "rightShoulder_y",    
                                    "topKeel_x", "topKeel_y",    
                                    "bottomKeel_x", "bottomKeel_y",    
                                    "tail_x", "tail_y",    
                                    "beak_x", "beak_y",    
                                    "nose_x", "nose_y",    
                                    "leftEye_x", "leftEye_y",    
                                    "rightEye_x", "rightEye_y"])

    # Loop through each set of 2 columns in the dataset
    # Each 2 cols represent the (x, y) of a given coordinate 
    for i in range(0, 18, 2):

        # Convert to numpy array
        coords = df_interpolated.iloc[:,i:(i+2)].to_numpy()

        temp_data = []

        # Loop through each row of the selected columns
        # And apply the GetObjCoords function
        for j in range(0, len(coords)):

            obj_coords = GetObjCoords(coords[j], all_R[j], all_T[j])
            temp_data.append(obj_coords)

        temp_data = pd.DataFrame(np.squeeze(temp_data))

        Dataset_trans.iloc[:,(i):(i+2)] = temp_data


    # Save as csv
    filename = "Dataset/ML_data/KP2D_" + num_frames_str + "_" + num_frames_str + "/" + dataset_type + "_kp_" + num_frames_str + "_" + num_frames_str + ".csv"
    Dataset_trans.to_csv(filename)



    ######################### 6. Create additional frame lengths ##############################################
    # Create with the same GT sample IDs, behaviour chunks of different frame lengths

    # GT = data_gt
    # KP = Dataset_trans
    # Movement = move_between_frames

    frame_lengths = [90, 60, 30, 15, 1]

    for num_frames in frame_lengths:

        num_frames_str = str(num_frames)

        # New dataframe to store results
        nuevo_kp = pd.DataFrame()
        nuevo_move = pd.DataFrame()

        # modify movement dataset to have 90 columns instead of 89, so that data can be divisible by 15, 10 and 5
        modified_move = move_between_frames
        modified_move["new"] = 0

        # No need to subset for 90 frames
        if num_frames == 90:
            nuevo_kp = Dataset_trans
            nuevo_move = modified_move

        # Subset to 60, 30, 15 or 1 frames
        else:
            # Loop through the full kp dataframe in chunks of 90 rows
            for i in range(0, len(Dataset_trans), 90):

                chunk = Dataset_trans[i:i+90]

                nuevo_kp = pd.concat([nuevo_kp, chunk.iloc[:num_frames]])

            nuevo_move = modified_move.iloc[:,0:num_frames]


        move_name = "Dataset/ML_data/KP2D_" + num_frames_str + "_" + num_frames_str + "/" + dataset_type + "_move_" + num_frames_str + "_" + num_frames_str + ".csv"
        kp_name  = "Dataset/ML_data/KP2D_" + num_frames_str + "_" + num_frames_str + "/" + dataset_type + "_kp_" + num_frames_str + "_" + num_frames_str + ".csv"
        gt_name  = "Dataset/ML_data/KP2D_" + num_frames_str + "_" + num_frames_str + "/" + dataset_type + "_gt_" + num_frames_str + "_" + num_frames_str + ".csv"

        
        nuevo_move.to_csv(move_name)
        nuevo_kp.to_csv(kp_name)
        data_gt.to_csv(gt_name)

        print(kp_name)

        # Subsets
        if num_frames == 1:
            reduced_frames = []

        elif num_frames == 15:
            reduced_frames = [5]

        else:
            reduced_frames = [15, 10, 5]

        # Loop through
        for r in reduced_frames:
                
            # interval to take frames from
            interval = int(num_frames/r)

            # Filter for kp rows
            keep_kp_cols = list(range(0, len(nuevo_kp), interval))
            kp_new = nuevo_kp.iloc[keep_kp_cols]

            # Filter for movement columns
            keep_move_cols = list(range(0, nuevo_move.shape[1], interval))

            # sum the movement of every -interval- columns
            summed_columns = []

            # Iterate through the DataFrame in chunks of 5 columns
            for i in range(0, nuevo_move.shape[1], interval):
                # Sum every 5 consecutive columns
                summed_col = nuevo_move.iloc[:, i:i+interval].sum(axis=1)
                summed_columns.append(summed_col)

            # Create a new DataFrame from the summed columns
            move_new = pd.DataFrame(summed_columns).transpose()

            # Save as csv
            r_str = str(r)
            move_name = "Dataset/ML_data/KP2D_" + num_frames_str + "_" + r_str + "/" + dataset_type + "_move_" + num_frames_str + "_" + r_str + ".csv"
            kp_name  = "Dataset/ML_data/KP2D_" + num_frames_str + "_" + r_str + "/" + dataset_type + "_kp_" + num_frames_str + "_" + r_str + ".csv"
            gt_name  = "Dataset/ML_data/KP2D_" + num_frames_str + "_" + r_str + "/" + dataset_type + "_gt_" + num_frames_str + "_" + r_str + ".csv"

            move_new.to_csv(move_name)
            kp_new.to_csv(kp_name)
            data_gt.to_csv(gt_name)

            print(kp_name)




    """ 

    Then, repeat the process but for 3D keypoints and angle

    """


    #################################### PARAMETERS ######################################
    # To loop entire thing:
    all_datatypes = ["KP3D", "Angle"]
    all_datasets = ["Train", "Val", "Test"]


    for angle_or_kp in all_datatypes:

        # How many frames to count?
        num_frames = 90
        num_frames_str = str(num_frames)



        ############################## 0. DATA FILE PATHS #######################################

        # Define the base directory
        base_dir = "/media/tianle/Elements/Dataset-3DPOP"

        # Import Metadata
        metadata = pd.read_csv(
            os.path.join(base_dir, "Dataset/Pop3DMetadata.csv"),
            dtype = str
        )

        # Remove files that are only 1 pigeon, and reset row numbers
        metadata = metadata[metadata['IndividualNum'] != "1"]
        metadata = metadata.reset_index(drop = True)


        # Create empty lists to store file path names for all .csv files
        paths_gt = []
        paths_kp = []

        # Get paths to all behaviour (gt) and keypoint (kp) files
        for i in range(0, len(metadata)):

            seq = str(metadata.at[i, "Sequence"])
            num = str(metadata.at[i, "IndividualNum"])
            date = str(metadata.at[i, "Date"])

            if len(num) != 2:
                num = "0" + num

            # Groundtruth
            path_gt = "Dataset/Pigeon" + num + "/Sequence" + seq + "_n" + num + "_" + date + "/TrainingSplit/" + dataset_type +"/Sequence" + seq + "_n" + num + "_" + date + "-Cam1-Behaviour.csv"
            # 3D
            path_kp = "Dataset/Pigeon" + num + "/Sequence" + seq + "_n" + num + "_" + date + "/TrainingSplit/" + dataset_type +"/Sequence" + seq + "_n" + num + "_" + date + "-Cam1-Keypoint3D.csv"

            paths_gt.append(path_gt)
            paths_kp.append(path_kp)


        ##############################################################################################
        ################################ 1. KEYPOINT DATA WRANGLING ##################################
        ##############################################################################################


        # Overall dataframe to store ALL ground truth from files, 27 columns
        data_kp = pd.DataFrame(columns=["leftShoulder_x","leftShoulder_y","leftShoulder_z",
                                        "rightShoulder_x","rightShoulder_y","rightShoulder_z",
                                        "topKeel_x","topKeel_y","topKeel_z",
                                        "bottomKeel_x","bottomKeel_y","bottomKeel_z",
                                        "tail_x","tail_y","tail_z",
                                        "beak_x","beak_y","beak_z",
                                        "nose_x","nose_y","nose_z",
                                        "leftEye_x","leftEye_y","leftEye_z",
                                        "rightEye_x","rightEye_y","rightEye_z"])

        ####~~~~ MAIN LOOP ~~~~####
        for i in range(0,len(paths_kp)):

            print("Now processing...", paths_kp[i])

            file_kp = pd.read_csv(
                os.path.join(base_dir, paths_kp[i])
            )


            # 1. Remove unused columns for keyp,oint file...
            remove_cols = ['Unnamed: 0', 'TrainingSplit', 'frame']
                
            # Remove unused columns based on above determination of 
            file_kp = file_kp.drop(columns = remove_cols)



            # 2. 
            # Loop through each set of 51 columns (all keypoints), 
            # and append every 27/51 columns to the bottom of the overall dataframe.
            for i in range(0, (file_kp.shape[1]), 51):

                # Store 27/51 columns in a temporary dataframe
                # This removes backpack & head markerpoints
                tempdata = file_kp.iloc[:, list(range(i+12, i+27)) + list(range(i+39, i+51))]
            
                # Rename columns
                tempdata.columns = ["leftShoulder_x","leftShoulder_y","leftShoulder_z",
                                        "rightShoulder_x","rightShoulder_y","rightShoulder_z",
                                        "topKeel_x","topKeel_y","topKeel_z",
                                        "bottomKeel_x","bottomKeel_y","bottomKeel_z",
                                        "tail_x","tail_y","tail_z",
                                        "beak_x","beak_y","beak_z",
                                        "nose_x","nose_y","nose_z",
                                        "leftEye_x","leftEye_y","leftEye_z",
                                        "rightEye_x","rightEye_y","rightEye_z"]

                # Combine (concatenate) overall dataframe with temporary dataframe
                # pd.dataframe with all kp data (1208728 x 27)
                data_kp = pd.concat([data_kp, tempdata])



        ##################################################################################
        ############################ 2. GROUNDTRUTH DATA WRANGLING #######################
        ##################################################################################

        # use the same data_gt produced by the 2D keypoints 

        # Save as csv
        filename = "Dataset/ML_data/"+ angle_or_kp + "_" + num_frames_str + "_" + num_frames_str + "/" + dataset_type + "_gt_" + num_frames_str + "_" + num_frames_str + ".csv"
        data_gt.to_csv(filename)


        #################################################################################
        ########################## 3. KP DATA INTERPOLATION #############################
        #################################################################################

        # empty dataframe to save final results
        df_interpolated = pd.DataFrame(columns=["leftShoulder_x","leftShoulder_y","leftShoulder_z",
                                        "rightShoulder_x","rightShoulder_y","rightShoulder_z",
                                        "topKeel_x","topKeel_y","topKeel_z",
                                        "bottomKeel_x","bottomKeel_y","bottomKeel_z",
                                        "tail_x","tail_y","tail_z",
                                        "beak_x","beak_y","beak_z",
                                        "nose_x","nose_y","nose_z",
                                        "leftEye_x","leftEye_y","leftEye_z",
                                        "rightEye_x","rightEye_y","rightEye_z"])

        # Interpolation loop
        for row in range(0, len(data_gt)):

            # Extract each chunk of behaviour
            row_id = data_gt.iloc[row, 0]
            row_behaviour = data_gt.iloc[row,1]

            ids = list(range(row_id, row_id + num_frames))

            rows_kp = data_kp.iloc[ids, 0:15]
            df_temp_1 = rows_kp.interpolate(method='linear', limit_direction="both", axis=0)

            rows_kp = data_kp.iloc[ids, 15:27]
            df_temp_2 = rows_kp.interpolate(method='linear', limit_direction="both", axis=0)

            df_temp = pd.concat([df_temp_1, df_temp_2], axis = 1)

            df_interpolated = pd.concat([df_interpolated, df_temp])



        ################################## 4. ADDING MOVEMENT #####################################

        # Create dataframe to store movement values
        move_between_frames = pd.DataFrame(np.nan, index=range(int(len(df_interpolated)/num_frames)), columns=range(num_frames-1))

        # keep track of loop iterations
        loop_num = 0

        # Main loop
        for row in range(0, len(df_interpolated),num_frames):

            # Extract columns and rows for bottomKeel (x, y, z)
            movement_kp = df_interpolated.iloc[row:row+num_frames, 9:12]

            # list to store movement
            temp_df = []

            # Calculate movement of bottomKeel between each frame
            for frame in range(0, num_frames-1):
                pt1 = np.array(movement_kp.iloc[frame])
                pt2 = np.array(movement_kp.iloc[frame+1])
                dist = np.linalg.norm(pt1-pt2)

                temp_df.append(dist)

            move_between_frames.iloc[loop_num] = temp_df
            loop_num += 1

        # Save as csv
        filename = "Dataset/ML_data/"+ angle_or_kp + "_" + num_frames_str + "_" + num_frames_str + "/" + dataset_type + "_movement_" + num_frames_str + "_" + num_frames_str + ".csv"
        move_between_frames.to_csv(filename)



        ############################ 5. NORMALISATION OF COORDINATES #########################################

        ### Functions for normalisation

        def GetMagnitude(point):
            """Return magnitute of vector"""
            return math.sqrt((point[0]**2+point[1]**2+point[2]**2))

        def GetMidPoint(p1, p2):
            return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2,(p1[2]+p2[2])/2]

        def GetEucDist(Point1,Point2):
            """Get euclidian error, both 2D and 3D"""
            
            if len(Point1) ==3 & len(Point2) ==3:
                EucDist =math.sqrt(((Point1[0] - Point2[0]) ** 2) + ((Point1[1] - Point2[1]) ** 2) + ((Point1[2] - Point2[2]) ** 2) )
            elif len(Point1) ==2 & len(Point2) ==2:
                EucDist =math.sqrt(((Point1[0] - Point2[0]) ** 2) + ((Point1[1] - Point2[1]) ** 2))
            else:
                import ipdb;ipdb.set_trace()
                Exception("point input size error")
            
            return EucDist

        def DefineObj(beak,pt1,pt2):
            """
            Define head object coordinate from 3 points: beak and 2 eyes
            Input: 3D Points, order matters. Beak, Eye1, Eye2; 
            Or Tail, ShoulderR, ShoulderL for body coordinate system!!!
            pt1 = ShoulderR, pt2 = ShoulderL always.
            """
            
            ##Get vector from eye to beak
            Vec1 = pt1-beak
            Vec2 = pt2-beak
            PlaneNormal = np.cross(Vec1,Vec2)
            NormalUnit = PlaneNormal/GetMagnitude(PlaneNormal)

            ###get vector of between eye to beak (as the y axis)
            BetweenEye = GetMidPoint(pt1,pt2)
            ForwardVec = BetweenEye-beak
            ForwardUnit = ForwardVec/GetMagnitude(ForwardVec)
            
            #get horizontal axis (x), normal is (z), mid eye to beak is y
            HorizontalAxis = np.cross(PlaneNormal,ForwardVec)
            HorizontalUnit = HorizontalAxis/GetMagnitude(HorizontalAxis)

            ###Calc rotation matrix against principle axes
            Xaxis = np.array([1,0,0])
            Yaxis = np.array([0,1,0])
            Zaxis = np.array([0,0,1])

            ##Rotate matrix:
            R = np.array([[np.dot(Xaxis,HorizontalUnit),-np.dot(Xaxis,ForwardUnit),np.dot(Xaxis,NormalUnit)],
                        [np.dot(Yaxis,HorizontalUnit),-np.dot(Yaxis,ForwardUnit),np.dot(Yaxis,NormalUnit)],
                        [np.dot(Zaxis,HorizontalUnit),-np.dot(Zaxis,ForwardUnit),np.dot(Zaxis,NormalUnit)]])
            
            T = BetweenEye

            return R,T  # returns rotation translation array

        def GetObjCoords(Point3D, R,T):
            """Given R and T and 3d points in world coordinate, get object coordinate
            
                Point3D: 3D points in world coordinates (as an array)

                input must be an array
            """

            ###Get points in object coordinate system:
            # ObjPointsObj = np.dot((Point3D-T),R)

            Point3D = Point3D.reshape(3,1)
            T = np.array(T).reshape(3,1)

            Translated = Point3D-T

            #test:
            ObjPointsObj = np.dot(np.linalg.inv(R),Translated).T

            return ObjPointsObj



        # Convert the tail and shoulder coordinates into numpy arrays for processing
        # The normalisation and transformation of data is based on the tail and shoulders
        coord_tail = df_interpolated.loc[:,"tail_x":"tail_z"].to_numpy()
        coord_rightShoulder = df_interpolated.loc[:,"rightShoulder_x":"rightShoulder_z"].to_numpy()
        coord_leftShoulder = df_interpolated.loc[:,"leftShoulder_x":"leftShoulder_z"].to_numpy()


        # Empty lists to store the Rotation[R] and Translation[T] of coordinates
        all_R = []
        all_T = []

        # Loop through each row of the coordinate data, 
        # and apply the DefineObj function to obtain R & T values 
        for i in range(0, len(coord_tail)):

            R, T = DefineObj(beak=coord_tail[i], 
                            pt1=coord_rightShoulder[i], 
                            pt2=coord_leftShoulder[i])
            
            all_R.append(R)
            all_T.append(T)

        # Empty dataframe to store rotated and transformed coordinates
        Dataset_trans = pd.DataFrame(np.nan, index=range(len(df_interpolated)), 
                                    columns=["leftShoulder_x","leftShoulder_y","leftShoulder_z",
                                            "rightShoulder_x","rightShoulder_y","rightShoulder_z",
                                            "topKeel_x","topKeel_y","topKeel_z",
                                            "bottomKeel_x","bottomKeel_y","bottomKeel_z",
                                            "tail_x","tail_y","tail_z",
                                            "beak_x","beak_y","beak_z",
                                            "nose_x","nose_y","nose_z",
                                            "leftEye_x","leftEye_y","leftEye_z",
                                            "rightEye_x","rightEye_y","rightEye_z"])

        # Loop through each set of 3 columns in the dataset
        # Each 3 cols represent the (x, y, z) of a given coordinate 
        for i in range(0, 27, 3):

            # Convert to numpy array
            coords = df_interpolated.iloc[:,i:(i+3)].to_numpy()

            temp_data = []

            # Loop through each row of the selected columns
            # And apply the GetObjCoords function
            for j in range(0, len(coords)):

                obj_coords = GetObjCoords(coords[j], all_R[j], all_T[j])
                temp_data.append(obj_coords)

            temp_data = pd.DataFrame(np.squeeze(temp_data))

            Dataset_trans.iloc[:,(i):(i+3)] = temp_data


        if angle_or_kp == "KP3D":
            data_final = Dataset_trans

        else:

            ########################## 6. ANGLE CALCULATIONS ##############################

            # Functions for angle calc and normalisation
            def GetMidPoint_vec(p1, p2):
                return (p1 + p2) / 2

            # Function to calculate angle at coord2, assuming the connection of points 1-2-3
            def calculate_angle(coord1, coord2, coord3):
                
                # Calculate vectors
                vector_a = coord1 - coord2
                vector_b = coord3 - coord2
                
                # Calculate dot product and magnitudes of vectors
                dot_product = np.sum(vector_a * vector_b, axis=1)
                magnitude_a = np.linalg.norm(vector_a, axis=1)
                magnitude_b = np.linalg.norm(vector_b, axis=1)
                
                # Calculate the cosine of the angle
                cos_angle = dot_product / (magnitude_a * magnitude_b)
                
                # Calculate the angle in radians
                angle_radians = np.arccos(cos_angle)
                
                # Convert the angle to degrees
                angle_degrees = np.degrees(angle_radians)
                
                return angle_degrees


            # Create new dataframe to store angle data
            # head up & down, neck and head side to side angles
            data_final = pd.DataFrame(0, index=range(len(Dataset_trans)), 
                                    columns=["head_pitch", "head_roll", "head_yaw", 
                                            "neck_pitch", "neck_roll", "neck_yaw"])

            # Pigeon defined as the angles between 4 points - beak, betweenEye, betweenShoulder and tail
            # Calculate point between eyes & shoulder
            EyeR = np.array(Dataset_trans.loc[:, "rightEye_x":"rightEye_z"])
            EyeL = np.array(Dataset_trans.loc[:, "leftEye_x":"leftEye_z"])
            ShoulderR = np.array(Dataset_trans.loc[:, "rightShoulder_x":"rightShoulder_z"])
            ShoulderL = np.array(Dataset_trans.loc[:, "leftShoulder_x":"leftShoulder_z"])

            BetweenEye = GetMidPoint_vec(EyeR, EyeL)
            BetweenShoulder = GetMidPoint_vec(ShoulderR, ShoulderR)


            # Get beak & tail coordinates in np.dataframe format
            beak = np.array(Dataset_trans.loc[:, "beak_x":"beak_z"])
            tail = np.array(Dataset_trans.loc[:, "tail_x":"tail_z"])

            data_final["head_pitch"] = calculate_angle(beak[:,[1,2]], BetweenEye[:,[1,2]], BetweenShoulder[:,[1,2]])
            data_final["head_roll"] = calculate_angle(beak[:,[0,2]], BetweenEye[:,[0,2]], BetweenShoulder[:,[0,2]])
            data_final["head_yaw"] = calculate_angle(beak[:,[0,1]], BetweenEye[:,[0,1]], BetweenShoulder[:,[0,1]])

            data_final["neck_pitch"] = calculate_angle(BetweenEye[:,[1,2]], BetweenShoulder[:,[1,2]], tail[:,[1,2]])
            data_final["neck_roll"] = calculate_angle(BetweenEye[:,[0,2]], BetweenShoulder[:,[0,2]], tail[:,[0,2]])
            data_final["neck_yaw"] = calculate_angle(BetweenEye[:,[0,1]], BetweenShoulder[:,[0,1]], tail[:,[0,1]])



        # Save as csv
        filename = "Dataset/ML_data/"+ angle_or_kp + "_" + num_frames_str + "_" + num_frames_str + "/" + dataset_type + "_kp_" + num_frames_str + "_" + num_frames_str + ".csv"

        data_final.to_csv(filename)




        ######################### 7. Create additional frame lengths ##############################################
        # Create with the same GT sample IDs, behaviour chunks of different frame lengths

        # GT = data_gt
        # KP = data_final
        # Movement = move_between_frames

        frame_lengths = [90, 60, 30, 15, 1]

        for num_frames in frame_lengths:

            num_frames_str = str(num_frames)

            # New dataframe to store results
            nuevo_kp = pd.DataFrame()
            nuevo_move = pd.DataFrame()

            # modify movement dataset to have 90 columns instead of 89, so that data can be divisible by 15, 10 and 5
            modified_move = move_between_frames
            modified_move["new"] = 0

            # No need to subset for 90 frames
            if num_frames == 90:
                nuevo_kp = data_final
                nuevo_move = modified_move

            # Subset to 60, 30, 15 or 1 frames
            else:
                # Loop through the full kp dataframe in chunks of 90 rows
                for i in range(0, len(data_final), 90):

                    chunk = data_final[i:i+90]

                    nuevo_kp = pd.concat([nuevo_kp, chunk.iloc[:num_frames]])

                nuevo_move = modified_move.iloc[:,0:num_frames]


            move_name = "Dataset/ML_data/"+ angle_or_kp + "_" + num_frames_str + "_" + num_frames_str + "/" + dataset_type + "_move_" + num_frames_str + "_" + num_frames_str + ".csv"
            kp_name = "Dataset/ML_data/"+ angle_or_kp + "_" + num_frames_str + "_" + num_frames_str + "/" + dataset_type + "_kp_" + num_frames_str + "_" + num_frames_str + ".csv"
            gt_name = "Dataset/ML_data/"+ angle_or_kp + "_" + num_frames_str + "_" + num_frames_str + "/" + dataset_type + "_gt_" + num_frames_str + "_" + num_frames_str + ".csv"

            
            nuevo_move.to_csv(move_name)
            nuevo_kp.to_csv(kp_name)
            data_gt.to_csv(gt_name)

            print(kp_name)

            # Subsets
            if num_frames == 1:
                reduced_frames = []

            elif num_frames == 15:
                reduced_frames = [5]

            else:
                reduced_frames = [15, 10, 5]

            # Loop through
            for r in reduced_frames:
                    
                # interval to take frames from
                interval = int(num_frames/r)

                # Filter for kp rows
                keep_kp_cols = list(range(0, len(nuevo_kp), interval))
                kp_new = nuevo_kp.iloc[keep_kp_cols]

                # Filter for movement columns
                keep_move_cols = list(range(0, nuevo_move.shape[1], interval))

                # sum the movement of every -interval- columns
                summed_columns = []

                # Iterate through the DataFrame in chunks of 5 columns
                for i in range(0, nuevo_move.shape[1], interval):
                    # Sum every 5 consecutive columns
                    summed_col = nuevo_move.iloc[:, i:i+interval].sum(axis=1)
                    summed_columns.append(summed_col)

                # Create a new DataFrame from the summed columns
                move_new = pd.DataFrame(summed_columns).transpose()

                # Save as csv
                r_str = str(r)

                move_name = "Dataset/ML_data/"+ angle_or_kp + "_" + num_frames_str  + "_" + r_str + "/" + dataset_type + "_move_" + num_frames_str + "_" + r_str + ".csv"
                kp_name = "Dataset/ML_data/"+ angle_or_kp + "_" + num_frames_str  + "_" + r_str + "/" + dataset_type + "_kp_" + num_frames_str + "_" + r_str + ".csv"
                gt_name = "Dataset/ML_data/"+ angle_or_kp + "_" + num_frames_str  + "_" + r_str + "/" + dataset_type + "_gt_" + num_frames_str + "_" + r_str + ".csv"

                move_new.to_csv(move_name)
                kp_new.to_csv(kp_name)
                data_gt.to_csv(gt_name)

                print(kp_name)

