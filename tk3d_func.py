import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns
from scipy.interpolate import griddata
import os
from scipy.stats import shapiro, levene, rankdata
from statsmodels.formula.api import ols
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm

def data_setup(df, coordinate, side):
    """
    Setup Medilojic pressure values file.

    Args:
        df (df): output df from pd.read_csv(medilojic file path). Pressure values in N/cm^2.
        coordinate (df): df mapping sensor number with x, y coordinates
        side (string): L (left) or R (right) side
        pressure_factor (int): calibration factor for insole size. Default = 1

    Returns:
        df (df): formatted dataframe 
    """
    coord = coordinate[coordinate["side"] == side]
    NcmtoKpa = 10
    if "Sync" in df.columns:
        df.drop(columns="Sync", inplace=True)
    df = df.transpose()
    df.columns = df.iloc[0]
    df.drop(index=["Sensor nummer"], inplace=True)
    df = df.astype(float)
    df = df*NcmtoKpa
    df['sensor number'] = df.index.astype(int)
    df["20% Average"] = df["20% Averarge"]
    df.drop(columns=["20% Averarge"], inplace=True)
    df = pd.merge(left=coord, right=df, left_on="sensor number", right_on="sensor number")

    return df

def trim_data(df, begin_time, end_time):
    """
    Trim dataframe from begin_time to end_time. 

    Args:
        df (df): df to trim
        begin_time (float) : timestamps [s] to begin trimming.
        end_time (float) : timestamps [s] to stop trimming

    Returns:
        trim_df (df): trimmed dataframe 
    """
    fs = 100
    zero_time = 5 # maximum, minimum, 20% Average rows
    start_zero_time = 3 # x, y and sensor number columns
    
    sensors = df["sensor number"]
    x = df["x"]
    y = df["y"]

    trim_df = df.iloc[:,(begin_time*fs+zero_time):(end_time*fs+zero_time)]
    trim_df = pd.concat([sensors, x, y, trim_df], axis=1)

    maximums = [] # recalculate maximum
    minimums = [] # recalculate minimum
    averages = [] # recalculate 20% average

    for i in range(trim_df.shape[0]): # sensor loop
        sensor_val_at_time = trim_df.iloc[i,start_zero_time:].to_list()
        maximums.append(max(sensor_val_at_time)) 
        minimums.append(min(sensor_val_at_time))
        limit = 0.2*maximums[-1] # calculate 20% of max. lower limit to calculate 20% average
        above_20 = [val for val in sensor_val_at_time if val >= limit] 
        averages.append(np.mean(above_20))
        
    trim_df["Maximum"] = maximums
    trim_df["Minimum"] = minimums
    trim_df["20% Average"] = averages
    
    return trim_df


def sensor_region(insole_size):
    """ 
    Separate regions by insole size into 7 regions. 

    Args:
        insole_size (int): size of WLAN sensor insole size. (37, 39, 43 or 45)

    Returns:
        regions_dict (dict): dictionnary of regions (key) and sensors in regions (values).
    """
    switch={
        37: {"med_FF":[56,57,68,69,70,80,81,82,92,93,94,104,105,106,116,117,118],
             "central_FF":[54,55,66,67,78,79,90,91,102,103,114,115],
             "lat_FF":[53,64,65,75,76,77,87,88,89,98,99,100,101,110,111,112,113],
             "med_MF":[127,128,129,130,139,140,141,151,152,153,163,164,165],
             "lat_MF":[122,123,124,125,126,135,136,137,138,148,149,150,160,161,162],
             "med_RF":[175,176,177,187,188,189,199,200,201,211,212,213,223,224],
             "lat_RF":[172,173,174,184,185,186,196,197,198,208,209,210,221,222]},
        39:{"med_FF":[45,57,58,69,70,71,81,82,83,93,94,95,105,106,107],
            "central_FF":[42,43,44,54,55,56,66,67,68,78,79,80,90,91,92,102,103,104],
            "lat_FF":[41,52,53,63,64,65,75,76,77,86,87,88,89,98,99,100,101],
            "med_MF":[115,116,117,118,127,128,129,130,139,140,141,142,151,152,153,154,163,164,165,166],
            "lat_MF":[111,112,113,114,123,124,125,126,135,136,137,138,147,148,149,150,159,160,161,162],
            "med_RF":[175,176,177,178,187,188,189,199,200,201,211,212],
            "lat_RF":[171,172,173,174,183,184,185,186,196,197,198,208,209,210]},
        41:{"med_FF":[45,46,57,58,69,70,71,81,82,83,93,94,95,105,106,107],
             "central_FF":[42,43,44,54,55,56,66,67,68,78,79,80,90,91,92,102,103,104],
             "lat_FF":[41,52,53,63,64,65,75,76,77,86,87,88,89,98,99,100,101],
             "med_MF":[115,116,117,118,127,128,129,130,139,140,141,142,151,152,153,154,163,164,165,166],
             "lat_MF":[110,111,112,113,114,122,123,124,125,126,135,136,137,138,147,148,149,150,159,160,161,162],
             "med_RF":[175,176,177,178,187,188,189,190,199,200,201,202,211,212,213,214,223,224,225],
             "lat_RF":[171,172,173,174,183,184,185,186,195,196,197,198,208,209,210,220,221,222]},
        43:{"med_FF":[33,34,45,46,47,57,58,59,69,70,71,72,81,82,83,84,93,94,95,96,105,106,107],
             "central_FF":[30,31,32,42,43,44,54,55,56,66,67,68,78,79,80,90,91,92,102,103,104],
             "lat_FF":[29,40,41,51,52,53,62,63,64,65,74,75,76,77,86,87,88,89,98,99,100,101],
             "med_MF":[115,116,117,118,119,127,128,129,130,131,139,140,141,142,143,151,152,153,154,155,163,164,165,166,167],
             "lat_MF":[110,111,112,113,114,122,123,124,125,126,135,136,137,138,147,148,149,150,159,160,161,162],
             "med_RF":[175,176,177,178,179,187,188,189,190,191,199,200,201,202,211,212,213,214,223,224,225],
             "lat_RF":[171,172,173,174,183,184,185,186,195,196,197,198,208,209,210,221,222]},
        45:{"med_FF":[21,33,34,35,45,46,47,57,58,59,60,69,70,71,72,81,82,83,84,93,94,95,105,106,107],
             "central_FF":[18,19,20,30,31,32,42,43,44,54,55,56,66,67,68,78,79,80,90,91,92,102,103,104],
             "lat_FF":[17,28,29,39,40,41,50,51,52,53,62,63,64,65,73,74,75,76,77,85,86,87,88,89,97,98,99,100,101],
             "med_MF":[115,116,117,118,127,128,129,130,139,140,141,142,151,152,153,154,163,164,165,166],
             "lat_MF":[110,111,112,113,114,122,123,124,125,126,134,135,136,137,138,146,147,148,149,150,158,159,160,161,162],
             "med_RF":[175,176,177,178,187,188,189,190,199,200,201,202,211,212,213,223,224],
             "lat_RF":[170,171,172,173,174,182,183,184,185,186,194,195,196,197,198,207,208,209,210,220,221,222]},
        }
    return switch.get(insole_size,"Invalid input")


def find_region_by_sensor(regions_dict, sensor_numbers):
    """ 
    Get WLAN sensor insole region from sensor number 

    Args:
        region_dict (dict): size of WLAN sensor insole size. (37, 39, 43 or 45)
        sensor_numbers (list): list of sensor number

    Returns:
        regions (list): list of the sensor's region.
    """
    regions = []
    for n in sensor_numbers:
        for key, values in regions_dict.items():
            if n in values:
                regions.append(key)
    return regions


def batch_process_data(participants_id, insole_design, tests, coordinates):
    """ 
    Setup whole dataset.

    Args:
        participants_id (list): ids of participants
        insole_design (list): tested insoles designs
        tests (list): name of tests

    Returns:
        data_dict (dict): dictionnary of the dataset. 
                          key : participant_id + insole_design + test + side + trial
        AVG_dict (dict): dictionnary of average of 3 trials for each participant, insole and test. 
                         key : participant_id + insole_design + test + side
    """

    data_dict = {}
    AVG_dict = {}

    for id in participants_id:

        if (id == "P3") or (id == "P6"):
            wlan_insole_size = 39
        else:
            wlan_insole_size = 43
                  
        for design in insole_design:
            for test in tests:
                dpath = "Pressure/"+id+"/"+design+"/"+test
                if os.path.exists(dpath):
                    files = [f for f in os.listdir(dpath) if os.path.isfile(os.path.join(dpath, f))]
                    i = 1

                    avg_left = {"20% Average":[],
                                "Maximum":[],
                                "Minimum":[]}
                    avg_right = {"20% Average":[],
                                "Maximum":[],
                                "Minimum":[]}
                    for idx, file in enumerate(files):
                        if "GL" in file: # gait
                            continue
                        elif "CYCL" in file:
                            continue
                        elif "FFT" in file:
                            continue
                        else:
                            if "pressure" in design:
                                mapping = 1
                                shape = 0
                            elif "shape" in design:
                                mapping = 0
                                shape = 1
                            elif "full" in design:
                                mapping = 1
                                shape = 1
                            else:
                                mapping = 0
                                shape = 0
                            if "_L.CSV" in file: # left insole
                                df_left = pd.read_csv(dpath+"/"+file, low_memory=False)

                                df_left = data_setup(df_left, coordinates, "L")
                                #df_left = trim_data(df_left, 5, 10)
                                
                                df_left = df_left[["x", "y", "sensor number", "Minimum", "Maximum", "20% Average"]]
                                region_dict = sensor_region(wlan_insole_size)
                                regions = find_region_by_sensor(region_dict, df_left["sensor number"].to_list())
                                df_left["region"] = regions
                                df_left["shape"] = shape
                                df_left["mapping"] = mapping
                            
                                avg_left["20% Average"].append(df_left["20% Average"].values)
                                avg_left["Maximum"].append(df_left["Maximum"].values)
                                avg_left["Minimum"].append(df_left["Minimum"].values)

                                data_dict[f"{id}_{design}_{test}_L_{i}"] = df_left
                
                            else: # right insole
                                df_right = pd.read_csv(dpath+"/"+file, low_memory=False)
                                df_right = data_setup(df_right, coordinates, "R")
                                #df_right = trim_data(df_right, 5, 10)

                                df_right = df_right[["x", "y", "sensor number", "Minimum", "Maximum", "20% Average"]]
                                region_dict = sensor_region(wlan_insole_size)
                                regions = find_region_by_sensor(region_dict, df_right["sensor number"].to_list())
                                df_right["region"] = regions
                                df_right["shape"] = shape
                                df_right["mapping"] = mapping
                        
                                avg_right["20% Average"].append(df_right["20% Average"].values)
                                avg_right["Maximum"].append(df_right["Maximum"].values)
                                avg_right["Minimum"].append(df_right["Minimum"].values)

                                data_dict[f"{id}_{design}_{test}_R_{i}"] = df_right

                                i = i+1

                                if i == 4:
                                    # average of 3 trials
                                    avg_left_df = pd.concat([df_left[["x", "y", "sensor number", "region", "shape", "mapping"]], pd.Series(np.average(avg_left["20% Average"], axis=0), name="20% Average").to_frame(), pd.Series(np.average(avg_left["Maximum"], axis=0), name="Maximum"), pd.Series(np.average(avg_left["Minimum"], axis=0), name="Minimum")], axis=1)
                                    AVG_dict[f"{id}_{design}_{test}_L"] = avg_left_df
                                    avg_right_df = pd.concat([df_right[["x", "y", "sensor number", "region", "shape", "mapping"]], pd.Series(np.average(avg_right["20% Average"], axis=0), name="20% Average").to_frame(), pd.Series(np.average(avg_right["Maximum"], axis=0), name="Maximum"), pd.Series(np.average(avg_right["Minimum"], axis=0), name="Minimum")], axis=1)
                                    AVG_dict[f"{id}_{design}_{test}_R"] = avg_right_df
                else:
                    continue

    return data_dict, AVG_dict


def calculate_metrics(data_dict, AVG_dict, weight_dict):
    """ 
    Calculate metrics for statistical analysis

    Args:
        data_dict (dict): dictionnary of the dataset. 
                        key : participant_id + insole_design + test + side + trial
        AVG_dict (dict): dictionnary of average of 3 trials for each participant, insole and test. 
                        key : participant_id + insole_design + test + side
    Returns:
        data_dict (dict): dictionnary of the dataset with added metrics
        AVG_dict (dict): dictionnary of average of 3 trials with added metrics
    """

    for key in data_dict.keys():
        key_split = key.split("_")
        data_dict.get(key)["id"] = key_split[0]
        data_dict.get(key)["insole"] = key_split[1]
        data_dict.get(key)["test"] = key_split[2]
        data_dict.get(key)["side"] = key_split[3]
        data_dict.get(key)["trial"] = key_split[4]
        data_dict.get(key)["stdPressure"] = data_dict.get(key)["20% Average"].std()
        data_dict.get(key)["peakPressure"] = data_dict.get(key)["20% Average"].max()
        data_dict.get(key)["meanPressure"] = data_dict.get(key)["20% Average"].mean()
        data_dict.get(key)["cvPressure"] = data_dict.get(key)["stdPressure"]/data_dict.get(key)["meanPressure"]
        data_dict.get(key)["avg_by_weight"] = ((data_dict.get(key)["20% Average"]/10)*1.125)/(weight_dict.get(key_split[0])*9.80665)*100
        data_dict.get(key)["stdPressure_bw"] = data_dict.get(key)["avg_by_weight"].std()
        data_dict.get(key)["peakPressure_bw"] = data_dict.get(key)["avg_by_weight"].max()
        data_dict.get(key)["meanPressure_bw"] = data_dict.get(key)["avg_by_weight"].mean()
        data_dict.get(key)["cvPressure_bw"] = data_dict.get(key)["stdPressure_bw"]/data_dict.get(key)["meanPressure_bw"]

    for key in AVG_dict.keys():
        key_split = key.split("_")
        AVG_dict.get(key)["id"] = key_split[0]
        AVG_dict.get(key)["insole"] = key_split[1]
        AVG_dict.get(key)["test"] = key_split[2]
        AVG_dict.get(key)["side"] = key_split[3]
        AVG_dict.get(key)["stdPressure"] = AVG_dict.get(key)["20% Average"].std()
        AVG_dict.get(key)["peakPressure"] = AVG_dict.get(key)["20% Average"].max()
        AVG_dict.get(key)["meanPressure"] = AVG_dict.get(key)["20% Average"].mean()
        AVG_dict.get(key)["cvPressure"] = AVG_dict.get(key)["stdPressure"]/AVG_dict.get(key)["meanPressure"]

        # calculate difference between insole and barefoot. if positive, increased pressure with insole. if negative, decreased pressure with insole
        AVG_dict.get(key)["difference"] = AVG_dict.get(key)["20% Average"] - AVG_dict.get(f"{key_split[0]}_barefoot_{key_split[2]}_{key_split[3]}")["20% Average"]
        AVG_dict.get(key)["avg_by_weight"] = ((AVG_dict.get(key)["20% Average"]/10)*1.125)/(weight_dict.get(key_split[0])*9.80665)*100
        AVG_dict.get(key)["stdPressure_bw"] = AVG_dict.get(key)["avg_by_weight"].std()
        AVG_dict.get(key)["peakPressure_bw"] = AVG_dict.get(key)["avg_by_weight"].max()
        AVG_dict.get(key)["meanPressure_bw"] = AVG_dict.get(key)["avg_by_weight"].mean()
        AVG_dict.get(key)["cvPressure_bw"] = AVG_dict.get(key)["stdPressure_bw"]/AVG_dict.get(key)["meanPressure_bw"]
        sum_insole = AVG_dict.get(key)["20% Average"].sum()
        
        # calculate cell / sum insole ratio
        AVG_dict.get(key)["ratio"] = AVG_dict.get(key)["20% Average"] / sum_insole
        # calculate cell / sum barefoot ratio. Used to define interval range for pressure mapping customization
        AVG_dict.get(key)["barefoot_ratio"] = AVG_dict.get(f"{key_split[0]}_barefoot_{key_split[2]}_{key_split[3]}")["20% Average"] / AVG_dict.get(f"{key_split[0]}_barefoot_{key_split[2]}_{key_split[3]}")["20% Average"].sum()
        
        # calculate differences in between insoles
        if "pressure" in key:
            AVG_dict.get(key)["pressure_standard_diff"] = AVG_dict.get(key)["20% Average"] - AVG_dict.get(f"{key_split[0]}_standard0709_{key_split[2]}_{key_split[3]}")["20% Average"]
            AVG_dict.get(key)["pressure_shape_diff"] = AVG_dict.get(key)["20% Average"] - AVG_dict.get(f"{key_split[0]}_shape_{key_split[2]}_{key_split[3]}")["20% Average"]
            AVG_dict.get(key)["full_pressure_diff"] = AVG_dict.get(f"{key_split[0]}_full_{key_split[2]}_{key_split[3]}")["20% Average"] - AVG_dict.get(key)["20% Average"]
        elif "shape" in key:
            AVG_dict.get(key)["shape_standard_diff"] = AVG_dict.get(key)["20% Average"] - AVG_dict.get(f"{key_split[0]}_standard0709_{key_split[2]}_{key_split[3]}")["20% Average"]
            AVG_dict.get(key)["full_shape_diff"] = AVG_dict.get(f"{key_split[0]}_full_{key_split[2]}_{key_split[3]}")["20% Average"] - AVG_dict.get(key)["20% Average"]

        elif "full" in key:
            AVG_dict.get(key)["full_standard_diff"] = AVG_dict.get(key)["20% Average"] - AVG_dict.get(f"{key_split[0]}_standard0709_{key_split[2]}_{key_split[3]}")["20% Average"]
            
        # create columns with given interval ranges based on the barefoot ratio for the left insole
        if "_L" in key:
            if "barefoot" in key:
                df = AVG_dict.get(key)
                interval0 = df[(df.barefoot_ratio < 0.005)]
                interval0["interval"] = "0 to 0.005"
                interval1 = df[(df.barefoot_ratio < 0.01) & (df.barefoot_ratio >= 0.005)]
                interval1["interval"] = "0.005 to 0.01"
                interval2 = df[(df.barefoot_ratio < 0.015) & (df.barefoot_ratio >= 0.01)]
                interval2["interval"] = "0.01 to 0.015"
                interval3 = df[(df.barefoot_ratio < 0.02) & (df.barefoot_ratio >= 0.015)]
                interval3["interval"] = "0.015 to 0.02"
                interval4 = df[(df.barefoot_ratio < 0.025) & (df.barefoot_ratio >= 0.02)]
                interval4["interval"] = "0.02 to 0.025"
                interval5 = df[(df.barefoot_ratio < 0.03) & (df.barefoot_ratio >= 0.025)]
                interval5["interval"] = "0.025 to 0.03"
                interval6 = df[(df.barefoot_ratio >= 0.03)]
                interval6["interval"] = "0.03+"
                dfi_L= pd.concat([interval0[["sensor number", "interval"]], interval1[["sensor number", "interval"]], interval2[["sensor number", "interval"]], interval3[["sensor number", "interval"]], interval4[["sensor number", "interval"]], interval5[["sensor number", "interval"]],interval6[["sensor number", "interval"]]])
                dfi_L.sort_values(by=['sensor number'], inplace=True)

            
            AVG_dict.get(key)["interval"] = dfi_L["interval"]

        # same for the right insole
        else:
            if "barefoot" in key:
                df = AVG_dict.get(key)
                interval0 = df[(df.barefoot_ratio < 0.005)]
                interval0["interval"] = "0 to 0.005"
                interval1 = df[(df.barefoot_ratio < 0.01) & (df.barefoot_ratio >= 0.005)]
                interval1["interval"] = "0.005 to 0.01"
                interval2 = df[(df.barefoot_ratio < 0.015) & (df.barefoot_ratio >= 0.01)]
                interval2["interval"] = "0.01 to 0.015"
                interval3 = df[(df.barefoot_ratio < 0.02) & (df.barefoot_ratio >= 0.015)]
                interval3["interval"] = "0.015 to 0.02"
                interval4 = df[(df.barefoot_ratio < 0.025) & (df.barefoot_ratio >= 0.02)]
                interval4["interval"] = "0.02 to 0.025"
                interval5 = df[(df.barefoot_ratio < 0.03) & (df.barefoot_ratio >= 0.025)]
                interval5["interval"] = "0.025 to 0.03"
                interval6 = df[(df.barefoot_ratio >= 0.03)]
                interval6["interval"] = "0.03+"
                dfi_R= pd.concat([interval0[["sensor number", "interval"]], interval1[["sensor number", "interval"]], interval2[["sensor number", "interval"]], interval3[["sensor number", "interval"]], interval4[["sensor number", "interval"]], interval5[["sensor number", "interval"]],interval6[["sensor number", "interval"]]])
                dfi_R.sort_values(by=['sensor number'], inplace=True)

            AVG_dict.get(key)["interval"] = dfi_R["interval"]

    return data_dict, AVG_dict


def display_pressure_from_df(df, name_map, measure, max_val, min_val, title, region, ax, save=False):
    """ 
    Interpolate an heatmap from pressure values and create new dataframe.

    Args:
        df (df): dataframe to display
        name_map: map strings to get new title
        measure (string): measure to display. (20% Average, Maximum, difference, etc.)
        max_val (int): scale maximum value
        min_val (int): scale minimum value
        title (string): titple of heatmap
        region (string): region to display (WF, FF, MF, RF)
        save (bool): True if want to save heatmap as .png and pressure values as .csv
    
    Returns:
        None
    """
    # rename
    for k, v in name_map.items():  
        title = title.replace(k, v)  
  
    # Define levels 
    levels = np.linspace(min_val, max_val, 500)

    # Create grid
    grid_x, grid_y = np.mgrid[
        df.x.min():df.x.max():200j,
        df.y.min():df.y.max():200j
    ]

    # display by regions
    if region == "WF":
        pass
    elif region == "FF":
        df = df.loc[df['region'].str.contains('FF', case=False, na=False)]
    elif region == "MF":
        df = df.loc[df['region'].str.contains('MF', case=False, na=False)]
    elif region == "RF":
        df = df.loc[df['region'].str.contains('RF', case=False, na=False)]

    # Interpolate pressure data onto grid
    grid_z = griddata(
        (df.x, df.y),
        df[measure],
        (grid_x, grid_y),
        method='cubic'
    )


    if "diff" in measure:
        grid_z = np.clip(grid_z, -1000, 1000)
    else:
        grid_z = np.clip(grid_z, 0, 1000)

    contour = ax.contourf(grid_x, grid_y, grid_z, 
                          levels=levels, cmap='jet', 
                          vmin=min_val, vmax=max_val)
    ax.set_title(title, fontsize=10)
    ax.axis('off')

    # Save if requested
    if save:
        plt.savefig(f'Pressure/{title}.png', bbox_inches='tight')
        df.to_csv(f"Pressure/{title}.csv", index=False)

    return contour


def prepare_anova_df(data_dict, insole_design):
    """ 
    Prepare  dataframe for ANOVA analysis from dictionary

    Args:
        data_dict (dict): dictionary for ANOVA analysis

    Returns:
        anova_df (df): dataframe ready for ANOVA analysis
    """

    anova_df = pd.DataFrame()
    for key in data_dict.keys():
        df = data_dict.get(key)
        anova_df = pd.concat([anova_df, df], ignore_index=True, axis=0)

    print(anova_df)

    anova_df = anova_df[(anova_df["insole"] == insole_design[0]) | (anova_df["insole"] == insole_design[1]) | (anova_df["insole"] == insole_design[2]) | (anova_df["insole"] == insole_design[3])] 
    anova_df = anova_df[["shape", "mapping", "id", "insole", "test", "side", "stdPressure", "peakPressure", "meanPressure", "cvPressure", "stdPressure_bw", "peakPressure_bw", "meanPressure_bw", "cvPressure_bw"]].drop_duplicates()

    return anova_df



def run_anova(df_zone, zone, dvs):
    """
    Run four 2‑way ANOVAs (Shape × Mapping) for standing trials on zones:

    Normality handling per DV, per zone:
      1) Fit OLS on raw DV and Shapiro‐test residuals.
      2) If p ≤ 0.05, try log1p(DV). Refit, re‑test.
      3) If still non‑normal, use **rank ANOVA**: replace DV by its ranks and run the same 2‑way OLS;
         flag method='Rank-ANOVA'. (This is a common non‑param alternative for factorial designs.)

    Heteroscedasticity note: we report Levene p‑values in the figure. If it fails,
    results are still produced (classical or rank ANOVA)
    Args:
        df_zone (df): dataframe for ANOVA analysis
        zone (string): zone of foot
        dvs (list[string]): metrics to perform ANOVA analysis

    Returns:
        anova_df (df): dataframe ready for ANOVA analysis
    """
    fig, axes = plt.subplots(len(dvs), 4, figsize=(20, 12), squeeze=False)
    fig.suptitle(f"Two‑way ANOVA (Shape×Mapping) — Zone: {zone}")

    results = []

    for i, dv in enumerate(dvs):
        if dv not in df_zone.columns:
            axes[i, 0].set_visible(False)
            axes[i, 1].set_visible(False)
            axes[i, 2].set_visible(False)
            axes[i, 3].axis('off')
            continue

        # --- 1) BARPLOT (group means) ---
        means = df_zone.groupby(['shape', 'mapping'])[dv].mean().reset_index()
        means.rename(columns={dv: 'Mean'}, inplace=True)
        ax_bar = axes[i, 0]
        sns.barplot(data=means, x='shape', y='Mean', hue='mapping', ax=ax_bar)
        ax_bar.set_title(f'{dv} Group Means')
        if dv in ['meanPressure', 'peakPressure', 'stdPressure']:
            ax_bar.set_ylabel('kPa')
        else:
            ax_bar.set_ylabel(dv)
        for container in ax_bar.containers:
            ax_bar.bar_label(container, fmt='%.2f', label_type='edge', padding=3)

        # --- 2) Fit base model on RAW DV ---
        method_used = 'ANOVA'
        transform = 'raw'
        model = ols(f"{dv} ~ C(shape) + C(mapping) + C(shape):C(mapping)", data=df_zone).fit()
        resid = model.resid

        # Levene for info
        lev_groups = [grp[dv].values for _, grp in df_zone.groupby(['shape', 'mapping'])]
        _, lev_p = levene(*lev_groups)

        # Shapiro
        _, shap_p = shapiro(resid) if len(resid) >= 3 else (np.nan, 1.0)

        if not np.isnan(shap_p) and shap_p <= 0.05:
            if (df_zone[dv] >= 0).all():
                df_zone['_log1p_dv_'] = np.log1p(df_zone[dv])
                model = ols("_log1p_dv_ ~ C(shape) + C(mapping) + C(shape):C(mapping)", data=df_zone).fit()
                resid = model.resid
                _, shap_p2 = shapiro(resid) if len(resid) >= 3 else (np.nan, 1.0)
                if np.isnan(shap_p2) or shap_p2 <= 0.05:
                    # Fall back to rank ANOVA
                    df_zone['_rank_dv_'] = rankdata(df_zone[dv])
                    model = ols("_rank_dv_ ~ C(shape) + C(mapping) + C(shape):C(mapping)", data=df_zone).fit()
                    method_used = 'Rank-ANOVA'
                    transform = 'rank'
                else:
                    method_used = 'ANOVA'
                    transform = 'log1p'
            else:
                # Negative values: skip log, go straight to rank ANOVA
                df_zone['_rank_dv_'] = rankdata(df_zone[dv])
                model = ols("_rank_dv_ ~ C(shape) + C(mapping) + C(shape):C(mapping)", data=df_zone).fit()
                method_used = 'Rank-ANOVA'
                transform = 'rank'

        # --- 3) Residual histogram & QQ ---
        ax_hist = axes[i, 1]
        sns.histplot(model.resid, bins=15, kde=True, ax=ax_hist)
        ax_hist.set_title(f'{dv} Residuals Histogram ({transform})')
        ax_hist.set_xlabel('Residuals')
        ax_hist.set_ylabel('Frequency')

        ax_qq = axes[i, 2]
        qqplot(model.resid, line='s', ax=ax_qq)
        ax_qq.set_title(f'{dv} Residuals Q‑Q ({transform})')

        # --- 4) Compute ANOVA table (Type II) ---
        anova_tbl = sm.stats.anova_lm(model, typ=2)
        anova_tbl.reset_index(inplace=True)
        anova_tbl.insert(0, 'Zone', zone)
        anova_tbl.insert(1, 'DependentVariable', dv)
        anova_tbl['Method'] = method_used
        anova_tbl['Transform'] = transform
        # Partial eta-squared (SS_effect / (SS_effect + SS_error))
        ss_error = float(anova_tbl.loc[anova_tbl['index'].str.lower() == 'residual', 'sum_sq'].values[0]) if 'Residual' in anova_tbl['index'].values else np.nan
        anova_tbl['EtaSq_partial'] = anova_tbl['sum_sq'] / (anova_tbl['sum_sq'] + ss_error) if not np.isnan(ss_error) else np.nan
        # Null decision
        if 'PR(>F)' in anova_tbl:
            anova_tbl['Reject_Null'] = anova_tbl['PR(>F)'].apply(lambda p: 'Yes' if p <= 0.05 else 'No')

        # Notes panel
        ax_note = axes[i, 3]
        ax_note.axis('off')

        note = (
            f"Zone: {zone}\n"
            f"DV: {dv}\n"
            f"Method: {method_used}\n"
            f"Transform: {transform}\n"
            f"Levene p: {lev_p:.3f}\n"
            f"Shapiro p: {shap_p if not np.isnan(shap_p) else float('nan'):.3f}\n"
            f"p-val: \n {anova_tbl['PR(>F)']}"
        )

        ax_note.text(
            0.02, 0.98, note,
            transform=ax_note.transAxes,
            va='top', ha='left', fontsize=10
        )

        results.append(anova_tbl)


        


        # Save per‑zone outputs
        #if results:
        #    out_dir = os.path.dirname("PILOT")
        #    final_df = pd.concat(results, ignore_index=True)
        #    csv_path = os.path.join(out_dir, f"ANOVA_Stand_Analysis_{zone}.csv")
        #    final_df.to_csv(csv_path, index=False)
        #    print(f"[INFO] ANOVA results saved to {csv_path}")
        
        fig.tight_layout()
        #    png_path = os.path.join(out_dir, f"ANOVA_Stand_Summary_{zone}.png")
        #    fig.savefig(png_path, dpi=300)
        #    print(f"[INFO] Summary figure saved to {png_path}")
        #else:
        #    print(f"[ERROR] No ANOVA results generated for zone {zone}.")

    return results


