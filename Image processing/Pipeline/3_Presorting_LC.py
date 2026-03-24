
#### PRESORTING IMAGES AND LABEL CHECKER CONVERTION ####

### PACKAGES ### 
import yaml
import sys
from pathlib import Path
import os 
import pandas as pd 
from pandas.errors import EmptyDataError
import cv2 as cv
import uuid as uuid
from tqdm import tqdm
import glob
import numpy as np
from PIL import Image

#============================================================================
#### GET INPUT DATA FROM CONFIG FILE ####
config_path =  Path(sys.argv[1]).resolve()
#config_path =  ""

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

output_folder = Path(config["output_folder"]).resolve()
convert_to_lc = config['convert_to_lc']

#============================================================================
#### FUNCTIONS ####

def lc_convert(f_batch_log_file):

    df_log = pd.read_csv(f_batch_log_file)

    # Set up DataFrame columns: BiovolumeHSosik,SurfaceAreaHSosik needs to be left out for this version
    LC_col_str = "Name,Date,Time,CollageFile,ImageFilename,Id,GroupId,Uuid,SrcImage,SrcX,SrcY,ImageX,ImageY,ImageW,ImageH,Timestamp,ElapsedTime,CalConst,CalImage,AbdArea,AbdDiameter,AbdVolume,AspectRatio,AvgBlue,AvgGreen,AvgRed,BiovolumeCylinder,BiovolumePSpheroid,BiovolumeSphere,Ch1Area,Ch1Peak,Ch1Width,Ch2Area,Ch2Ch1Ratio,Ch2Peak,Ch2Width,Ch3Area,Ch3Peak,Ch3Width,CircleFit,Circularity,CircularityHu,Compactness,ConvexPerimeter,Convexity,EdgeGradient,Elongation,EsdDiameter,EsdVolume,FdDiameter,FeretMaxAngle,FeretMinAngle,FiberCurl,FiberStraightness,FilledArea,FilterScore,GeodesicAspectRatio,GeodesicLength,GeodesicThickness,Intensity,Length,Perimeter,Ppc,RatioBlueGreen,RatioRedBlue,RatioRedGreen,Roughness,ScatterArea,ScatterPeak,SigmaIntensity,SphereComplement,SphereCount,SphereUnknown,SphereVolume,SumIntensity,Symmetry,Transparency,Width,Preprocessing,PreprocessingTrue,LabelPredicted,ProbabilityScore,LabelTrue"
    LC_columns = LC_col_str.split(",")
    LC_df = pd.DataFrame(columns=LC_columns)

    for i, row in df_log.iterrows():

        # extract information from previous step

        img_path = row['image_path']
        filename = row['crop_filename']
        dir_name = os.path.basename(os.path.dirname(img_path))

        img = cv.imread(img_path)
        height, width = img.shape[:2] # Get dimensions
        crop_id = row['roi_id']
        
        # blob parameters
        area = row['area']
        compactness = row['compactness']
        esd_diameter = row['ESD']
        

        # Populate DataFrame
        LC_df.loc[i, 'Name'] = dir_name # name of the folder
        LC_df.loc[i, 'ImageFilename'] = filename # filename of the image
        LC_df.loc[i, 'Id'] = crop_id
        LC_df.loc[i, 'GroupId'] = crop_id
        LC_df.loc[i, 'ImageW'] = width
        LC_df.loc[i, 'ImageH'] = height
        LC_df.loc[i, 'FilledArea'] = area
        LC_df.loc[i, 'AbdArea'] = area
        LC_df.loc[i, 'Compactness'] = compactness    
        LC_df.loc[i, 'EsdDiameter'] = esd_diameter
        LC_df.loc[i, 'AbdDiameter'] = esd_diameter
        LC_df.loc[i, 'Uuid'] = uuid.uuid4().hex.upper()
    
    LC_file_name = f"LabelChecker_{dir_name}.csv"

    return LC_file_name, LC_df

#============================================================================
#### MAIN CODE ####

# LABEL CHECKER CONVERSION 
if convert_to_lc:
    # Search for files matching the pattern
    f_batch_logs = glob.glob(os.path.join(output_folder, 'ROIs', '**', 'f_batch_*_log.csv'), recursive=True)

    for current_log in tqdm(f_batch_logs, desc="Converting batches", total=len(f_batch_logs)):

        # SAFETY: skip empty crop logs
        try:
            log_df = pd.read_csv(current_log)
        except EmptyDataError:
            # File is completely empty — skip
            continue

        # extract directory
        dir_path = os.path.dirname(current_log)

        # create LabelChecker data format
        LC_filename, LC_df = lc_convert(current_log)

        # save Label Checker csv
        to_save = os.path.join(dir_path, LC_filename)
        os.makedirs(os.path.dirname(to_save), exist_ok=True)
        LC_df.to_csv(to_save, index=False)
    
    print('Converted to LabelChecker format')
    print("=" * 40)
    
    

