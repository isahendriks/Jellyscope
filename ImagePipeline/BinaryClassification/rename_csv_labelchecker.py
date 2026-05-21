#%%import pandas as pd
import os
import cv2 as cv
import uuid as uuid
from pathlib import Path
import pandas as pd

#%%
def lc_convert(f_batch_log_file):

    df_log = pd.read_csv(f_batch_log_file)
    dir_name = os.path.basename(os.path.dirname(f_batch_log_file))

    # Set up DataFrame columns: BiovolumeHSosik,SurfaceAreaHSosik needs to be left out for this version
    LC_col_str = "Name,Date,Time,CollageFile,ImageFilename,Id,GroupId,Uuid,SrcImage,SrcX,SrcY,ImageX,ImageY,ImageW,ImageH,Timestamp,ElapsedTime,CalConst,CalImage,AbdArea,AbdDiameter,AbdVolume,AspectRatio,AvgBlue,AvgGreen,AvgRed,BiovolumeCylinder,BiovolumePSpheroid,BiovolumeSphere,Ch1Area,Ch1Peak,Ch1Width,Ch2Area,Ch2Ch1Ratio,Ch2Peak,Ch2Width,Ch3Area,Ch3Peak,Ch3Width,CircleFit,Circularity,CircularityHu,Compactness,ConvexPerimeter,Convexity,EdgeGradient,Elongation,EsdDiameter,EsdVolume,FdDiameter,FeretMaxAngle,FeretMinAngle,FiberCurl,FiberStraightness,FilledArea,FilterScore,GeodesicAspectRatio,GeodesicLength,GeodesicThickness,Intensity,Length,Perimeter,Ppc,RatioBlueGreen,RatioRedBlue,RatioRedGreen,Roughness,ScatterArea,ScatterPeak,SigmaIntensity,SphereComplement,SphereCount,SphereUnknown,SphereVolume,SumIntensity,Symmetry,Transparency,Width,Preprocessing,PreprocessingTrue,LabelPredicted,ProbabilityScore,LabelTrue"
    LC_columns = LC_col_str.split(",")
    LC_df = pd.DataFrame(columns=LC_columns)

    for i, row in df_log.iterrows():

        filename = row['image_name']
        crop_width = row['crop_width']
        crop_height = row['crop_height']  
        area = row['region_size_pixels']
        prob_mean = row['region_mean_intensity']
        prob_max = row['region_max_intensity']
          

        # Populate DataFrame
        LC_df.loc[i, 'Name'] = dir_name
        LC_df.loc[i, 'ImageFilename'] = filename # filename of the image
        LC_df.loc[i, 'ImageW'] = crop_width
        LC_df.loc[i, 'ImageH'] = crop_height
        LC_df.loc[i, 'FilledArea'] = area
        LC_df.loc[i, 'AbdArea'] = crop_width * crop_height
        LC_df.loc[i, 'CircleFit'] = prob_mean
        LC_df.loc[i, 'SurfaceAreaMS'] = prob_max
        LC_df.loc[i, 'Uuid'] = uuid.uuid4().hex.upper()
    
    LC_file_name = f"LabelChecker_{dir_name}.csv"

    return LC_file_name, LC_df

#%%
filename_csv = Path("C:\\Users\\Admin\\Documents\\Jellyscope\\Segmentation_results\\Kristineberg_250915\\crop_metadata.csv")

filename_new, df_new = lc_convert(filename_csv)

# Save the new DataFrame to a CSV file
output_path = filename_csv.parent / filename_new
print(f"Saving new CSV file to: {output_path}")
df_new.to_csv(output_path, index=False)
print(f"New CSV file saved to: {output_path}")