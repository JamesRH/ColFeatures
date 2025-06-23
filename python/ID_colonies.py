#!/usr/bin/env python3

##################################################
## {ID_colonies is a tool to easily segment, label and estract information from bacterial colonies from a 136 mm plate}
##################################################
## {Apache}
##################################################
## Author: {Daniela Azucena Garcia Soriano}
## Credits: [{Daniela Azucena Garcia Soriano, Thomas Torring, Jens Vinge Nygaard}]
## License: {Apache}
## Version: {1}.{0}.{1}
## Maintainer: {Daniela Azucena Garcia Soriano}
## Email: {daniela.ags13@gmail.com}
## Status: {Production}
##################################################

#MODULES to use in the analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
# importing math module  
import math 
# Importing the statistics module
#import statistics
#import os
#import glob
#import skimage.io as io
#for text
import cv2
#filter
import skimage.filters
#adjust intensity
from skimage.exposure import rescale_intensity
#morphology
import skimage.morphology
#from scipy import ndimage as ndi
from skimage.measure import label, regionprops, regionprops_table
from skimage import data, util, measure
# Import threshold and gray convertor functions
from skimage.filters import try_all_threshold, threshold_otsu, threshold_multiotsu
from skimage.color import rgb2gray, label2rgb
#watershed
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
#To PCA
from sklearn.preprocessing import StandardScaler
#For clustering
from sklearn.cluster import KMeans

from os import path, makedirs

#FUNCTIONS to use in the analysis
def show_image(image, cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.axis('off')
    plt.show()

import argparse

# Load input data
parser = argparse.ArgumentParser()
parser.add_argument('--file', help = "filename of image")
parser.add_argument('--background', metavar='FILE', help = "filename of background image for subtraction")
parser.add_argument('--outdir', default=False, help = "directory for output files, or else saved to current dir")
parser.add_argument('--outbasename', default=False, metavar='NAME', help = "base name for output files, otherwise the input basename")
parser.add_argument('--cluster', help="integer value of clusters to be use with Kmeans to cluster the data")
parser.add_argument('--color', action='store_true', help="only use color channels in clustering")
args = parser.parse_args()

#READ PICTURES
file = args.file

if args.background != None:
    background = args.background
else:
    background = 0

img = cv2.imread(file, 1)

#Create file name for output
# Proposal to get file name

if args.outbasename == 'None':
    name = ''
elif args.outbasename:
    name = args.outbasename + "_"
else:
    name = path.basename(file).split('.')[0] + "_" 

#make outdir
if args.outdir: makedirs(args.outdir, exist_ok=True)

#convert BGR to RGB
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #cv2 opens as BGR, need RGB

def mergeRGB(image):
    return image[:,:,0]/3 + image[:,:,1]/3 + image[:,:,2]/3
    
#show_image(image)
if background == 1:
    #read background
    back = cv2.imread(background) 
    #convert BGR to RGB
    back = cv2.cvtColor(back, cv2.COLOR_BGR2RGB)
    #substract background
    rgbiAll = mergeRGB(image)
    rgbbAll = mergeRGB(back)
    rgbIAll = (rgbiAll - rgbbAll)
    #rgb_lessbackgroud = (rgbiR - rgbbR)
    #p1, p99 = np.percentile(rgb_lessbackgroud, (1,99))
    #rgbIR = rescale_intensity(rgb_lessbackgroud, in_range=(p1, p99))
else:
    rgbIAll = mergeRGB(image)
    #rgbiR = rgb2gray(image[:,:,0])
    #p1, p99 = np.percentile(rgbiR, (1,99))
    #rgbIR = rescale_intensity(rgbiR, in_range=(p1, p99))
    #show_image(rgbI)

#show_image(rgbIR)

#CONTRAST
p10, p90 = np.percentile(rgbIAll, (10,90))
rgb_constrast = rescale_intensity(rgbIAll, in_range=(p10, p90))

#MASK AND SEGMENTATION
#Get image size
imageSize = rgbIAll.shape
ci = [1030, 1050, 820]
#Generate grid same size as the original image
x = np.arange(0,imageSize[0])-ci[0]
y = np.arange(0,imageSize[1])-ci[1]
xx, yy = np.meshgrid(x, y)
#Generate mask
mask = (xx**2 + yy**2) < ci[2]**2
#Turn mask into integer
mask = mask.astype(int)
#plt.imshow(mask)
rgbI_mask=mask*rgb_constrast
#show_image(rgbI_mask)

#FOR BRIGHT COLONIES
#rgbI_otsu_thr = skimage.filters.threshold_otsu(rgbIR, nbins=256)
#rgbI_otsu_thr
#image_threshold_bright = rgbI_mask >= rgbI_otsu_thr
image_threshold_bright = rgbI_mask >= 200/256
#show_image(image_threshold_bright)
image_local_open = skimage.morphology.binary_opening(image_threshold_bright, footprint=skimage.morphology.disk(5))
image_area_closing = skimage.morphology.area_closing(image_local_open)
binary_image_bright = image_area_closing
#show_image(binary_image_bright)

#FOR DARK COLONIES
#rgbI_otsu_thr = skimage.filters.threshold_otsu(rgbI_mask)
#image_threshold_dark = (rgbI_mask < rgbI_otsu_thr)*mask
image_threshold_dark = rgbI_mask < 70/256
#show_image(image_threshold_dark)
image_local_open = skimage.morphology.binary_opening(image_threshold_dark, footprint=skimage.morphology.disk(5))
image_area_closing = skimage.morphology.area_closing(image_local_open)
binary_image_dark = image_area_closing
#show_image(binary_image_dark)

#FUSED BRIGHT AND DARK COLONIES
binary_image = (binary_image_bright + binary_image_dark) * mask
#show_image(binary_image)

#WATERSHED
#Find distance between colonies
distance = ndi.distance_transform_edt(binary_image)
#Find peaks of max intensity
local_max_coords = peak_local_max(distance, min_distance=30, num_peaks_per_label=1)
local_max_mask = np.zeros(distance.shape, dtype=bool)
local_max_mask[tuple(local_max_coords.T)] = True
markers = measure.label(local_max_mask)
#Segment using watershed
segmented_bact = watershed(-distance, markers, mask=binary_image, watershed_line=True)
segmented_bact_BW = np.array((segmented_bact > 1)*1)
#show_image(segmented_bact_BW)

#save image
#name_image_output = (name+'_output-segmented.png')
#cv2.imwrite(name_image_output, segmented_bact_BW*255)
#cv2.imwrite('image_segIDs.tiff', segmented_bact_BW*255)

#EXTRACT INFORMATION FROM REGIONS

#Merged CHANNEL
#label image for mapping
image_labeled = label(segmented_bact)
# analyze regions
regions = regionprops_table(image_labeled, intensity_image=rgbIAll, properties = ('label','centroid', 'area', 'perimeter',
'equivalent_diameter', 'eccentricity', 'convex_area', 'mean_intensity'))
#turn into data frame for easy access
df = pd.DataFrame(regions)
data_All = df.rename(columns={'mean_intensity':'mean_intensity-All'})

#DATA FILTERING

#Filter first by size and shape
data_region = data_All.loc[(data_All['area']>60) & (data_All['area']<500000) & (data_All['eccentricity'] < 0.5)]

#Filter second by position on the plate
# Calculate indexes to plot a circle and compare with the indexes in mask to remove colonies near or outside the border.
idx = []
label_ID = []
j = 1
th = np.arange(0,2*np.pi,np.pi/5)
for i in range(data_region.shape[0]):
    #Fit a circle
    xunit = data_region.iloc[i].iloc[5]/2 * np.cos(th) + data_region.iloc[i].iloc[1]
    yunit = data_region.iloc[i].iloc[5]/2 * np.sin(th) + data_region.iloc[i].iloc[2]
    #Find within the boundaries. Check ci variable
    y1 = np.array(yunit>200)
    y2 = np.array(yunit<1875)
    x1 = np.array(xunit>200)
    x2 = np.array(xunit<1875)
    xy = y1 & y2 & x1 & x2
    mean_xy = np.mean(np.multiply(xy, 1))
    if mean_xy == 1:
        idx.append(i)
        label_ID.append(j)
        j = j + 1
#Filtered data frame
data_regions_All = data_region.iloc[idx]
data_regions_All.reset_index(drop=True, inplace=True)
data_All = data_regions_All[data_regions_All.columns[1:]]


#RED CHANNEL
rgbIR = image[:,:,0]
#show_image(rgbIG)
#analyze regions
regions = regionprops_table(image_labeled, intensity_image=rgbIR, properties = ('label', 'mean_intensity'))
#turn into data frame for easy access
data = pd.DataFrame(regions)
data = data.rename(columns={'mean_intensity':'mean_intensity-R'})
data_R = data.iloc[idx]
data_R.reset_index(drop=True, inplace=True)

#GREEN CHANNEL
rgbIG = image[:,:,1]
#show_image(rgbIG)
#analyze regions
regions = regionprops_table(image_labeled, intensity_image=rgbIG, properties = ('label', 'mean_intensity'))
#turn into data frame for easy access
data = pd.DataFrame(regions)
data = data.rename(columns={'mean_intensity':'mean_intensity-G'})
data_G = data.iloc[idx]
data_G.reset_index(drop=True, inplace=True)

#BLUE CHANNEL
rgbIB = image[:,:,2]
#show_image(rgbIG)
# analyze regions
regions = regionprops_table(image_labeled, intensity_image=rgbIB, properties = ('label', 'mean_intensity'))
#turn into data frame for easy access
data = pd.DataFrame(regions)
data = data.rename(columns={'mean_intensity':'mean_intensity-B'})
data_B = data.iloc[idx]
data_B.reset_index(drop=True, inplace=True)


#GENERATE UNIQUE LABELS
name_label = []
for i in range(data_regions_All.shape[0]):
    label_str = str(label_ID[i])
    name_label.append(name+'_'+label_str)

temp_df = {'unique_label':name_label, 'label':label_ID}
name_df = pd.DataFrame(data=temp_df)

#JOIN DATA FRAMES
df_All = name_df.join(data_All, how='outer')
df_R = df_All.join(data_R['mean_intensity-R'])
df_RG = df_R.join(data_G['mean_intensity-G'])
df_RGB = df_RG.join(data_B['mean_intensity-B'])
name_df_output = path.join(args.outdir, name+'outputDF.csv')
df_RGB.to_csv(name_df_output)

#MAP COLONIES IN PLATE

#Read image again using cv2
#img = cv2.imread(file)
for i in range(df_RGB.shape[0]) :
    #draw circle
    cv2.circle(img,(round(df_RGB['centroid-1'][i]), round(df_RGB['centroid-0'][i])), round(df_RGB['equivalent_diameter'][i]/2), (0,255,0), 3)
    #draw text, label
    cv2.putText(img, str(df_RGB['label'][i]), (round(df_RGB['centroid-1'][i]), round(df_RGB['centroid-0'][i])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (1, 1, 1), 4)

#show image
#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#save image
name_image_output_ID = path.join(args.outdir, name+'outputID.png')
cv2.imwrite(name_image_output_ID, img)

#print("END")
print("result colonies: %s and .csv" %name_image_output_ID)

#DO CLUSTERING
#Do clustering based on data
if args.color: 
    features = ['mean_intensity-R', 'mean_intensity-G', 'mean_intensity-B']
else:
    features = ['equivalent_diameter', 'eccentricity', 'convex_area', 'mean_intensity-R', 'mean_intensity-G', 'mean_intensity-B']

#features

data = df_RGB.loc[:,features]

#Normalise data as a first step
x = StandardScaler().fit_transform(data)
np.mean(x), np.std(x)

#Clusters
n=int(args.cluster)
clusters = KMeans(n_clusters=n, random_state=10).fit(x)
d = {'cluster_n':clusters.labels_}
clusters_n = pd.DataFrame(data=d)

#Add cluster labels
df_RGB_cluster = df_RGB.join(clusters_n)
name_df_output = path.join(args.outdir, name+'output_clusterDF.csv')
df_RGB_cluster.to_csv(name_df_output)

#Read image again using cv2 
img = cv2.imread(file, 1)
for i in range(df_RGB.shape[0]):
    #draw circle
    cv2.circle(img,(round(df_RGB_cluster['centroid-1'][i]), round(df_RGB_cluster['centroid-0'][i])), round(df_RGB_cluster['equivalent_diameter'][i]/2), (0,255,0), 3)
    #draw text, label 
    cv2.putText(img, str(df_RGB_cluster['cluster_n'][i]), (round(df_RGB_cluster['centroid-1'][i])+30, round(df_RGB_cluster['centroid-0'][i])+10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (1, 1, 1), 4)

#save image
name_image_output_cluster = path.join(args.outdir, name+'output_clusterID.png')
cv2.imwrite(name_image_output_cluster, img)

#print("END")
print("result clusters: %s and .csv" %name_image_output_cluster)
