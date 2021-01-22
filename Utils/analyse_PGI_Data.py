import os
import pydicom
import csv 
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import cv2
# import pylab as plt

#################################################
# For unpaired data (August/2019)
# Also for paired data (Jan/2020)
def create_csv_file():
    file_path = "/Users/3pi/Documents/Research/CT Scan/PGI_Data/Paired_data/"
    fields = ['S_Num', 'SliceThickness', 'KVP', 'XRayTubeCurrent', 'Image_Shape', 
    'SmallestPixel', 'LargestPixel', 'Modality', 'Manufacturer', 'slice_path']
    filename = "CT_Scan_paired_data_sclices.csv"
    with open(filename, 'w') as csvfile:  
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        slice_num = 0
        for (root,dirs,files) in os.walk(file_path, topdown=True):
            print(root)
            if files:
                for each_file in files:
                    # if each_file.endswith(".dcm"):
                    slice_path = os.path.join(root, each_file)
                    slice_num += 1
                    ds = pydicom.dcmread(slice_path, force=True)
                    try:
                        row = [slice_num, ds.SliceThickness, ds.KVP, ds.XRayTubeCurrent, 
                        ds.pixel_array.shape, np.min(ds.pixel_array), np.max(ds.pixel_array), 
                        ds.Modality, ds.Manufacturer, slice_path]
                        csvwriter.writerow(row)
                    except (AttributeError, OSError, NotImplementedError):
                        continue

#################################################
# For unpaired data (August/2019)
def read_and_analyse_csv_file_1():
    file_path = "/Users/3pi/Documents/Research/CT Scan/Analysis/"
    csvreader = csv.reader(csvfile)
    rows = []
    count = 0
    for row in csvreader:
        if count < 20 and row[1] == '1' and row[2] == '140' and row[3] < '150':
            slice_path = row[9]
            ds = pydicom.dcmread(slice_path)
            if ds.InstanceNumber > 200 and ds.InstanceNumber < 400:
                plt.imsave("./Images_140kV_below_150mA/" + str(row[0]) + ".jpg", ds.pixel_array, cmap=plt.cm.bone)
                file1 = open("Images_140kV_below_150mA.txt","a")
                file1.write(("/".join(slice_path.split("/")[5:])))
                file1.write("\n")
                file1.close()
                count += 1
                if count == 20:
                    break
    print(count)
    # plt.hist(rows)
    # plt.ylabel('Number of Slices')
    # plt.xlabel('XRayTubeCurrent')
    # plt.savefig("./140_hist.png")
    # print(len(rows))
    # rows_set = set(rows)
    # print(rows_set)
    # print(len(rows_set))
    # print(min(rows_set))
    # print(max(rows_set))

    # for k, v in data.items():
    #     print(k)
    #     print(Counter(v))
    # for row in csvreader:
    # 	if row[8] == "SIEMENS":
    # 		count += 1
    # print(count)
    # print("Total no. of rows: %d"%(csvreader.line_num))

#################################################
# For unpaired data (August/2019)
def read_and_analyse_csv_file_2():
    import pathlib
    file_path = "/Users/3pi/Documents/Research/CT Scan/PGI_Data/Paired_data/Charanpreet/DICOM/19091811/08330000/84113600"
    filename, file_extension = os.path.splitext(file_path)
    print(file_extension)
    print(pathlib.Path(file_path).suffix)
    exit()
    ds = pydicom.dcmread(file_path)
    numpy_array = ds.pixel_array
    img_min = np.min(numpy_array)
    img_max = np.max(numpy_array)
    numpy_array = (numpy_array - img_min) / (img_max - img_min)
    print(np.max(numpy_array), np.min(numpy_array))
    cv2.imwrite("./using_cv2.jpg", numpy_array * 255)
    create_csv_file()

#################################################
# For paired data (2/1/2020)
def process_the_data():
    ###### change only this part of the function during every run.
    file_path = "/Users/3pi/Documents/Research/CT Scan/PGI_Data/Paired_data/raw_data/008_Ram kumar"
    sub_num = '007'
    last_img_num = 2178
    start_100KVP = 1
    end_100KVP = 440
    ######
    processed_the_data = "/Users/3pi/Documents/Research/CT Scan/PGI_Data/processed_data"
    for (root,dirs,files) in os.walk(file_path, topdown=True):
        if files:
            for each_file in files:
                # if each_file.endswith(".dcm"):
                slice_path = os.path.join(root, each_file)
                ds = pydicom.dcmread(slice_path, force=True)
                try:
                    if str(ds.SliceThickness) == '1' and str(ds.BodyPartExamined) == 'ABDOMEN' and str(ds.SeriesDescription) == 'Venous Phase  1.0  B30f':

                        # if int(ds.KVP) == 70 and str(ds.SeriesDescription) == 'Abdomen 70  .  1.0  B20f':
                        if int(ds.KVP) == 70:
                            img_num = name_numbers(last_img_num + int(ds.InstanceNumber))
                            instance_num = name_numbers(int(ds.InstanceNumber))
                            img_name = img_num + '_sub_' + sub_num + '_Instance_' + instance_num
                            np.save('/Users/3pi/Documents/Research/CT Scan/PGI_Data/processed_data/70KVP/' + str(img_name) + '.npy', ds.pixel_array)
                            
                        # elif int(ds.KVP) == 80 and str(ds.SeriesDescription) == 'Abdomen 80  .  1.0  B20f':
                        elif int(ds.KVP) == 80:
                            img_num = name_numbers(last_img_num + int(ds.InstanceNumber))
                            instance_num = name_numbers(int(ds.InstanceNumber))
                            img_name = img_num + '_sub_' + sub_num + '_Instance_' + instance_num
                            np.save('/Users/3pi/Documents/Research/CT Scan/PGI_Data/processed_data/80KVP/' + str(img_name) + '.npy', ds.pixel_array)

                        # elif int(ds.KVP) == 100 and str(ds.SeriesDescription) == 'Abdomen 100  .  1.0  B20f' and int(ds.InstanceNumber) > 14:
                        elif int(ds.KVP) == 100 and int(ds.InstanceNumber) > (start_100KVP - 1) and int(ds.InstanceNumber) < (end_100KVP + 1):
                            img_num = name_numbers(last_img_num + int(ds.InstanceNumber) - (start_100KVP - 1))
                            instance_num = name_numbers(int(ds.InstanceNumber))
                            img_name = img_num + '_sub_' + sub_num + '_Instance_' + instance_num 
                            np.save('/Users/3pi/Documents/Research/CT Scan/PGI_Data/processed_data/100KVP/' + str(img_name) + '.npy', ds.pixel_array)

                except (AttributeError, OSError, NotImplementedError):
                    continue

#################################################
# For paired data (2/1/2020)
def name_numbers(number):
    return '0' * (5 - len(str(number))) + str(number)

#################################################
# For paired data (5/1/2020)

def min_max_normalization(img):
    #Volume normaization of each image, x_grad and y_grad.
    img_vol_min = np.min(img)
    img_vol_max = np.max(img)
    img = (img - img_vol_min) / (img_vol_max - img_vol_min) #img.shape == (img_row, slices, img_col : 259, 51, 259)
    
    return img

def normalize_data():
    un_normalized_data_path = "/home/ada/Preethi/PGI_DATA/Paired_Data/processed_data/un_normalized"
    normalized_data_path = "/home/ada/Preethi/PGI_DATA/Paired_Data/processed_data/normalized"
    all_KVPs = sorted(os.listdir(un_normalized_data_path))
    for each_KVP in all_KVPs:
        each_KVP_path = os.path.join(un_normalized_data_path, each_KVP)
        all_subs = sorted(os.listdir(each_KVP_path))
        for each_sub in all_subs:
            each_sub_path = os.path.join(each_KVP_path, each_sub)
            all_npy_files = sorted(os.listdir(each_sub_path))
            each_sub_data = []
            for each_npy in all_npy_files:
                each_npy_path = os.path.join(each_sub_path, each_npy)
                each_sub_data.append(np.load(each_npy_path))
            each_sub_data = np.asarray(each_sub_data)

            # save the normalized data
            each_sub_data_normalized = min_max_normalization(each_sub_data)
            each_norm_sub_path = os.path.join(os.path.join(normalized_data_path, each_KVP), each_sub)
            os.mkdir(each_norm_sub_path)
            for i in range(len(each_sub_data_normalized)):
                np.save(os.path.join(each_norm_sub_path, all_npy_files[i]), each_sub_data_normalized[i])

# process_the_data()
normalize_data()

