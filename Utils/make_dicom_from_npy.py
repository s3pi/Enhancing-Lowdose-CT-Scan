'''
1.  Make a folder for test subjects in local path.
2.  Into each test subject there will be 3 folders, 
    2.1 Copy the dicom files from the unnormalized KVP_70 folder
    2.2 Copy the dicom files from the unnormalized KVP_100 folder
    2.2 From the corresponding subject in unnormalized folder, 
        find the np.max of the volume and unnomralize the predicted .npy volumes.
        
        Convert to dicom files.

        Copy dicom files into predicted_KVP_100 folder. 
3. Open with dicom and check if its clear.
'''
import os
import pydicom
import shutil

def name_numbers(length, number):
    return '0' * (length - len(str(number))) + str(number)

def make_folders_to_store(local_path, valid_subs, un_normalized_KVP_70_path, un_normalized_KVP_100_path):
    path = os.path.join(local_path, 'valid_dicom_imgs')
    try:
        os.mkdir(path)
    except (FileExistsError):
        pass
    for i in valid_subs:
        sub_num = name_numbers(3, i)
        each_sub_path = os.path.join(path, sub_num)
        try:
            os.mkdir(each_sub_path)
        except (FileExistsError):
            pass

def store_dicom(valid_subs, valid_dicom_imgs_path, un_normalized_KVP_70_path, un_normalized_KVP_100_path, inc_psnr_files_path):
    # for each_valid_sub_num in valid_subs:
    pass

def convert_npy_to_dicom():
    for i in valid_subs:
        last_img_num = 0
        for k in range(i):
            last_img_num += slice_list_100KVP[k][-1]

        each_valid_sub_num = name_numbers(3, i)
        file_path = os.path.join(raw_data_path, each_valid_sub_num)
        start_100KVP, end_100KVP = slice_list_100KVP[i]
        
        for (root,dirs,files) in os.walk(file_path, topdown=True):
            if files:
                for each_file in files:
                    # if each_file.endswith(".dcm"):
                    slice_path = os.path.join(root, each_file)
                    ds = pydicom.dcmread(slice_path, force=True)
                    try:
                        if str(ds.SliceThickness) == '1' and str(ds.BodyPartExamined) == 'ABDOMEN':
                            # if int(ds.KVP) == 70 and str(ds.SeriesDescription) == 'Abdomen 70  .  1.0  B20f':
                            if int(ds.KVP) == 70:
                                img_num = name_numbers(5, last_img_num + int(ds.InstanceNumber))
                                instance_num = name_numbers(5, int(ds.InstanceNumber))
                                img_name = img_num + '_Patient_' + each_valid_sub_num + '_Instance_' + instance_num
                                new_path = os.path.join(os.path.join(os.path.join(valid_dicom_imgs_path, each_valid_sub_num), 'KVP_70'), img_name)
                                shutil.copy(slice_path, new_path)
                                normalized_slice_path = os.path.join(os.path.join(normalized_KVP_70_path, each_valid_sub_num), img_name)
                                print(normalized_slice_path)
                                exit()
                                #write the corresponding attributes in that sheet.
                                
                                #fill the array with 0 first and fill the predicted values from exact patient, and instance from the inc_psnr_files folder.
                                    #save it in predicted_KVP_100 folder.
                                
                            # elif int(ds.KVP) == 100 and str(ds.SeriesDescription) == 'Abdomen 100  .  1.0  B20f' and int(ds.InstanceNumber) > 14:
                            # elif int(ds.KVP) == 100 and int(ds.InstanceNumber) > (start_100KVP - 1) and int(ds.InstanceNumber) < (end_100KVP + 1):
                            #     img_num = name_numbers(5, last_img_num + int(ds.InstanceNumber) - (start_100KVP - 1))
                            #     instance_num = name_numbers(int(ds.InstanceNumber))
                            #     img_name = img_num + '_Patient_' + patient_num + '_Instance_' + instance_num 
                            #     np.save('/Users/3pi/Documents/Research/CT Scan/PGI_Data/processed_data/100KVP/' + str(img_name) + '.npy', ds.pixel_array)

                            #first count the instances. 
                            #write the corresponding attributes in that sheet.
                            #save the .dcm files as is in KVP_70 folder
                    except (AttributeError, OSError, NotImplementedError):
                        continue


#--------------------------------------------------------------------------------------------------------
# Arguments:
local_path = "/home/ada/Preethi/PGI_DATA/Paired_Data/Code/unet"
un_normalized_KVP_70_path = "/home/ada/Preethi/PGI_DATA/Paired_Data/Data/processed_data/un_norm/KVP_70"
un_normalized_KVP_100_path = "/home/ada/Preethi/PGI_DATA/Paired_Data/Data/processed_data/un_norm/KVP_100"
valid_subs = [2]
#--------------------------------------------------------------------------------------------------------

# make_folders_to_store(local_path, valid_subs, un_normalized_KVP_70_path, un_normalized_KVP_100_path)
inc_psnr_files_path = os.path.join(local_path, 'inc_psnr_files')
valid_dicom_imgs_path = os.path.join(local_path, 'valid_dicom_imgs')
# store_dicom(valid_subs, valid_dicom_imgs_path, un_normalized_KVP_70_path, un_normalized_KVP_100_path, inc_psnr_files_path)

slice_list_100KVP = [[1, 635], [121, 317], [109, 329], [104, 392], [107, 382], [1, 303], [43, 299], [1, 440]]
raw_data_path = "/mnt/Data1/Preethi/CT_Scan_PGI/paired_data/raw_data"
convert_npy_to_dicom()





