import os
import pydicom
import shutil
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

def name_numbers(length, number):
    return '0' * (length - len(str(number))) + str(number)

def mkdir(path):
    try:
        os.mkdir(path)
    except (FileExistsError):
        pass

def find_mean_min_max(path):
    files = sorted(os.listdir(path))
    mean_each_sub_min = 0
    mean_each_sub_max = 0
    for each_file in files:
        each_file_path = os.path.join(path, each_file)
        array = np.load(each_file_path)
        mean_each_sub_min += np.min(array)
        mean_each_sub_max += np.max(array)
    
    mean_each_sub_min = int(round(mean_each_sub_min / len(files)))
    mean_each_sub_max = int(round(mean_each_sub_max / len(files)))

    return mean_each_sub_min, mean_each_sub_max

def find_mean_min_max_all_subs(un_norm_path):
    mean_all_subs_min = 0
    mean_all_subs_max = 0
    all_subs = sorted(os.listdir(un_norm_path))
    for each_sub in all_subs:
        each_sub_num = name_numbers(3, each_sub)
        each_sub_path = os.path.join(un_norm_KVP_70_path, each_sub_num)
        mean_each_sub_min, mean_each_sub_max = find_mean_min_max(each_sub_path)
        mean_all_subs_min += mean_each_sub_min
        mean_all_subs_max += mean_each_sub_max

    mean_all_subs_min = int(round(mean_all_subs_min / len(all_subs)))
    mean_all_subs_max = int(round(mean_all_subs_max / len(all_subs)))

    return mean_all_subs_min, mean_all_subs_max

def find_last_img_num(i):
    # Images of all subjects are numbered in order
    last_img_num = 0
    for k in range(i):
        last_img_num += slice_list_100KVP[k][-1]

    return last_img_num

def make_path(name_list):
    temp = name_list[0]
    for i in range(1, len(name_list)):
        temp = os.path.join(temp, name_list[i])
        mkdir(temp)

    return temp

def store_KVP_70(slice_path, last_img_num, each_test_sub_num, instance_num):
    img_num = name_numbers(5, last_img_num + int(instance_num))
    instance_num = name_numbers(5, int(instance_num))
    # img_name = img_num + '_Patient_' + each_test_sub_num + '_Instance_' + instance_num
    img_name = 'Patient_' + each_test_sub_num + '_Instance_' + instance_num

    # Only few slices from the original are important
    orig_dcm_path = make_path([local_path, "test_dicom_imgs", each_test_sub_num, "KVP_70", "orig_dcm"])
    shutil.copy(slice_path, os.path.join(orig_dcm_path, img_name))

    # Save the dcm made from un_norm npy array
    # un_norm_npy_to_dcm_path = make_path([local_path, "test_dicom_imgs", each_test_sub_num, "KVP_70", "un_norm_npy_to_dcm"])
    # new_path = os.path.join(un_norm_npy_to_dcm_path, img_name)
    # shutil.copy(slice_path, new_path)
    # ds = pydicom.dcmread(new_path, force=True)
    # un_norm_npy_path = make_path([un_norm_KVP_70_path, each_test_sub_num, img_name])
    # array = np.load(un_norm_npy_path + '.npy')
    # for n, val in enumerate(ds.pixel_array.flat):
    #     ds.pixel_array.flat[n] = array.flat[n]
    # ds.PixelData = ds.pixel_array.tobytes()
    # ds.save_as(new_path)

    # Save the dcm made from Norm npy array
    # norm_npy_to_dcm_path = make_path([local_path, "test_dicom_imgs", each_test_sub_num, "KVP_70", "norm_npy_to_dcm"])
    # new_path = os.path.join(norm_npy_to_dcm_path, img_name)
    # shutil.copy(slice_path, new_path)
    # ms = pydicom.dcmread(new_path, force=True)
    # norm_npy_path = make_path([norm_KVP_70_path, each_test_sub_num, img_name])
    # max_value = np.max(ms.pixel_array)
    # array = np.load(norm_npy_path + '.npy')
    # for n, val in enumerate(ms.pixel_array.flat):
    #     ms.pixel_array.flat[n] = int(round(array.flat[n] * max_value))
    # ms.PixelData = ms.pixel_array.tobytes()
    # ms.save_as(new_path)

def store_pred_KVP_100(slice_path, last_img_num, each_test_sub_num, instance_num):
    img_num = name_numbers(5, last_img_num + int(instance_num))
    instance_num = name_numbers(5, int(instance_num))
    # img_name = img_num + '_Patient_' + each_test_sub_num + '_Instance_' + instance_num
    img_name = 'Patient_' + each_test_sub_num + '_Instance_' + instance_num

    # Save the 70KVP slice as is first
    pred_npy_to_dcm_path = make_path([local_path, "test_dicom_imgs", each_test_sub_num, "pred_KVP_100"])
    new_path = os.path.join(pred_npy_to_dcm_path, img_name)
    shutil.copy(slice_path, new_path)
    ps = pydicom.dcmread(new_path, force=True)

    # save the pred_100KVP npy into that pixel array of the 70KVP slice
    pred_npy_path = make_path([inc_psnr_files_path, each_test_sub_num, "best.npy"])
    array = np.load(pred_npy_path)[ps.InstanceNumber - 1]

    for n, val in enumerate(ps.pixel_array.flat):
        ps.pixel_array.flat[n] = int(array.flat[n] * 4095.0)

    ps.PixelData = ps.pixel_array.tobytes()
    ps.KVP = "99"
    ps.save_as(new_path)


def store_KVP_100(slice_path, last_img_num, each_test_sub_num, instance_num, start_100KVP):
    img_num = name_numbers(5, last_img_num + int(instance_num) - (start_100KVP - 1))
    instance_num = name_numbers(5, int(instance_num))
    # img_name = img_num + '_Patient_' + each_test_sub_num + '_Instance_' + instance_num
    img_name = 'Patient_' + each_test_sub_num + '_Instance_' + instance_num

    # Only few slices from the original are important
    orig_dcm_path = make_path([local_path, "test_dicom_imgs", each_test_sub_num, "KVP_100", "orig_dcm"])
    shutil.copy(slice_path, os.path.join(orig_dcm_path, img_name))

    # Save the dcm made from un_norm npy array
    # un_norm_npy_to_dcm_path = make_path([local_path, "test_dicom_imgs", each_test_sub_num, "KVP_100", "un_norm_npy_to_dcm"])
    # new_path = os.path.join(un_norm_npy_to_dcm_path, img_name)
    # shutil.copy(slice_path, new_path)
    # ds = pydicom.dcmread(new_path, force=True)
    # un_norm_npy_path = make_path([un_norm_KVP_100_path, each_test_sub_num, img_name])
    # array = np.load(un_norm_npy_path + '.npy')
    # for n, val in enumerate(ds.pixel_array.flat):
    #     ds.pixel_array.flat[n] = array.flat[n]
    # ds.PixelData = ds.pixel_array.tobytes()
    # ds.save_as(new_path)

    # Save the dcm made from Norm npy array
    # norm_npy_to_dcm_path = make_path([local_path, "test_dicom_imgs", each_test_sub_num, "KVP_100", "norm_npy_to_dcm"])
    # new_path = os.path.join(norm_npy_to_dcm_path, img_name)
    # shutil.copy(slice_path, new_path)
    # ms = pydicom.dcmread(new_path, force=True)
    # norm_npy_path = make_path([norm_KVP_100_path, each_test_sub_num, img_name])
    # max_value = np.max(ms.pixel_array)
    # array = np.load(norm_npy_path + '.npy')
    # for n, val in enumerate(ms.pixel_array.flat):
    #     ms.pixel_array.flat[n] = int(round(array.flat[n] * max_value))
    # ms.PixelData = ms.pixel_array.tobytes()
    # ms.save_as(new_path)

def convert_npy_to_dicom():
    # mean_all_subs_min, mean_all_subs_max = find_mean_min_max_all_subs(un_norm_KVP_70_path)
    # mean_all_KVP_70_subs_max = 3245
    for i in test_subs:
        last_img_num = find_last_img_num(i)
        each_test_sub_num = name_numbers(3, i)
        start_100KVP, end_100KVP = slice_list_100KVP[i][:2]

        file_path = os.path.join(raw_data_path, each_test_sub_num)

        for (root,dirs,files) in os.walk(file_path, topdown=True):
            if files:
                for each_file in files:
                    slice_path = os.path.join(root, each_file)
                    ds = pydicom.dcmread(slice_path, force=True)
                    try:
                        if str(ds.SliceThickness) == '1' and str(ds.BodyPartExamined) == 'ABDOMEN':
                            # if int(ds.KVP) == 70 and str(ds.SeriesDescription) == 'Abdomen 70  .  1.0  B20f':
                            if int(ds.KVP) == 70:
                                store_KVP_70(slice_path, last_img_num, each_test_sub_num, ds.InstanceNumber)
                                store_pred_KVP_100(slice_path, last_img_num, each_test_sub_num, ds.InstanceNumber)
                            # if int(ds.KVP) == 100 and str(ds.SeriesDescription) == 'Abdomen 100  .  1.0  B20f' and int(ds.InstanceNumber) > 14:
                            elif int(ds.KVP) == 100 and int(ds.InstanceNumber) > (start_100KVP - 1) and int(ds.InstanceNumber) < (end_100KVP + 1):
                                store_KVP_100(slice_path, last_img_num, each_test_sub_num, ds.InstanceNumber, start_100KVP)

                    except (AttributeError, OSError, NotImplementedError):
                        continue


#--------------------------------------------------------------------------------------------------------
# Arguments:
local_path = "/home/ada/Preethi/PGI_CT_Scan/Paired_Data/Code/unet"
# un_norm_path = "/Users/3pi/Documents/Research/CT Scan/Data/PGI_Data/Paired_data/processed_data/un_norm"
test_subs = [2]
slice_list_100KVP = [[1, 635, 635], [121, 317, 197], [109, 329, 221], [104, 392, 289], [107, 382, 276], [1, 303, 303], [43, 299, 257], [1, 440, 440]]
min_max_of_subs_KVP70 = {0: [0.0, 4095.0], 1: [0.0, 4095.0], 2: [0.0, 4095.0], 3: [0.0, 3072.0], 4: [0.0, 4095.0], 5: [0.0, 4095.0], 6: [0.0, 4091.0], 7: [0.0, 4095.0]}
min_max_of_subs_KVP100 = {0: [0.0, 4095.0], 1: [0.0, 4095.0], 2: [0.0, 4095.0], 3: [0.0, 2514.0], 4: [0.0, 4095.0], 5: [0.0, 4095.0], 6: [0.0, 3388.0], 7: [0.0, 2613.0]}
raw_data_path = "/mnt/Data1/Preethi/CT_Scan_PGI/paired_data/raw_data"
#--------------------------------------------------------------------------------------------------------
# un_norm_KVP_70_path = os.path.join(un_norm_path, "KVP_70")
# un_norm_KVP_100_path = os.path.join(un_norm_path, "KVP_100")
# norm_KVP_70_path = os.path.join(norm_path, "KVP_70")
# norm_KVP_100_path = os.path.join(norm_path, "KVP_100")
inc_psnr_files_path = os.path.join(local_path, 'inc_psnr_files')
convert_npy_to_dicom()


