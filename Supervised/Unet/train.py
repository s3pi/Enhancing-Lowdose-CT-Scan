import UNET_model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from contextlib import redirect_stdout
from tensorflow.keras.optimizers import RMSprop
import os
import numpy as np 
from sklearn.utils import shuffle
import math
from skimage.metrics import structural_similarity as ssim
import cv2
import random
import time

def make_model():
    # Making all the modules of the model architecture
    i_s = 512
    unet_inp_shape = (i_s,i_s,1)

    # Making the graph by connecting all the modules of the model architecture
    # Each of this model can be seen as a layer now.
    input_tensor = Input(unet_inp_shape)
    
    enc_model = UNET_model.make_encoder_model(unet_inp_shape)
    # enc_model.load_weights("/home/ada/Preethi/CT_Scan/Code/Supervised/Unet/LDCT/Test_Splits_6_2_2/Part_5/save_in_parts_ch_1/Model/enc_model_Last_EpochNum.h5")
    c5, c4, c3, c2, c1 = enc_model(input_tensor)
    
    dec_model = UNET_model.make_decoder_model(c5.shape, c4.shape, c3.shape, c2.shape, c1.shape)
    # dec_model.load_weights("/home/ada/Preethi/CT_Scan/Code/Supervised/Unet/LDCT/Test_Splits_6_2_2/Part_5/save_in_parts_ch_1/Model/dec_model_Last_EpochNum.h5")
    output = dec_model([c5, c4, c3, c2, c1])

    model = Model(inputs=[input_tensor], outputs=[output])

    model.compile(loss= "mean_squared_error", optimizer = RMSprop())

    with open('unet.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

    return enc_model, dec_model, model

def make_train_path_matrices():
    #All these paths are to images in npy format. 
    data_path_mat = []
    label_path_mat = []

    for each_sub in train_subs:
        each_sub_path_data = os.path.join(pro_low_dose_data_path, each_sub)
        all_slices = sorted(os.listdir(each_sub_path_data))
        for each_slice in all_slices:
            each_slice_path = os.path.join(each_sub_path_data, each_slice)
            data_path_mat.append(each_slice_path)

        each_sub_path_label = os.path.join(pro_full_dose_data_path, each_sub)
        all_slices = sorted(os.listdir(each_sub_path_label))
        for each_slice in all_slices:
            each_slice_path = os.path.join(each_sub_path_label, each_slice)
            label_path_mat.append(each_slice_path)

    return data_path_mat, label_path_mat


def make_valid_path_matrices(subs_list):
    data_path_mat = dict()
    label_path_mat = dict()

    for each_sub in subs_list:
        each_sub_imgs_data = []
        each_sub_path_data = os.path.join(pro_low_dose_data_path, each_sub)
        all_slices = sorted(os.listdir(each_sub_path_data))
        for each_slice in all_slices:
            each_slice_path = os.path.join(each_sub_path_data, each_slice)
            each_sub_imgs_data.append(each_slice_path)

        data_path_mat[each_sub] = each_sub_imgs_data

        each_sub_imgs_label = []
        each_sub_path_label = os.path.join(pro_full_dose_data_path, each_sub)
        all_slices = sorted(os.listdir(each_sub_path_label))
        for each_slice in all_slices:
            each_slice_path = os.path.join(each_sub_path_label, each_slice)
            each_sub_imgs_label.append(each_slice_path)

        label_path_mat[each_sub] = each_sub_imgs_label
    
    return data_path_mat, label_path_mat

# def make_train_path_matrices():
#     train_data_path_mat = []
#     train_label_path_mat = []

#     random.shuffle(train_subs)
#     for each_sub in train_subs:     
#         train_data_path_mat.extend(KVP_70_img_path_mat[each_sub])
#         train_label_path_mat.extend(KVP_100_img_path_mat[each_sub])

#     train_data_path_mat = np.asarray(train_data_path_mat)
#     train_label_path_mat = np.asarray(train_label_path_mat)

#     return train_data_path_mat, train_label_path_mat

def name_numbers(length, number):
    return '0' * (length - len(str(number))) + str(number)

def make_directories(local_path, valid_subs):
    for each in ['result_files', 'Images', 'inc_psnr_files', 'Model']:
        file_path = os.path.join(local_path, each)
        try:
            os.mkdir(file_path)
        except (FileExistsError):
            pass      
        if each != 'Model':
            for each_valid_sub in valid_subs:
                # each_valid_sub_num = name_numbers(3, each_valid_sub)
                valid_sub_path = os.path.join(file_path, each_valid_sub)
                try:
                    os.mkdir(valid_sub_path)
                except (FileExistsError):
                   pass 

def find_min_max_of_img(number):
    cumm_sum = 0
    for key, value in num_of_slices_per_sub.items():
        cumm_sum += value
        if number <= cumm_sum:
            return key
    else:
        print("check")

def train_model(model, img_size):
    start_time = time.time()
    
    make_directories(local_path, valid_subs)

    result_files_path = os.path.join(local_path, 'result_files')
    Model_path = os.path.join(local_path, 'Model')
    Images_path = os.path.join(local_path, 'Images')
    inc_psnr_files_path = os.path.join(local_path, 'inc_psnr_files')

    valid_data_path_mat, valid_label_path_mat = make_valid_path_matrices(valid_subs)
    valid_avg_psnr_img = open(result_files_path+ '/valid_avg_psnr_img' + '.txt', 'a')
    valid_avg_mse_img = open(result_files_path+ '/valid_avg_mse_img' + '.txt', 'a')

    max_psnr = 0

    for jj in range(numEpochs):
        print('time in secs', time.time() - start_time)
        start_time = time.time()
        # with open(log_file_path, 'a') as log_file:
        #     log_file.write("Running epoch : %d" % jj)

        train_data_path_mat, train_label_path_mat = make_train_path_matrices()
        # Shuffling all the slicess each epoch.
        train_data_path_mat, train_label_path_mat = shuffle(train_data_path_mat, train_label_path_mat, random_state=2)

        # Creating text file to store training loss metrics.
        batch_loss_file = open(result_files_path + '/batch_loss_file' + '.txt', 'a')
        batch_loss_per_epoch_file = open(result_files_path + '/batch_loss_per_epoch' + '.txt', 'a')
    
        batch_loss_per_epoch = 0.0
        num_batches = int(len(train_data_path_mat)/batch_size)

        for batch in range(num_batches):
            batch_train_data = np.zeros((batch_size, img_size, img_size, 1))
            batch_train_label = np.zeros((batch_size, img_size, img_size, 1))
            element_in_batch = 0

            for each_npy in range(batch*batch_size, min((batch+1) * batch_size, len(train_data_path_mat))):
                # sub_num = find_min_max_of_img(batch + (batch * batch_size))
                # min_data_val = min_max_of_subs_KVP_70[sub_num][0]
                # max_data_val = min_max_of_subs_KVP_70[sub_num][1]
                # min_label_val = min_max_of_subs_KVP_100[sub_num][0]
                # max_label_val = min_max_of_subs_KVP_100[sub_num][1]

                batch_train_data[element_in_batch, :, :, 0] = min_max_normalization(np.load(train_data_path_mat[each_npy]))
                batch_train_label[element_in_batch, :, :, 0] = min_max_normalization(np.load(train_label_path_mat[each_npy]))

                element_in_batch += 1
                
            loss = model.train_on_batch(batch_train_data, batch_train_label)

            # with open(log_file_path, 'a') as log_file:
            #     log_file.write('epoch_num: %d batch_num: %d loss: %f\n' % (jj, batch, loss))
            print(('epoch_num: %d batch_num: %d loss: %f\n' % (jj, batch, loss)))

            batch_loss_file.write("%d %d %f\n" % (jj, batch, loss))
            batch_loss_per_epoch += loss
        
        batch_loss_per_epoch = batch_loss_per_epoch / num_batches
        batch_loss_per_epoch_file.write("%d %f\n" % (jj, batch_loss_per_epoch))

        if jj % save_at_every == 0:
            model.save_weights(Model_path + "/EpochNum"+ str(jj) +".h5")
        
        mse_img = []
        psnr_img = []
        mae_img = []
        ssim_val = []
        valid_subs_decoded_imgs = dict()

        
        for each_valid_sub in valid_subs:
            # each_valid_sub_num = name_numbers(3, valid_subs[i])
            valid_sub_path = os.path.join(result_files_path, each_valid_sub)
            valid_psnr_img = open(valid_sub_path + '/valid_psnr_img' + '.txt', 'a')
            valid_mse_img = open(valid_sub_path + '/valid_mse_img' + '.txt', 'a')
            valid_mae_img = open(valid_sub_path + '/valid_mae_img' + '.txt', 'a')
            ssim_file = open(valid_sub_path + '/ssim.txt', 'a')

            valid_data = np.zeros((len(valid_data_path_mat[each_valid_sub]), 512, 512))
            valid_label = np.zeros((len(valid_label_path_mat[each_valid_sub]), 512, 512))

            # min_data_val = min_max_of_subs_KVP_70[valid_subs[i]][0]
            # max_data_val = min_max_of_subs_KVP_70[valid_subs[i]][1]
            # min_label_val = min_max_of_subs_KVP_100[valid_subs[i]][0]
            # max_label_val = min_max_of_subs_KVP_100[valid_subs[i]][1]

            for k in range(len(valid_label_path_mat[each_valid_sub])):
                valid_data[k] = min_max_normalization(np.load(valid_data_path_mat[each_valid_sub][k]))
                valid_label[k] = min_max_normalization(np.load(valid_label_path_mat[each_valid_sub][k]))

            valid_data = valid_data[:, :, :, np.newaxis]
            valid_label = valid_label[:, :, :, np.newaxis]
            decoded_imgs = model.predict(valid_data)
            valid_subs_decoded_imgs[each_valid_sub] = decoded_imgs

            mse_img.append(math.sqrt(np.mean((valid_label[:,:,:,0] - decoded_imgs[:,:,:,0]) ** 2)))
            psnr_img.append(20 * math.log10( 1.0 / (mse_img[-1])))
            mae_img.append(np.mean(np.abs((valid_label[:,:,:,0] - decoded_imgs[:,:,:,0]))))
            ssim_val.append(ssim(decoded_imgs[:,:,:,0], valid_label[:,:,:,0], multichannel=True))

            valid_mse_img.write("%f \n" %(mse_img[-1]))
            valid_psnr_img.write("%f \n" %(psnr_img[-1]))
            valid_mae_img.write("%f \n" %(mae_img[-1]))
            ssim_file.write("SSIM for test_sub_num is %f \n" %(ssim_val[-1]))

            if (jj % save_at_every == 0) or (jj == (numEpochs - 1)):
                for slice_num in [20,40,60,80]:
                    temp = np.zeros([img_size, img_size*4])
                    temp[:img_size,:img_size] = valid_data[slice_num,:,:,0]
                    temp[:img_size,img_size:img_size*2] = valid_label[slice_num,:,:,0]
                    temp[:img_size,img_size*2:img_size*3] = decoded_imgs[slice_num,:,:,0]
                    temp[:img_size,img_size*3:] = abs(decoded_imgs[slice_num,:,:,0] - valid_label[slice_num,:,:,0])
                    temp = temp * 255
                    # each_valid_sub_num = name_numbers(3, valid_subs[i])
                    path = os.path.join(os.path.join(Images_path, each_valid_sub), str(slice_num))
                    try:
                        os.mkdir(path)
                    except OSError:
                        pass

                    cv2.imwrite(path + "/EpochNum"+ str(jj) + "_Slice_num" + str(slice_num) + '.jpg', temp)

        psnr_img = np.asarray(psnr_img)
        avg_psnr = np.mean(psnr_img)
        if avg_psnr > max_psnr:
            valid_avg_psnr_img.write("%f \n" %(avg_psnr))
            avg_mse = np.mean(np.asarray(mse_img))
            valid_avg_mse_img.write("%f \n" %(avg_mse))
            max_psnr = avg_psnr
            enc_model.save_weights(Model_path + "/enc_model_Last_EpochNum.h5")
            dec_model.save_weights(Model_path + "/dec_model_Last_EpochNum.h5")
            for each_valid_sub in valid_subs:
                # each_valid_sub_num = name_numbers(3, valid_subs[i])
                path = os.path.join(inc_psnr_files_path, each_valid_sub)
                np.save(path + "/EpochNum"+ str(jj) + ".npy", valid_subs_decoded_imgs[each_valid_sub][:,:,:,0])


    valid_psnr_img.close()
    valid_mse_img.close()
    valid_mae_img.close()
    valid_avg_mse_img.close()
    valid_avg_psnr_img.close()

    batch_loss_file.close()
    batch_loss_per_epoch_file.close()

def min_max_normalization(img):
    img_min = np.min(img)
    img_max = np.max(img)
    img = (img - img_min) / (img_max - img_min)
    return img

def find_min_max_of_subs():
    min_max_of_subs_KVP_70 = {}
    min_max_of_subs_KVP_100 = {}
    num_of_slices_per_sub = {}

    for i in range(num_of_subs):
        sub_KVP_70 = np.zeros((len(KVP_70_img_path_mat[i]), 512, 512))
        for k in range(len(KVP_70_img_path_mat[i])):
            sub_KVP_70[k] = np.load(KVP_70_img_path_mat[i][k])

        sub_KVP_100 = np.zeros((len(KVP_100_img_path_mat[i]), 512, 512))
        for k in range(len(KVP_100_img_path_mat[i])):
            sub_KVP_100[k] = np.load(KVP_100_img_path_mat[i][k])

        min_max_of_subs_KVP_70[i] = [np.min(sub_KVP_70), np.max(sub_KVP_70)]
        min_max_of_subs_KVP_100[i] = [np.min(sub_KVP_100), np.max(sub_KVP_100)]
        num_of_slices_per_sub[i] = len(KVP_70_img_path_mat[i])

    return (min_max_of_subs_KVP_70, min_max_of_subs_KVP_100, num_of_slices_per_sub)

def extract_imgs_from_model():
    valid_data_path_mat, valid_label_path_mat = make_valid_path_matrices(valid_subs)
    for i in range(len(valid_subs)):
        each_valid_sub_num = name_numbers(3, valid_subs[i])

        valid_data = np.zeros((len(valid_data_path_mat[i]), 512, 512))
        valid_label = np.zeros((len(valid_label_path_mat[i]), 512, 512))

        min_data_val = min_max_of_subs_KVP_70[valid_subs[i]][0]
        max_data_val = min_max_of_subs_KVP_70[valid_subs[i]][1]
        min_label_val = min_max_of_subs_KVP_100[valid_subs[i]][0]
        max_label_val = min_max_of_subs_KVP_100[valid_subs[i]][1]

        for k in range(len(valid_label_path_mat[i])):
            valid_data[k] = min_max_normalization(np.load(valid_data_path_mat[i][k]), min_data_val, max_data_val)
            valid_label[k] = min_max_normalization(np.load(valid_label_path_mat[i][k]), min_label_val, max_label_val)

        valid_data = valid_data[:, :, :, np.newaxis]
        valid_label = valid_label[:, :, :, np.newaxis]
        decoded_imgs = model.predict(valid_data)
        print(decoded_imgs.shape)
        path = os.path.join(unet_outputs_path, each_valid_sub_num)
        np.save(path + ".npy", decoded_imgs[:,:,:,0])

def test_model():
    start_time = time.time()
    
    make_directories(test_results_path, extended_test_subs)

    result_files_path = os.path.join(test_results_path, 'result_files')
    Images_path = os.path.join(test_results_path, 'Images')
    inc_psnr_files_path = os.path.join(test_results_path, 'inc_psnr_files')

    mse_img = []
    psnr_img = []
    mae_img = []
    ssim_val = []
    test_subs_decoded_imgs = dict()

    test_data_path_mat, test_label_path_mat = make_valid_path_matrices(extended_test_subs)
    count = 0
    for each_test_sub in extended_test_subs:
        test_sub_path = os.path.join(result_files_path, each_test_sub)
        test_psnr_img = open(test_sub_path+ '/test_psnr_img' + '.txt', 'a')
        test_mse_img = open(test_sub_path+ '/test_mse_img' + '.txt', 'a')
        test_mae_img = open(test_sub_path+ '/test_mae_img' + '.txt', 'a')
        test_ssim_file = open(test_sub_path + '/test_ssim.txt', 'a')

        test_data = np.zeros((len(test_data_path_mat[each_test_sub]), 512, 512))
        test_label = np.zeros((len(test_label_path_mat[each_test_sub]), 512, 512))

        for k in range(len(test_label_path_mat[each_test_sub])):
            test_data[k] = min_max_normalization(np.load(test_data_path_mat[each_test_sub][k]))
            test_label[k] = min_max_normalization(np.load(test_label_path_mat[each_test_sub][k]))

        test_data = test_data[:, :, :, np.newaxis]
        test_label = test_label[:, :, :, np.newaxis]
        decoded_imgs = model.predict(test_data)
        test_subs_decoded_imgs[each_test_sub] = decoded_imgs

        mse_img.append(math.sqrt(np.mean((test_label[:,:,:,0] - decoded_imgs[:,:,:,0]) ** 2)))
        psnr_img.append(20 * math.log10( 1.0 / (mse_img[-1])))
        mae_img.append(np.mean(np.abs((test_label[:,:,:,0] - decoded_imgs[:,:,:,0]))))
        ssim_val.append(ssim(decoded_imgs[:,:,:,0], test_label[:,:,:,0], multichannel=True))

        test_mse_img.write("%f \n" %(mse_img[-1]))
        test_psnr_img.write("%f \n" %(psnr_img[-1]))
        test_mae_img.write("%f \n" %(mae_img[-1]))
        test_ssim_file.write("SSIM for test_sub_num is %f \n" %(ssim_val[-1]))

        for slice_num in [20,40,60,80]:
            temp = np.zeros([img_size, img_size*4])
            temp[:img_size,:img_size] = test_data[slice_num,:,:,0]
            temp[:img_size,img_size:img_size*2] = test_label[slice_num,:,:,0]
            temp[:img_size,img_size*2:img_size*3] = decoded_imgs[slice_num,:,:,0]
            temp[:img_size,img_size*3:] = abs(decoded_imgs[slice_num,:,:,0] - test_label[slice_num,:,:,0])
            temp = temp * 255

            path = os.path.join(os.path.join(Images_path, each_test_sub), str(slice_num))
            try:
                os.mkdir(path)
            except OSError:
                pass

            cv2.imwrite(path + "_Slice_num" + str(slice_num) + '.jpg', temp)

        count += 1

    psnr_img = np.asarray(psnr_img)
    print(psnr_img)
    avg_psnr = np.mean(psnr_img)
    print(avg_psnr)
    
    # Do while writing paper
    # for each_test_sub in test_subs:
    #     path = os.path.join(inc_psnr_files_path, each_test_sub)
    #     np.save(path + ".npy", test_subs_decoded_imgs[each_test_sub][:,:,:,0])

    test_psnr_img.close()
    test_mse_img.close()
    test_mae_img.close()

#--------------------------------------------------------------------------------------------------------
# Arguments:
pro_low_dose_data_path = "/home/ada/Preethi/CT_Scan/Data/LDCT/Processed_Data/Low_and_Full_Dose/Low_Dose"
pro_full_dose_data_path = "/home/ada/Preethi/CT_Scan/Data/LDCT/Processed_Data/Low_and_Full_Dose/Full_Dose"
numEpochs = 1000
save_at_every = 10
num_of_subs = 40
img_size = 512
train_subs = ['L160', 'L123', 'L145', 'L210', 'L134', 'L186']
valid_subs = ['L058', 'L131']
total_subs_list = os.listdir(pro_low_dose_data_path)
test_subs = ['L170','L004']
extended_test_subs = list(set(total_subs_list) - set(train_subs + valid_subs + test_subs))
# print(len(total_subs_list), len(extended_test_subs))

batch_size = 1
local_path = "/home/ada/Preethi/CT_Scan/Code/Supervised/Unet/LDCT/Test_Splits_6_2_2/Part_5/save_in_parts_ch_1_bs_1"
try:
    os.mkdir(local_path)
except (FileExistsError):
   pass
#--------------------------------------------------------------------------------------------------------

# log_file_path = local_path + "/log_file.txt"
# with open(log_file_path, 'a') as log_file:
#     log_file.write("-----------------------------------------\nProgram Starts")
#--------------------------------------------------------------------------------------------------------

enc_model, dec_model, model = make_model()
train_model(model, img_size = 512)
#--------------------------------------------------------------------------------------------------------
# best_model_path = "/home/ada/Preethi/CT_Scan/Code/Supervised/Unet/LDCT/Split_6_2_2/Valid_Results/Model/Last_EpochNum.h5"
# model = make_model()
# test_results_path = "/home/ada/Preethi/CT_Scan/Code/Supervised/Unet/LDCT/Test_Splits_6_2_2/Part_5/Test_Results"
# try:
#     os.mkdir(test_results_path)
# except (FileExistsError):
#    pass
# test_model()

#--------------------------------------------------------------------------------------------------------
# valid_subs = [4, 5, 6, 7]
# unet_outputs_path = "/home/ada/Preethi/PGI_CT_Scan/Code/Supervised/Unet/Results_high_to_low/unet_outputs"
# extract_imgs_from_model()

