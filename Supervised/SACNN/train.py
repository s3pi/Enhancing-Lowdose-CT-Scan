from tensorflow.keras.layers import Input,Dense,Flatten,Dropout,Reshape,Conv2D,Conv3D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Input, concatenate, Add, Permute
from tensorflow.keras.models import Model, Sequential
from contextlib import redirect_stdout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import BatchNormalization
import os
import numpy as np 
from sklearn.utils import shuffle
import math
from skimage.metrics import structural_similarity as ssim
import cv2
import random
import time
import tensorflow as tf
import copy
import sys
sys.path.insert(1, '/home/ada/Preethi/CT_Scan/Code/Supervised/SACNN/LDCT/Split_6_2_2/Part_5/SA')
sys.path.insert(1, '/home/ada/Preethi/CT_Scan/Code/Supervised/SACNN/LDCT/Split_6_2_2/Part_5/AE')
import train_ae
import train_sa
from keras import backend as K

def make_generator_model(d, h, w, c):
    inp_shape = (d, h, w, c)
    x = Input(inp_shape)
    input_tensor = Input(inp_shape)

    x1 = Conv3D(filters=64, kernel_size=3, padding='same', activation='relu')(input_tensor)
    x2 = Conv3D(filters=32, kernel_size=3, padding='same', activation='relu')(x1)

    sa_model_1 = train_sa.make_model(x2.shape[1], x2.shape[2], x2.shape[3], x2.shape[4])
    x3 = sa_model_1(x2)

    x4 = Conv3D(filters=16, kernel_size=3, padding='same', activation='relu')(x3)
    
    sa_model_2 = train_sa.make_model(x4.shape[1], x4.shape[2], x4.shape[3], x4.shape[4])
    x5 = sa_model_2(x4)

    x6 = Conv3D(filters=32, kernel_size=3, padding='same', activation='relu')(x5)

    sa_model_3 = train_sa.make_model(x6.shape[1], x6.shape[2], x6.shape[3], x6.shape[4])
    x7 = sa_model_3(x6)
    
    x8 = Conv3D(filters=64, kernel_size=3, padding='same', activation='relu')(x7)

    output = Conv3D(filters=1, kernel_size=1, padding='same', activation='relu')(x7)
    
    output = output[:,1,:,:,:]
    
    model = Model(input_tensor, outputs=[output])
    
    return model

def make_model(d, h, w, c):
    g_model = make_generator_model(d, h, w, c)
    g_model.load_weights(best_model_path)

    ae_model = train_ae.make_model((1, 64, 64, 1))
    ae_model.trainable = False
    
    input_tensor = Input((d, h, w, c))
    g_output = g_model(input_tensor)
    output = ae_model(g_output)
    ae_on_g_model = Model(input_tensor, output)

    ae_on_g_model.compile(loss = "mean_squared_error", optimizer = adam)

    with open('sacnn.txt', 'w') as f:
        with redirect_stdout(f):
            ae_on_g_model.summary()

    return g_model, ae_on_g_model

def make_train_path_matrices():
    #All these paths are to images in npy format. 
    data_path_mat = []
    label_path_mat = []

    for each_sub in train_subs:
        each_sub_path_data = os.path.join(SACNN_pro_low_dose_data_path, each_sub)
        all_slices = sorted(os.listdir(each_sub_path_data))
        for each_slice in all_slices:
            each_slice_path = os.path.join(each_sub_path_data, each_slice)
            data_path_mat.append(each_slice_path)

        each_sub_path_label = os.path.join(SACNN_pro_full_dose_data_path, each_sub)
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
        each_sub_path_data = os.path.join(SACNN_pro_valid_low_dose_data_path, each_sub)
        all_slices = sorted(os.listdir(each_sub_path_data))
        for each_slice in all_slices:
            each_slice_path = os.path.join(each_sub_path_data, each_slice)
            each_sub_imgs_data.append(each_slice_path)

        data_path_mat[each_sub] = each_sub_imgs_data

        each_sub_imgs_label = []
        each_sub_path_label = os.path.join(SACNN_pro_valid_full_dose_data_path, each_sub)
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

def train_model():
    start_time = time.time()
    
    make_directories(local_path, valid_subs)

    result_files_path = os.path.join(local_path, 'result_files')
    Model_path = os.path.join(local_path, 'Model')
    Images_path = os.path.join(local_path, 'Images')
    inc_psnr_files_path = os.path.join(local_path, 'inc_psnr_files')

    max_psnr = 0

    train_data_path_mat, train_label_path_mat = make_train_path_matrices()

    for jj in range(numEpochs):
        print('time in secs', time.time() - start_time)
        start_time = time.time()
        # with open(log_file_path, 'a') as log_file:
        #     log_file.write("Running epoch : %d" % jj)

        # Shuffling all the slicess each epoch.
        train_data_path_mat, train_label_path_mat = shuffle(train_data_path_mat, train_label_path_mat, random_state=2)

        # Creating text file to store training loss metrics.
        batch_loss_file = open(result_files_path + '/batch_loss_file' + '.txt', 'a')
        batch_loss_per_epoch_file = open(result_files_path + '/batch_loss_per_epoch' + '.txt', 'a')
    
        batch_loss_per_epoch = 0.0
        num_batches = int(len(train_data_path_mat)/batch_size)
        
        for batch in range(100):
            batch_train_data = np.zeros((batch_size, 3, h, w, 1))
            batch_train_label = np.zeros((batch_size, h, w, 1))

            element_in_batch = 0

            for each_npy in range(batch*batch_size, min((batch+1) * batch_size, len(train_data_path_mat))):
                batch_train_data[element_in_batch, :, :, :, 0] = min_max_normalization(np.load(train_data_path_mat[each_npy]))
                batch_train_label[element_in_batch, :, :, 0] = min_max_normalization(np.load(train_label_path_mat[each_npy])[0,:,:])

                element_in_batch += 1

            ae_model = train_ae.make_model(batch_train_label.shape)
            batch_train_label = ae_model(batch_train_label)
 
            loss = ae_on_g_model.train_on_batch(batch_train_data, batch_train_label)
            
            # with open(log_file_path, 'a') as log_file:
            #     log_file.write('epoch_num: %d batch_num: %d loss: %f\n' % (jj, batch, loss))
            print(('epoch_num: %d batch_num: %d loss: %f\n' % (jj, batch, loss)))

            batch_loss_file.write("%d %d %f\n" % (jj, batch, loss))
            batch_loss_per_epoch += loss
        
        batch_loss_per_epoch = batch_loss_per_epoch / num_batches
        batch_loss_per_epoch_file.write("%d %f\n" % (jj, batch_loss_per_epoch))

        if jj % save_at_every == 0:
            g_model.save_weights(Model_path + "/EpochNum"+ str(jj) +".h5")

        mse_img = []
        psnr_img = []
        mae_img = []
        ssim_val = []
        valid_subs_decoded_imgs = dict()

        valid_data_path_mat, valid_label_path_mat = make_valid_path_matrices(valid_subs)
        valid_avg_psnr_img = open(result_files_path+ '/valid_avg_psnr_img' + '.txt', 'a')
        valid_avg_mse_img = open(result_files_path+ '/valid_avg_mse_img' + '.txt', 'a')
        
        for each_valid_sub in valid_subs:
            # each_valid_sub_num = name_numbers(3, valid_subs[i])
            valid_sub_path = os.path.join(result_files_path, each_valid_sub)
            valid_psnr_img = open(valid_sub_path+ '/valid_psnr_img' + '.txt', 'a')
            valid_mse_img = open(valid_sub_path+ '/valid_mse_img' + '.txt', 'a')
            valid_mae_img = open(valid_sub_path+ '/valid_mae_img' + '.txt', 'a')
            ssim_file = open(valid_sub_path + '/ssim.txt', 'a')

            valid_data = np.zeros((len(valid_data_path_mat[each_valid_sub]), 3, 512, 512))
            valid_label = np.zeros((len(valid_label_path_mat[each_valid_sub]), 1, 512, 512))
            decoded_imgs = np.zeros((len(valid_data_path_mat[each_valid_sub]), 512, 512))

            for k in range(len(valid_label_path_mat[each_valid_sub])):
            # for k in range(1):
                valid_data[k] = min_max_normalization(np.load(valid_data_path_mat[each_valid_sub][k]))
                valid_label[k] = min_max_normalization(np.load(valid_label_path_mat[each_valid_sub][k]))
                
                for i in range(8):
                    for j in range(8):
                        data = np.zeros((3, 64, 64))
                        start_row = i * 64
                        start_col = j * 64
                        data[0,:,:] = valid_data[k][0, start_row:start_row+64, start_col:start_col+64]
                        data[1,:,:] = valid_data[k][1, start_row:start_row+64, start_col:start_col+64]
                        data[2,:,:] = valid_data[k][2, start_row:start_row+64, start_col:start_col+64]
                
                        data = data[np.newaxis, :, :, :, np.newaxis]
                        pred_img = g_model(data)
                        decoded_imgs[k][start_row:start_row+64, start_col:start_col+64] = pred_img[0,:,:,0]

            valid_subs_decoded_imgs[each_valid_sub] = decoded_imgs
            mse_img.append(math.sqrt(np.mean((valid_label[:,0, :,:] - decoded_imgs[:,:,:]) ** 2)))
            psnr_img.append(20 * math.log10( 1.0 / (mse_img[-1])))
            mae_img.append(np.mean(np.abs((valid_label[:,0, :,:] - decoded_imgs[:,:,:]))))
            ssim_val.append(ssim(decoded_imgs[:,:,:], valid_label[:,0, :,:], multichannel=True))
            
            valid_mse_img.write("%f \n" %(mse_img[-1]))
            valid_psnr_img.write("%f \n" %(psnr_img[-1]))
            valid_mae_img.write("%f \n" %(mae_img[-1]))
            ssim_file.write("SSIM for test_sub_num is %f \n" %(ssim_val[-1]))

            if (jj % save_at_every == 0) or (jj == (numEpochs - 1)):
                for slice_num in [0,40,60,80]:
                    temp = np.zeros([op_img_size, op_img_size*4])
                    temp[:op_img_size,:op_img_size] = valid_data[slice_num,1,:,:]
                    temp[:op_img_size,op_img_size:op_img_size*2] = valid_label[slice_num,:,:]
                    temp[:op_img_size,op_img_size*2:op_img_size*3] = decoded_imgs[slice_num,:,:]
                    temp[:op_img_size,op_img_size*3:] = abs(decoded_imgs[slice_num,:,:] - valid_label[slice_num,:,:,0])
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
            g_model.save_weights(Model_path + "/Last_EpochNum.h5")
            
            for each_valid_sub in valid_subs:
                # each_valid_sub_num = name_numbers(3, valid_subs[i])
                path = os.path.join(inc_psnr_files_path, each_valid_sub)
                np.save(path + "/EpochNum"+ str(jj) + ".npy", valid_subs_decoded_imgs[each_valid_sub][:,:,:])


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
        decoded_imgs = g_model.predict(valid_data)
        print(decoded_imgs.shape)
        path = os.path.join(unet_outputs_path, each_valid_sub_num)
        np.save(path + ".npy", decoded_imgs[:,:,:,0])

def test_model():
	start_time = time.time()

	make_directories(test_results_path, test_subs)

	result_files_path = os.path.join(test_results_path, 'result_files')
	Images_path = os.path.join(test_results_path, 'Images')
	inc_psnr_files_path = os.path.join(test_results_path, 'inc_psnr_files')

	mse_img = []
	psnr_img = []
	mae_img = []
	ssim_val = []
	test_subs_decoded_imgs = dict()

	test_data_path_mat, test_label_path_mat = make_valid_path_matrices(test_subs)
	test_avg_psnr_img = open(test_results_path+ '/test_avg_psnr_img' + '.txt', 'a')
	test_avg_mse_img = open(test_results_path+ '/test_avg_mse_img' + '.txt', 'a')

	for each_test_sub in test_subs:
		print(each_test_sub)
		test_sub_path = os.path.join(result_files_path, each_test_sub)
		test_psnr_img = open(test_sub_path+ '/test_psnr_img' + '.txt', 'a')
		test_mse_img = open(test_sub_path+ '/test_mse_img' + '.txt', 'a')
		test_mae_img = open(test_sub_path+ '/test_mae_img' + '.txt', 'a')
		test_ssim_file = open(test_sub_path + '/test_ssim.txt', 'a')

		test_data = np.zeros((len(test_data_path_mat[each_test_sub]), 3, 512, 512))
		test_label = np.zeros((len(test_label_path_mat[each_test_sub]), 1, 512, 512))
		decoded_imgs = np.zeros((len(test_data_path_mat[each_test_sub]), 512, 512))

		for k in range(len(test_label_path_mat[each_test_sub])):
		# for k in range(1):
			test_data[k] = min_max_normalization(np.load(test_data_path_mat[each_test_sub][k]))
			test_label[k] = min_max_normalization(np.load(test_label_path_mat[each_test_sub][k]))

			for i in range(8):
				for j in range(8):
					data = np.zeros((d, h, w))
					start_row = i * 64
					start_col = j * 64
					data[0,:,:] = test_data[k][0, start_row:start_row+h, start_col:start_col+w]
					data[1,:,:] = test_data[k][1, start_row:start_row+h, start_col:start_col+w]
					data[2,:,:] = test_data[k][2, start_row:start_row+h, start_col:start_col+w]

					data = data[np.newaxis, :, :, :, np.newaxis]
					pred_img = g_model(data)
					
					decoded_imgs[k][start_row:start_row+h, start_col:start_col+w] = pred_img[0,:,:,0]

		mse_img.append(math.sqrt(np.mean((test_label[:,0, :,:] - decoded_imgs[:,:,:]) ** 2)))
		psnr_img.append(20 * math.log10( 1.0 / (mse_img[-1])))
		mae_img.append(np.mean(np.abs((test_label[:,0, :,:] - decoded_imgs[:,:,:]))))
		ssim_val.append(ssim(decoded_imgs[:,:,:], test_label[:,0, :,:], multichannel=True))

		test_mse_img.write("%f \n" %(mse_img[-1]))
		test_psnr_img.write("%f \n" %(psnr_img[-1]))
		test_mae_img.write("%f \n" %(mae_img[-1]))
		test_ssim_file.write("SSIM for test_sub_num is %f \n" %(ssim_val[-1]))

		test_subs_decoded_imgs[each_test_sub] = decoded_imgs

		for slice_num in [0,40,60,80]:
			temp = np.zeros([op_img_size, op_img_size*4])
			temp[:op_img_size,:op_img_size] = test_data[slice_num,1,:,:]
			temp[:op_img_size,op_img_size:op_img_size*2] = test_label[slice_num,:,:]
			temp[:op_img_size,op_img_size*2:op_img_size*3] = decoded_imgs[slice_num,:,:]
			temp[:op_img_size,op_img_size*3:] = abs(decoded_imgs[slice_num,:,:] - test_label[slice_num,0,:,:])
			temp = temp * 255
			# each_valid_sub_num = name_numbers(3, valid_subs[i])
			path = os.path.join(os.path.join(Images_path, each_test_sub), str(slice_num))

			cv2.imwrite(path + '.jpg', temp)

	psnr_img = np.asarray(psnr_img)
	avg_psnr = np.mean(psnr_img)

	test_avg_psnr_img.write("%f \n" %(avg_psnr))
	avg_mse = np.mean(np.asarray(mse_img))
	test_avg_mse_img.write("%f \n" %(avg_mse))
		   
	for each_test_sub in test_subs:
	    # each_valid_sub_num = name_numbers(3, valid_subs[i])
	    path = os.path.join(inc_psnr_files_path, each_test_sub)
	    np.save(path + ".npy", test_subs_decoded_imgs[each_test_sub][:,:,:])

	test_psnr_img.close()
	test_mse_img.close()
	test_mae_img.close()

#--------------------------------------------------------------------------------------------------------
# Arguments:

SACNN_pro_low_dose_data_path = "/home/ada/Preethi/CT_Scan/Data/LDCT/SACNN_Processed_Data/Low_and_Full_Dose/Training_Data/Low_Dose"
SACNN_pro_full_dose_data_path = "/home/ada/Preethi/CT_Scan/Data/LDCT/SACNN_Processed_Data/Low_and_Full_Dose/Training_Data/Full_Dose"

SACNN_pro_valid_low_dose_data_path = "/home/ada/Preethi/CT_Scan/Data/LDCT/SACNN_Processed_Data/Low_and_Full_Dose/Test_Data/Low_Dose"
SACNN_pro_valid_full_dose_data_path = "/home/ada/Preethi/CT_Scan/Data/LDCT/SACNN_Processed_Data/Low_and_Full_Dose/Test_Data/Full_Dose"

# train_subs = list(set(os.listdir(pro_low_dose_data_path)) - set(valid_subs + test_subs))
train_subs = ['L160', 'L123', 'L145', 'L210', 'L134', 'L186']
valid_subs = ['L058', 'L131']
test_subs = ['L170','L004']

adam = tf.keras.optimizers.Adam(
    learning_rate=0.0001, beta_1=0.9, beta_2=0.999)

numEpochs = 1000
save_at_every = 2
# num_of_subs = 4
less_than_C_by = 0
op_img_size = 512
batch_size = 1
local_path = "/home/ada/Preethi/CT_Scan/Code/Supervised/SACNN/LDCT/Split_6_2_2/Part_5/Valid_Results"
try:
    os.mkdir(local_path)
except (FileExistsError):
   pass

d = 3 
h = 64
w = 64
c = 1
#--------------------------------------------------------------------------------------------------------

# log_file_path = local_path + "/log_file.txt"
# with open(log_file_path, 'a') as log_file:
#     log_file.write("-----------------------------------------\nProgram Starts")
#--------------------------------------------------------------------------------------------------------
# g_model, ae_on_g_model = make_model(d, h, w, c)
# train_model()
#--------------------------------------------------------------------------------------------------------
best_model_path = "/home/ada/Preethi/CT_Scan/Code/Supervised/SACNN/LDCT/Split_6_2_2/Part_5/Valid_Results/Model/Last_EpochNum.h5"
test_results_path = "/home/ada/Preethi/CT_Scan/Code/Supervised/SACNN/LDCT/Split_6_2_2/Part_5/Test_Results"
try:
    os.mkdir(test_results_path)
except (FileExistsError):
   pass
g_model, ae_on_g_model = make_model(d, h, w, c)
test_model()
# --------------------------------------------------------------------------------------------------------


