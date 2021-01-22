from UNET_model import get_unet
from keras.layers import Input
from keras.models import Model, Sequential
from contextlib import redirect_stdout
from keras.optimizers import RMSprop
import os
import numpy as np 
from sklearn.utils import shuffle
import math
from skimage.measure import compare_ssim as ssim
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
    model = get_unet(input_tensor)

    model.compile(loss= "mean_squared_error", optimizer = RMSprop())

    with open('unet.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

    return model

def make_path_matrices(root_folder): 
    img_mat = []
    all_subs = sorted(os.listdir(root_folder))
    for each_sub in all_subs:
        each_sub_imgs = []
        all_slices_path = os.path.join(root_folder, each_sub)
        all_slices = sorted(os.listdir(all_slices_path))
        for each_slice in all_slices:
            each_slice_path = os.path.join(all_slices_path, each_slice)
            each_sub_imgs.append(each_slice_path)

        each_sub_imgs = np.asarray(each_sub_imgs) #is it necessary to make it numpy array: easy to find shape
        img_mat.append(each_sub_imgs)
        # print(each_sub_imgs.shape)

    img_mat = np.asarray(img_mat) #is it necessary to make it numpy array

    return img_mat

def make_valid_path_matrices():
    valid_data_path_mat = []
    valid_label_path_mat = []
    
    for each_valid_sub in valid_subs:
        valid_data_path_mat.append(KVP_70_img_path_mat[each_valid_sub])
        valid_label_path_mat.append(KVP_100_img_path_mat[each_valid_sub])

    valid_data_path_mat = np.asarray(valid_data_path_mat)
    valid_label_path_mat = np.asarray(valid_label_path_mat)

    return valid_data_path_mat, valid_label_path_mat

def make_train_path_matrices():
    train_data_path_mat = []
    train_label_path_mat = []

    random.shuffle(train_subs)
    for each_sub in train_subs:     
        train_data_path_mat.extend(KVP_70_img_path_mat[each_sub])
        train_label_path_mat.extend(KVP_100_img_path_mat[each_sub])

    train_data_path_mat = np.asarray(train_data_path_mat)
    train_label_path_mat = np.asarray(train_label_path_mat)

    return train_data_path_mat, train_label_path_mat

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
                each_valid_sub_num = name_numbers(3, each_valid_sub)
                valid_sub_path = os.path.join(file_path, each_valid_sub_num)
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
    valid_data_path_mat, valid_label_path_mat = make_valid_path_matrices()
    
    make_directories(local_path, valid_subs)

    result_files_path = os.path.join(local_path, 'result_files')
    Model_path = os.path.join(local_path, 'Model')
    Images_path = os.path.join(local_path, 'Images')
    inc_psnr_files_path = os.path.join(local_path, 'inc_psnr_files')

    max_psnr = 0

    for jj in range(numEpochs):
        print('time in secs', time.time() - start_time)
        start_time = time.time()
        # with open(log_file_path, 'a') as log_file:
        #     log_file.write("Running epoch : %d" % jj)

        # Shuffling patient wise each epoch.
        train_data_path_mat, train_label_path_mat = make_train_path_matrices()

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
                sub_num = find_min_max_of_img(batch + (batch * batch_size))
                min_data_val = min_max_of_subs_KVP_70[sub_num][0]
                max_data_val = min_max_of_subs_KVP_70[sub_num][1]
                min_label_val = min_max_of_subs_KVP_100[sub_num][0]
                max_label_val = min_max_of_subs_KVP_100[sub_num][1]

                batch_train_data[element_in_batch, :, :, 0] = min_max_normalization(np.load(train_data_path_mat[each_npy]), min_data_val, max_data_val)
                batch_train_label[element_in_batch, :, :, 0] = min_max_normalization(np.load(train_label_path_mat[each_npy]), min_label_val, max_label_val)

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
        valid_subs_decoded_imgs = []

        for i in range(len(valid_subs)):
            each_valid_sub_num = name_numbers(3, valid_subs[i])
            valid_sub_path = os.path.join(result_files_path, each_valid_sub_num)
            valid_psnr_img = open(valid_sub_path+ '/valid_psnr_img' + '.txt', 'a')
            valid_mse_img = open(valid_sub_path+ '/valid_mse_img' + '.txt', 'a')
            valid_mae_img = open(valid_sub_path+ '/valid_mae_img' + '.txt', 'a')
            ssim_file = open(valid_sub_path + '/ssim.txt', 'a')

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
            valid_subs_decoded_imgs.append(decoded_imgs)

            mse_img.append(math.sqrt(np.mean((valid_label[:,:,:,0] - decoded_imgs[:,:,:,0]) ** 2)))
            psnr_img.append(20 * math.log10( 1.0 / (mse_img[-1])))
            mae_img.append(np.mean(np.abs((valid_label[:,:,:,0] - decoded_imgs[:,:,:,0]))))
            ssim_val.append(ssim(decoded_imgs[:,:,:,0], valid_label[:,:,:,0], multichannel=True))

            valid_mse_img.write("%f \n" %(mse_img[-1]))
            valid_psnr_img.write("%f \n" %(psnr_img[-1]))
            valid_mae_img.write("%f \n" %(mae_img[-1]))
            ssim_file.write("SSIM for test_sub_num is %f \n" %(ssim_val[-1]))

            if (jj % save_at_every == 0) or (jj == (numEpochs - 1)):
                for slice_num in [20,40,60,80,100,120,140,160,180]:
                    temp = np.zeros([img_size, img_size*4])
                    temp[:img_size,:img_size] = valid_data[slice_num,:,:,0]
                    temp[:img_size,img_size:img_size*2] = valid_label[slice_num,:,:,0]
                    temp[:img_size,img_size*2:img_size*3] = decoded_imgs[slice_num,:,:,0]
                    temp[:img_size,img_size*3:] = abs(decoded_imgs[slice_num,:,:,0] - valid_label[slice_num,:,:,0])
                    temp = temp * 255
                    each_valid_sub_num = name_numbers(3, valid_subs[i])
                    path = os.path.join(os.path.join(Images_path, each_valid_sub_num), str(slice_num))
                    try:
                        os.mkdir(path)
                    except OSError:
                        pass

                    cv2.imwrite(path + "/EpochNum"+ str(jj) + "_Slice_num" + str(slice_num) + '.jpg', temp)


        psnr_img = np.asarray(psnr_img)
        avg_psnr = np.mean(psnr_img)
        if avg_psnr > max_psnr:
            max_psnr = avg_psnr
            model.save_weights(Model_path + "/Last_EpochNum.h5")
            
            for i in range(len(valid_subs)):
                each_valid_sub_num = name_numbers(3, valid_subs[i])
                path = os.path.join(inc_psnr_files_path, each_valid_sub_num)
                np.save(path + "/EpochNum"+ str(jj) + ".npy", valid_subs_decoded_imgs[i][:,:,:,0])


    valid_psnr_img.close()
    valid_mse_img.close()
    valid_mae_img.close()

    batch_loss_file.close()
    batch_loss_per_epoch_file.close()

def min_max_normalization(img, img_min, img_max):
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

#--------------------------------------------------------------------------------------------------------
# Arguments:
pro_data_path = "/home/ada/Preethi/PGI_CT_Scan/Paired_Data/Data/processed_data/un_norm"
numEpochs = 300
save_at_every = 20
num_of_subs = 8
valid_subs = [2, 3]
batch_size = 30
local_path = "/home/ada/Preethi/PGI_CT_Scan/Paired_Data/Code/unet_7_1_21"
#--------------------------------------------------------------------------------------------------------
# log_file_path = local_path + "/log_file.txt"
# with open(log_file_path, 'a') as log_file:
#     log_file.write("-----------------------------------------\nProgram Starts")
#--------------------------------------------------------------------------------------------------------
KVP_70 = pro_data_path + "/KVP_70"
KVP_70_img_path_mat = make_path_matrices(KVP_70)

KVP_100 = pro_data_path + "/KVP_100"
KVP_100_img_path_mat = make_path_matrices(KVP_100)

min_max_of_subs_KVP_70, min_max_of_subs_KVP_100, num_of_slices_per_sub = find_min_max_of_subs()
#--------------------------------------------------------------------------------------------------------
train_subs = list(set(range(8))-set(valid_subs))
print(train_subs)
model = make_model()
train_model(model, img_size = 512)

