import os

# filenames_path = "/home/ada/Preethi/PGI_CT_Scan/Paired_Data/Data/processed_data/un_norm/KVP_70/002"
path = "/home/ada/Preethi/PGI_CT_Scan/Paired_Data/Data/processed_data/un_norm/KVP_100/002_double"
# dest_path = "/home/ada/Preethi/PGI_CT_Scan/Paired_Data/Data/processed_data/un_norm/KVP_100/002"

filenames = os.listdir(path)
for each in filenames:
    a = each.split('.')
    if len(a) == 1:
        remove_path = os.path.join(path, each)
        os.rmdir(remove_path)
    


