from google_images_download import google_images_download
import os

def get_images():
    response = google_images_download.googleimagesdownload()
    global take_input ,number_of_data_images
    take_input = input('please enter the name of the items you wish the classify with commas separating each : ')
    number_of_data_images = input('please enter the number of images you need to train the model on (less number = less time = less efficiency) : ')
    arguments = {"keywords": take_input, 'limit': int(number_of_data_images), "print_urls": True}
    paths = response.download(arguments)
    print(paths)
    splitted = take_input.split(',')
    final_for_file_name = '_'.join(splitted)
    if os.path.exists(os.getcwd()+'\\Data_files'):
        pass
    else:
        os.mkdir(os.getcwd()+'\\Data_files')
    if os.path.exists(os.getcwd()+'\\Data_files\\'+final_for_file_name):
        try:
            os.remove(os.getcwd()+'\\Data_files\\'+final_for_file_name)
        except PermissionError :
            pass
    else:
        os.mkdir(os.getcwd() + '\\Data_files\\' + final_for_file_name)


    for i in os.listdir(os.getcwd()+'\\downloads'):
        try:
            os.rename(os.getcwd()+'\\downloads\\'+i,os.getcwd() + '\\Data_files\\' + final_for_file_name+'\\'+i)

        except Exception as e:
            print(e)

        try:
            if os.path.exists(os.getcwd() + '\\Data_files\\' + final_for_file_name + '\\' + i + '\\' + 'train'):
                pass
            else:
                os.mkdir(os.getcwd() + '\\Data_files\\' + final_for_file_name + '\\' + i + '\\' + 'train')

            if os.path.exists(os.getcwd() + '\\Data_files\\' + final_for_file_name + '\\' + i + '\\' + 'test'):
                pass
            else:
                os.mkdir(os.getcwd() + '\\Data_files\\' + final_for_file_name + '\\' + i + '\\' + 'test')
        except Exception as e:
            print(e)

    try:
        os.remove(os.getcwd()+'\\downloads')
    except PermissionError :
        pass

    for k in os.listdir(os.getcwd() + '\\Data_files\\' + final_for_file_name):
        main_lib = os.listdir(os.getcwd() + '\\Data_files\\' + final_for_file_name + '\\' + k)
        amount_files = len(main_lib)
        int_partition_calculator = int(amount_files*25/100)
        counter = 0
        for j in os.listdir(os.getcwd() + '\\Data_files\\' + final_for_file_name + '\\' + k):
            if j == 'test' or j == 'train':
                pass
            else:
                if counter < int_partition_calculator:
                    try:
                        os.rename(os.getcwd() + '\\Data_files\\' + final_for_file_name + '\\' + k + '\\' + j,os.getcwd() + '\\Data_files\\' + final_for_file_name + '\\' + k + '\\' + 'test' + '\\' + j)
                        counter += 1
                    except Exception as e:
                        print(e)
                else:
                    try:
                        os.rename(os.getcwd() + '\\Data_files\\' + final_for_file_name + '\\' + k + '\\' + j,os.getcwd() + '\\Data_files\\' + final_for_file_name + '\\' + k + '\\' + 'train' + '\\' + j)
                    except Exception as e :
                        print(e)



def make_files_for_me():

    primary_file_path = os.getcwd() + '\\Oneshotclassifier'
    secondary_images_path = primary_file_path+'\\images_downloaded\\'
    if os.path.exists(primary_file_path):
        pass
    else:
        os.mkdir(primary_file_path)
    os.chdir(primary_file_path)

make_files_for_me()
get_images()




