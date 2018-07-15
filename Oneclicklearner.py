from google_images_download import google_images_download
import os
from fastai.plots import *
import numpy as np
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *

sz = 224

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
    if os.path.exists(os.getcwd()+'/Data_files'):
        pass
    else:
        os.mkdir(os.getcwd()+'/Data_files')
    if os.path.exists(os.getcwd()+'/Data_files/'+final_for_file_name):
        try:
            os.remove(os.getcwd()+'/Data_files/'+final_for_file_name)
        except PermissionError :
            pass
    else:
        os.mkdir(os.getcwd() + '/Data_files/' + final_for_file_name)


    for i in os.listdir(os.getcwd()+'/downloads'):
        try:
            os.rename(os.getcwd()+'/downloads/'+i,os.getcwd() + '/Data_files/' + final_for_file_name+'/'+i)

        except Exception as e:
            print(e)

        try:
            if os.path.exists(os.getcwd() + '/Data_files/' + final_for_file_name + '/' + i + '/' + 'train'):
                pass
            else:
                os.mkdir(os.getcwd() + '/Data_files/' + final_for_file_name + '/' + i + '/' + 'train')

            if os.path.exists(os.getcwd() + '/Data_files/' + final_for_file_name + '/' + i + '/' + 'valid'):
                pass
            else:
                os.mkdir(os.getcwd() + '/Data_files/' + final_for_file_name + '/' + i + '/' + 'valid')
        except Exception as e:
            print(e)

    try:
        os.remove(os.getcwd()+'/downloads')
    except PermissionError :
        pass

    for k in os.listdir(os.getcwd() + '/Data_files/' + final_for_file_name):
        main_lib = os.listdir(os.getcwd() + '/Data_files/' + final_for_file_name + '/' + k)
        amount_files = len(main_lib)
        int_partition_calculator = int(amount_files*25/100)
        counter = 0
        for j in os.listdir(os.getcwd() + '/Data_files/' + final_for_file_name + '/' + k):
            if j == 'valid' or j == 'train':
                pass
            else:
                if counter < int_partition_calculator:
                    try:
                        os.rename(os.getcwd() + '/Data_files/' + final_for_file_name + '/' + k + '/' + j,os.getcwd() + '/Data_files/' + final_for_file_name + '/' + k + '/' + 'valid' + '/' + j)
                        counter += 1
                    except Exception as e:
                        print(e)
                else:
                    try:
                        os.rename(os.getcwd() + '/Data_files/' + final_for_file_name + '/' + k + '/' + j,os.getcwd() + '/Data_files/' + final_for_file_name + '/' + k + '/' + 'train' + '/' + j)
                    except Exception as e :
                        print(e)


    global PATH
    PATH = os.getcwd() + '/Data_files/' + final_for_file_name
def make_files_for_me():

    primary_file_path = os.getcwd() + '/Oneshotclassifier'
    secondary_images_path = primary_file_path+'/images_downloaded/'
    if os.path.exists(primary_file_path):
        pass
    else:
        os.mkdir(primary_file_path)
    os.chdir(primary_file_path)


def train():
    os.chdir(PATH)
    arch = resnet34
    tfms = tfms_from_model(sz=sz,f_model= arch,aug_tfms=transforms_side_on,max_zoom=1.1)
    data = ImageClassifierData.from_paths(PATH,tfms=tfms)
    learn = ConvLearner.pretrained(arch,data,precompute=True)
    learn.fit(0.01,2)
    learn.save('elementary')
    learn.load('elementary')
    learn.precompute = False
    learn.fit(0.01,1,cycle_len=1)
    learn.save('lastlayer')
    learn.load('lastlayer')
    learn.unfreeze()
    lrf = np.array([1e-4,1e-3,1e-2])
    learn.fit(lrf,1,cycle_len=1,cycle_mult=2)
    learn.save('all')
    learn.load('all')
    print('Yay !! you have made your Classifier !')
    g = input('please enter the place where your pic is stored')
    learn.load('all')
    trn_tfms, val_tfms = tfms_from_model(arch, sz)
    im = val_tfms(open_image(g))
    learn.precompute = False
    preds = learn.predict_array(im[None])
    print(preds)

    while 1:
        input_checker = input('would you like to continue (yes/no)')
        if 'no' or 'n' in input_checker:
            break
        else:
            g = input('please enter the place where your pic is stored')
            learn.load('all')
            trn_tfms, val_tfms = tfms_from_model(arch, sz)
            im = val_tfms(open_image(g))
            learn.precompute = False
            preds = learn.predict_array(im[None])
            print(preds)


make_files_for_me()
get_images()
train()
