from google_images_download import google_images_download
import os
from fastai.plots import *
import numpy as np
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
import cv2
from PIL import Image
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
        except Exception :
            pass
    else:
        os.mkdir(os.getcwd() + '/Data_files/' + final_for_file_name)


    for i in os.listdir(os.getcwd()+'/downloads'):
        try:
            os.rename(os.getcwd()+'/downloads/'+i,os.getcwd() + '/Data_files/' + final_for_file_name+'/'+i)

        except Exception as e:
            print(e)

        try:
            if os.path.exists(os.getcwd() + '/Data_files/' + final_for_file_name + '/' + 'train'):
                pass
            else:
                os.mkdir(os.getcwd() + '/Data_files/' + final_for_file_name + '/' + 'train')

            if os.path.exists(os.getcwd() + '/Data_files/' + final_for_file_name + '/' + 'valid'):
                pass
            else:
                os.mkdir(os.getcwd() + '/Data_files/' + final_for_file_name + '/' + 'valid')
        except Exception as e:
            print(e)

    try:
        if os.path.exists(os.getcwd() + '/temp'):
            pass
        else:
            os.mkdir(os.getcwd()+'/temp')

        os.rename(os.getcwd()+'/downloads',os.getcwd()+'/temp')
        os.remove(os.getcwd()+'/downloads')

    except Exception :
        pass

    for k in os.listdir(os.getcwd() + '/Data_files/' + final_for_file_name):
        main_lib = os.listdir(os.getcwd() + '/Data_files/' + final_for_file_name + '/' + k)
        amount_files = len(main_lib)
        int_partition_calculator = int(amount_files*25/100)
        counter = 0
        if k == 'valid' or k == 'train':
            global back_path
            back_path = os.getcwd()
            os.chdir(os.getcwd() + '/Data_files/' + final_for_file_name + '/' + k + '/')
            pass

        else:
            if os.path.exists(os.getcwd() + '/Data_files/' + final_for_file_name + '/' + 'valid' + '/' + k):
                inital = input('it seems this file already exists would you like to contnue (y/n)')
                if inital == 'n' or inital == 'no':
                    break
                else:
                    pass
            else:
                os.mkdir(os.getcwd() + '/Data_files/' + final_for_file_name + '/' + 'valid' + '/' + k)
                os.mkdir(os.getcwd() + '/Data_files/' + final_for_file_name + '/' + 'train' + '/' + k)

            back_path = os.getcwd()
            os.chdir(os.getcwd() + '/Data_files/' + final_for_file_name + '/' + k + '/')

            for j in os.listdir(back_path + '/Data_files/' + final_for_file_name + '/' + k):
                if counter < int_partition_calculator:
                    print(j)
                    print(os.getcwd() + '/Data_files/' + final_for_file_name + '/' + k + '/' + j)
                    im = Image.open(back_path+ '/Data_files/' + final_for_file_name + '/' + k + '/' + j)
                    width, height = im.size
                    img = cv2.imread(back_path+ '/Data_files/' + final_for_file_name + '/' + k + '/' + j)
                    if width <650 and height < 650:
                        resized_v = cv2.resize(img,(256,256))
                    elif width > 1000 and height < 1000:
                        resized_v = cv2.resize(cv2.resize(img,(0,0),fx = 0.6 ,fy=1), (256, 256))
                    elif width < 1000 and height > 1000:
                        resized_v = cv2.resize(cv2.resize(img, (0,0), fx=1, fy=0.6), (256, 256))
                    elif width > 1000 and height > 1000:
                        resized_v = cv2.resize(cv2.resize(img, (0,0), fx=0.6, fy=0.6), (256, 256))
                    else:
                        resized_v = cv2.resize(img, (256, 256))

                    converted = cv2.cvtColor(resized_v,cv2.COLOR_BGR2RGB)
                    cv2.imwrite(j,converted)
                    os.rename(back_path + '/Data_files/' + final_for_file_name + '/' + k + '/' + j,back_path + '/Data_files/' + final_for_file_name + '/' + 'valid' + '/' + k + '/' + j)
                    counter += 1
                else:
                    try:
                        print(j)
                        print(back_path + '/Data_files/' + final_for_file_name + '/' + k + '/' + j)
                        img = cv2.imread(os.getcwd() + '/Data_files/' + final_for_file_name + '/' + k + '/' + j)
                        resized_v = cv2.resize(img, (256, 256))
                        converted = cv2.cvtColor(resized_v, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(j, converted)
                        os.rename(os.getcwd() + '/Data_files/' + final_for_file_name + '/' + k + '/' + j,os.getcwd() + '/Data_files/' + final_for_file_name + '/' + 'train' + '/' + k + '/' + j)
                    except Exception as e :
                        print(e)

    os.chdir(back_path)
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

    try:
        os.chdir(PATH)
        print(os.getcwd())
        global arch
        arch = resnet34
        tfms = tfms_from_model(sz=sz,f_model= arch,aug_tfms=transforms_side_on,max_zoom=1.1)
        global data
        data = ImageClassifierData.from_paths(PATH,tfms=tfms)
        global learn
        learn = ConvLearner.pretrained(arch,data,precompute=True)
        learn.precompute = False
        learn.unfreeze()
        lrf = np.array([1e-4,1e-3,1e-2])
        learn.fit(lrf,1,cycle_len=1,cycle_mult=2)
        learn.save('all')
        learn.load('all')
        print('Yay !! you have made your Classifier !')
        g = input('please enter the place where your pic is stored : ')
        learn.load('all')
        trn_tfms, val_tfms = tfms_from_model(arch, sz)
        im = val_tfms(open_image(g))
        learn.precompute = False
        preds = learn.predict_array(im[None])
        print(data.classes[np.argmax(preds)])

    except Exception as e :
        print(e)

    while 1:
        input_checker = input('would you like to continue (yes/no) : ')
        if 'no' in input_checker or 'n' in input_checker:
            break
        else:
            g = input('please enter the place where your pic is stored')
            learn.load('all')
            trn_tfms, val_tfms = tfms_from_model(arch, sz)
            im = val_tfms(open_image(g))
            learn.precompute = False
            preds = learn.predict_array(im[None])
            print(data.classes[np.argmax(preds)])

make_files_for_me()
get_images()

