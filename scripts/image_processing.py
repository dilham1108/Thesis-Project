import cv2
import numpy as np
import pandas as pd
import glob
import os

# Direktori file input
file_dir = 'dataset/folderName'
save_dir = 'dataset/folderSave'

# Pastikan direktori output tersedia
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ROI gambar yang ingin diproses
img_roi = [154, 82, 307, 163]

# Inisialisasi DataFrame kosong
dataframe = pd.DataFrame()

# Loop untuk setiap sampel (s) dari 1 hingga 999
for s in range(1, 1000):
    intensity = {
        'sampel': [],
        'skenario': [],
        'LED2': [], 'LED3': [], 'LED4': [], 'LED5': [], 'LED6': [],
        'LED7': [], 'LED8': [], 'LED9': []
    }
    
    # Mencari file sampel yang sesuai
    samples = glob.glob(f'{file_dir}/*{s}_*')

    for sample in samples:
        print(sample)
    
    depan = []
    belakang = []

    # Memisahkan sampel depan dan belakang berdasarkan skenario
    for i in samples:
        skenario = i.split('_')[3].split('.')[0]
        if skenario == 'D':
            depan.append(i)
        elif skenario == 'B':
            belakang.append(i)
    
    # Proses gambar untuk sampel depan
    for i in depan:
        intensity['sampel'] = [s]
        skenario = i.split('_')[3].split('.')[0]
        intensity['skenario'] = [skenario]
        nfilter = i.split('_')[2]
        
        citra = cv2.imread(i)
        res = citra[img_roi[1]:img_roi[1]+img_roi[3], img_roi[0]:img_roi[0] + img_roi[2]]
        mean = np.mean(res)
        intensity[nfilter] = mean
    
    df = pd.DataFrame(intensity, index=[1])
    dataframe = dataframe.append(df)
    depan = []
    print(f'intensity df >>>>>>>>> \n {df}')
    print(f'dataframe >>>>>>>>> \n {dataframe}')
    
    # Proses gambar untuk sampel belakang
    for i in belakang:
        intensity['sampel'] = [s]
        skenario = i.split('_')[3].split('.')[0]
        intensity['skenario'] = [skenario]
        nfilter = i.split('_')[2]
        
        citra = cv2.imread(i)
        res = citra[img_roi[1]:img_roi[1]+img_roi[3], img_roi[0]:img_roi[0] + img_roi[2]]
        mean = np.mean(res)
        intensity[nfilter] = mean
    
    df = pd.DataFrame(intensity, index=[1])
    dataframe = dataframe.append(df)
    belakang = []
    print(f'intensity df >>>>>>>>> \n {df}')
    print(f'dataframe >>>>>>>>> \n {dataframe}')

# Simpan DataFrame ke dalam file Excel jika diperlukan
dataframe.to_excel('Intensity-PP.xlsx')
