import cv2
import numpy as np
import threading
import time
import os
from vimba import *
from filterwheel import FW102C

# Inisialisasi Vimba untuk mengakses kamera
with Vimba.get_instance() as vimba:
    # Ambil semua kamera yang terhubung
    cams = vimba.get_all_cameras()
    # Gunakan kamera pertama
    with cams[0] as cam:
        # Inisialisasi capture video dari kamera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # Status dan variabel untuk pengendalian proses
        status = True
        status_capturing = False
        frameCount = 0
        FolderName = 'Dilham - 18 Juli 2022'
        
        # Kamus untuk label jenis plastik, organik, dan skenario
        plastik_dic = {'1': 'PET(PETE)', '2': 'HDPE', '3': 'PP', '4': ''}
        organik_dic = {'1': 'DAUN', '2': 'RANTING', '3': 'KERTAS', '4': 'KARDUS', '5': ''}
        skenario_dic = {'1': 'dengan-bungkus', '2': 'tanpa-bungkus', '3': ''}
        
        # Inisialisasi filterwheel
        fwl = FW102C(port='COM4')
        if not fwl.isOpen:
            print("FWL INIT FAILED")
            sys.exit(2)
        print('**info', fwl.getinfo())
        print('**idn?', fwl.query('*idn?'))

        # Fungsi untuk menangani setiap frame yang diakuisisi
        def handler(cam, frame):
            print('Frame acquired: {}'.format(frame), flush=True)
            cam.queue_frame(frame)

        # Fungsi utama untuk mengatur streaming dan akuisisi gambar
        def main():
            with Vimba.get_instance() as vimba:
                cam = vimba.get_all_cameras()[0]
                with cam:
                    cam.TriggerSource.set('Software')
                    cam.TriggerSelector.set('FrameStart')
                    cam.TriggerMode.set('On')
                    cam.AcquisitionMode.set('Continuous')
                    try:
                        cam.start_streaming(handler)
                        time.sleep(1)
                        cam.TriggerSoftware.run()
                        time.sleep(1)
                        cam.TriggerSoftware.run()
                        time.sleep(1)
                        cam.TriggerSoftware.run()
                    finally:
                        cam.stop_streaming()

        # Memulai proses streaming dan akuisisi gambar
        if __name__ == '__main__':
            main()

        # Loop utama untuk memproses gambar dari kamera
        while status:
            key_input = cv2.waitKey(1)

            if key_input == ord('q'):
                break

            # Set exposure time kamera
            exposure_time = cam.ExposureTime
            exposure_time.set(200000)
            output_image = cam.get_frame()
            output_image.convert_pixel_format(PixelFormat.Mono8)
            output_image = output_image.as_opencv_image()
            output_image = cv2.resize(output_image, (648, 486))
            cv2.imshow(f"image", output_image)

            # Deteksi gerakan sederhana menggunakan input keyboard 's'
            if key_input == ord('s'):
                responses['moved'] = '1'
            else:
                responses['moved'] = '0'

            # Proses penyimpanan gambar berdasarkan status_capturing dan frameCount
            if frameCount < 11 and status_capturing == True:
                frame = cam.get_frame()
                frame.convert_pixel_format(PixelFormat.Mono8)
                cv2.imwrite(f'{FolderName}/{sampel}-f{frameCount}_({jenis_plastik},{jenis_organik})_{skenario}.png', frame.as_opencv_image())
                frameCount += 1

                # Logika untuk menentukan jenis sampel plastik, organik, dan skenario
                if responses['moved'] == '1':
                    status_capturing = True
                    sampel = input('sampel ke: ')
                    print(f'jenis sampel plastik = {plastik_dic}')
                    jenis_plast = input('masukkan jenis plastik: ')
                    print(f'jenis sampel organik = {organik_dic}')
                    jenis_orga = input('masukkan jenis organik: ')
                    print(f'skenario = {skenario_dic}')
                    skenario_acq = input('masukkan skenario: ')

                    jenis_plastik = plastik_dic.get(str(jenis_plast), '')
                    jenis_organik = organik_dic.get(str(jenis_orga), '')
                    skenario = skenario_dic.get(str(skenario_acq), '')

                    if not os.path.exists(FolderName):
                        os.makedirs(FolderName)
                    status_capturing = True

        # Selesai akuisisi gambar, tutup proses
        st_device.acquisition_stop()
        st_datastream.stop_acquisition()
