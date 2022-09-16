# -*- coding: utf-8 -*-
import cv2
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import csv
from collections import Counter
from pathlib import Path

os.chdir(r"C:\Users\JOHNYA\Desktop\python_files\dictionary_values")


def normalization(image, image_black, image_white):
    '''
    Apply normalization with white, black videos

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    image_black : TYPE
        DESCRIPTION.
    image_white : TYPE
        DESCRIPTION.

    Returns
    -------
    image_normed : TYPE
        DESCRIPTION.

    '''
    nominator = cv2.subtract(image, image_black)
    denominator = cv2.subtract(image_white, image_black)
    image_normed = cv2.divide(nominator, denominator, scale=255)

    return image_normed


class CircleDetection():

    def __init__(self, image_source, white_source, black_source):

        self.list_folder_messtag = image_source
        self.list_folder_messtag_weiss = white_source
        self.list_folder_messtag_schwarz = black_source
        self.dx1 = 20
        self.dy1 = 45
        self.dx2 = 25
        self.dy2 = 55
        self.minDist = 500
        self.param1 = 300
        self.param2 = 3
        self.minRadius = 60
        self.maxRadius = 80
        self.compute_radius = 20
    
    def find_circle_otsu(self, image_normed):

        # **************************************************
        # Applying thresholds
        # th1 - binary threshold with otsu, th2 inverse binary with otsu
        # binarize the image
        # temp = cv2.bilateralFilter(image_normed, 21, 21, 21)
        temp = cv2.GaussianBlur(image_normed,(7,7),1.4)
        # **************************************************
        kernel = np.ones((2, 2), dtype=np.uint8)
        erode = cv2.erode(temp, kernel, iterations=1)

        th1 = cv2.threshold(
            erode, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        th2 = cv2.threshold(
            erode, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

        closingSize = 4
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2*closingSize + 1, 2*closingSize + 1), (closingSize, closingSize))
        th1 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel, iterations=8)
        th1 = cv2.Canny(th1, 200, 250)

        # **************************************************
        #  Applýing connected components
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
            th1, connectivity=8)

        # Find the largest non background component.
        # Note: range() starts from 1 since 0 is the background label.
        max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA])
                                  for i in range(0, nb_components)], key=lambda x: x[0])

        # **************************************************

        all_area = [(stats[i, cv2.CC_STAT_AREA])
                    for i in range(0, nb_components)]

        max_size = all_area[0]

        # **************************************************
        # Finding and fitting circle
        #####Find Light Radius#######

        area = self.minRadius**2 * np.pi

        if max_size > area:

            #########Fit Circle##########
            circlesLight = cv2.HoughCircles(th1, cv2.HOUGH_GRADIENT, 1, self.minDist, param1=self.param1,
                                            param2=self.param2, minRadius=self.minRadius, maxRadius=self.maxRadius)

            th1 = cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR)
            th2 = cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR)
            image_normed = cv2.cvtColor(image_normed, cv2.COLOR_GRAY2BGR)

            if circlesLight is not None:

                circles = np.uint16(np.around(circlesLight))

                for i in circles[0, :]:
                    # draw the outer circle
                    # green color = (0, 255, 0)
                    cv2.circle(th1, (i[0], i[1]), i[2],
                               color=(0, 255, 0), thickness=2)
                    # draw the center of the circle
                    cv2.circle(th1, (i[0], i[1]), self.compute_radius, color=(
                        0, 255, 0), thickness=2)

                circle_feature = (i[0], i[1], i[2])
                mask = np.zeros_like(image_normed[:,:,2])
                #  white color =(255,255,255)
                mask = cv2.circle(mask, ((i[0] + th1.shape[1]//2)//2, (i[1] + th1.shape[0]//2)//2),
                                  self.compute_radius, color=(255, 255, 255), thickness=-1)

            else:
                # red color = (0,0,255)
                cv2.circle(th1, (th1.shape[0]//2, th1.shape[1]//2),
                           self.compute_radius, color=(0, 0, 255), thickness=2)
                circle_feature = th1.shape[0]//2, th1.shape[1]//2, 0.0
                mask = None
        else:
            th1 = cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR)
            # th2 = cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR)
            image_normed = cv2.cvtColor(image_normed, cv2.COLOR_GRAY2BGR)

            circle_feature = th1.shape[0]//2, th1.shape[1]//2, 0.0
            # yellow color = (0,255,255)
            cv2.circle(th1, (th2.shape[0]//2, th1.shape[1]//2),
                       self.compute_radius, color=(0, 255, 255), thickness=2)
            mask = None
            
        cv2.imshow('Image, Thresholded image',
                   np.hstack((image_normed, th1)))
        # cv2.imwrite(r'C:\Users\JOHNYA\Desktop\python_files\results\final\Image_Thresholdedimage.png', np.hstack(
        #     (image_normed, th1)))
        cv2.waitKey(1)

        signal = cv2.mean(image_normed[:,:,2], mask = mask)[0]

        return circle_feature, self.minRadius, self.maxRadius, signal


    def convert_video_to_images(self):
        '''
        Function to convert videos(.avi) files to images, plot signals, save as dictionary

        Returns
        -------
        dict_values : dict
            Dictionary with Device number, Filename, time, X coordinate, Y coordinate, 
            R coordinate, Minimum Radius, Maximum Radius

        '''
        # list_folder_messtag = self.list_folder_messtag
        for path_avi in self.list_folder_messtag:
            
            device_no = path_avi.split('\\')[-2].split('_')[-1]
            date = path_avi.split('_')[-5]
            replicate = path_avi.split('_')[-2][0]
            filename = path_avi.split('\\')[-1]
            x = []
            y = []
            r = []
            s = []
            # Read image
            cap = cv2.VideoCapture(path_avi)
            
            # starting with first image
            success, image = cap.read()
            # starting with second image
            success, image = cap.read()
            success, image = cap.read()
            for white in self.list_folder_messtag_weiss:
                cap_white = cv2.VideoCapture(path_avi)
                success, image_white = cap_white.read()

            for black in self.list_folder_messtag_schwarz:
                cap_black = cv2.VideoCapture(black)
                success, image_black = cap_black.read()

            count = 0

            height = image.shape[1]
            width = image.shape[0]

            while success:
                image_normed = normalization(
                    image[:, :, 2], image_black[:, :, 2], image_white[:, :, 2])
                image_normed = image_normed[self.dy1:height -
                                            self.dx1, self.dy2:width-self.dx2]
                image = image[self.dy1:height-self.dx1,
                              self.dy2:width-self.dx2, 2]
                
                circle_features, min_radius, max_radius, s_temp = self.find_circle_otsu(image_normed)
                x_temp, y_temp, r_temp = circle_features
                x.append(x_temp)
                y.append(y_temp)
                r.append(r_temp)
                s.append(s_temp)

                success, image = cap.read()
                count += 1
                cv2.waitKey(1)
            
            signal = scipy.signal.savgol_filter(s, 10, 1, deriv=0, delta=1.0, axis = - 1, mode='nearest', cval=0.0)
            slope = scipy.signal.savgol_filter(s, 10, 1, deriv=1, delta=1.0, axis = - 1, mode='nearest', cval=0.0)
            
            # Frames per second. We are taking frames from 2nd frame.
            t = [i for i in range(0, len(r))]

            dict_values = {
                'Device': [path_avi.split('\\')[-2].split('_')[-1]],
                'Date': [path_avi.split('_')[-5]],
                'Filename': [path_avi.split('\\')[-1]],
                'Replicate': [path_avi.split('_')[-2][0]],
                'Time': t,
                'X coordinate': x,
                'Y coordinate': y,
                'R coordinate': r,
                'Signal' : signal,
                'Slope' : slope,
                'Minimum Radius': min_radius,
                'Maximum Radius': max_radius
            }

            dict_filename = str('Date_' + date + '_Device_' +
                                device_no + '_Replicate_' + replicate + '_6')
            
            # ***************************************************************
            csv_file = r'C:\Users\JOHNYA\Desktop\python_files\dictionary_values\{}.csv'.format(
                dict_filename)
            # w = csv.writer(open(csv_file, "w"))
            # loop over dictionary keys and values
            # for key, val in dict_values.items():
            #     # write every key and value to file
            #     w.writerow([key, val])
            
            # ***************************************************************
            xlsx_file = r'C:\Users\JOHNYA\Desktop\python_files\dictionary_values\{}.xlsx'.format(
                dict_filename)
           
            # Creating dataframe to store dictionary values
            df = pd.DataFrame(dict_values, columns = ['Time','X coordinate', 'Y coordinate', 'R coordinate', 'Signal', 'Slope'])
            df['Filename'] = str('Date_' + date + '_Device_' +
                                device_no + '_Replicate_' + replicate + '_6')
            df = df.set_index('Filename', append=True).unstack('Filename')

            # creating excel writer object
            writer = pd.ExcelWriter(xlsx_file)
            df.to_excel(writer)
            # save the excel
            writer.save()
            # ***************************************************************
            cap.release()
            cv2.destroyAllWindows()
            
        return dict_values
    
    
    def plots(self):
        
        path = r'C:\Users\JOHNYA\Desktop\python_files\dictionary_values'
        csv_files = glob.glob(os.path.join(path, "*_6.xlsx"))
          
        # loop over the list of csv files
        for f in csv_files:

            date = [f.split('\\')[-1].split('_')[1]]
            device_no = [f.split('\\')[-1].split('_')[3]]
            replicate = [f.split('\\')[-1].split('_')[5]]

            # read the csv file
            all_data = pd.read_excel(f)
            df = pd.DataFrame(all_data)
            
            signal = df['Signal'].tolist()
            slope = df['Slope'].tolist()
            t = df['Time'].tolist()
            x = df['X coordinate'].tolist()
            y = df['Y coordinate'].tolist()
            r = df['R coordinate'].tolist()
           
            plt.subplots(figsize=(15, 7))
            plt.suptitle('Date {}, Device_number {}, Replicate {}_6'.format(
                date, device_no, replicate))
            
            plt.subplot(121)
            plt.plot(t[1:],signal[1:])
            plt.xlabel('Frames')
            plt.ylabel('Signal')
            
            plt.subplot(122)
            plt.plot(t[1:],slope[1:])
            plt.xlabel('Frames')
            plt.ylabel('Slope')
            plt.savefig(r'C:\Users\JOHNYA\Desktop\python_files\results\plots\Date {}, Device_number {}, Replicate {}_6, Signal, Slope'.format(date,device_no,replicate))
            plt.show()    
              
            # using subplot function and creating
            plt.subplots(figsize=(15, 5))
            plt.suptitle('Date {}, Device_number {}, Replicate {}_6'.format(
                date, device_no, replicate))
    
            plt.subplot(1, 3, 1)
            plt.scatter(t[1:], x[1:])
            plt.xlabel('Frames')
            plt.ylabel('X coordinate')
    
            plt.subplot(1, 3, 2)
            plt.scatter(t[1:], y[1:])
            plt.ylabel('Y coordinate')
            plt.xlabel('Frames')
    
            plt.subplot(1, 3, 3)
            plt.scatter(t[1:], r[1:])
            plt.ylabel('R coordinate')
            plt.xlabel('Frames')
            # plt.savefig(r'C:\Users\JOHNYA\Desktop\python_files\results\plots\Day {}, Device_number {}, Replicate {}_6, X, Y, R ccordinate'.format(day_no,device_no,replicate))
            plt.show()
            plt.close()

    
    def combine_dataframe(self):
        '''
        Combine all datframes into one dataframe

        Returns
        -------
        None.

        '''
        path = r'C:\Users\JOHNYA\Desktop\python_files\dictionary_values'
        files = os.path.join(path, "*.xlsx")
        filenames = glob.glob(files)
        combined_csv = pd.DataFrame()
        for file_name in filenames:
            
            x = pd.read_excel(file_name)
            combined_csv = pd.concat([combined_csv, x], axis=1)
            combined_csv.to_excel('combined_xlsx.xlsx',index=True)
            # print(combined_csv)
        print('done')
    
    
    def combine_csv(self):
        '''
        To combine all CSV into one.
        CSV contains values such as Device, Filename, time, X coordinate, Y coordinate, R coordinate, Minimum Radius, Maximum Radius

        Returns
        -------
        None.

        '''
        # setting the path for joining multiple files
        path = r'C:\Users\JOHNYA\Desktop\python_files\dictionary_values'
        files = os.path.join(path, "*.csv")
        filenames = glob.glob(files)
        combined_csv = pd.DataFrame()
        for file_name in filenames:
            x = pd.read_csv(file_name, low_memory=False)
            combined_csv = pd.concat([combined_csv, x], axis=0)
            print(combined_csv)

    def signal_extraction_scattering_study():
        '''
        Extract signals and plot

        Returns
        -------
        None.

        '''
        source = r"G:\.shortcut-targets-by-id\1Mu4lv_LWn6F08ie0hnbKRVze0TO-U0aJ\UnifudiOC\03_Dokumentation\1.3_Anprobe\Glucose_2022_001_BewertungSML3\01_BewertungGeräteStreuung\Glucose\*\*\*\\"

        # Plot absorbtion as function of time.
        plt.figure(figsize=(6.8, 6.8), dpi=600)

        result = pd.DataFrame()
        for files in glob.glob(source + '\\*_6.xlsx'):

            df = pd.read_excel(files)

            # Extracting device number
            device_no = files.split('\\')[-2].split('_')[-1][-2:]

            #  Extracting days
            day_no = files.split('\\')[-4].split('Day')[-1]

            # Extracting replicates
            replicate_no = files.split('_')[-2][0]

            for index in range(0, len(df.columns), 2):

                time = df.iloc[:, index]

                signal = df.iloc[:, index+1]

                signal = scipy.signal.savgol_filter(
                    signal, 21, 1, deriv=0, delta=1.0, axis=- 1, mode='nearest', cval=0.0)

                signal_120s = np.mean(signal[1200:1210])
                signal_125s = np.mean(signal[1250:1260])
                signal_130s = np.mean(signal[1300:1310])
                signal_135s = np.mean(signal[1350:1360])
                signal_140s = np.mean(signal[1400:1410])

                temp_result = {"Device_Number": device_no, "Day": day_no, "Replicate":  replicate_no, "Signal_at_120s": signal_120s,
                               "Signal_at_125s": signal_125s, "Signal_at_130s": signal_130s, "Signal_at_135s": signal_135s, "Signal_at_140s": signal_140s}

                result = result.append(temp_result, ignore_index=True)

        result.to_csv("result_scattering_device_glucose.csv",
                      sep=";", decimal=".")


if __name__ == '__main__':

    source = r'G:\.shortcut-targets-by-id\1Mu4lv_LWn6F08ie0hnbKRVze0TO-U0aJ\UnifudiOC\03_Dokumentation\1.3_Anprobe\Glucose_2022_001_BewertungSML3\01_BewertungGeräteStreuung\Glucose\*\*\\'
    counter = 1

    # Extract path for all data files.
    list_folder_messtag = glob.glob(source + "\\*_6.avi")
    list_folder_messtag_weiss = [
        element for element in list_folder_messtag if "white" in element]
    list_folder_messtag_schwarz = [
        element for element in list_folder_messtag if "black" in element]

    list_folder_messtag = [
        element for element in list_folder_messtag if "white" not in element]
    list_folder_messtag = [
        element for element in list_folder_messtag if "black" not in element]

    circle_detection = CircleDetection(
        list_folder_messtag, list_folder_messtag_weiss, list_folder_messtag_schwarz)
    circle_detection.plots()