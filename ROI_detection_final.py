import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import final_with_class as fwc
import matplotlib.pyplot as plt
import time
from bokeh.plotting import figure


class streamlit_generator:
   
   def __init__(self):
        
        # **********************************************************
        st.title('Smart MiniLab')
        
        # Displays in main portion
        self.uploaded_file = st.file_uploader("Video File of Measurement", accept_multiple_files=True, type = ["avi"])    
        self.uploaded_file_avi_black = st.file_uploader("Video File of Black Normalization Measurement", accept_multiple_files=False, type = ["avi"])      
        self.uploaded_file_avi_white = st.file_uploader("Video File of White Normalization Measurement", accept_multiple_files=False, type = ["avi"])      
       
        if(self.uploaded_file is not None):
            #To DO Loop
            for uploaded_files in self.uploaded_file:
                # bytes_data = uploaded_file.read()
                # st.write("filename:", uploaded_file.name)
                # st.write(bytes_data)
                self.uploaded_file_path = self.save_uploadedfile(uploaded_files)
           
        if(self.uploaded_file_avi_black is not None):
            self.uploaded_file_path_black = self.save_uploadedfile(self.uploaded_file_avi_black)

        if(self.uploaded_file_avi_white is not None):            
            self.uploaded_file_path_white = self.save_uploadedfile(self.uploaded_file_avi_white)
            
        # **********************************************************
        # Displays in sidebar, various options to choose
        st.sidebar.header("Settings")
        
        # Selection of color channel, use st.sidebar.radio for displaying as bullet points
        
        self.color_channel = st.sidebar.selectbox("Choose color channel for intensity computation",('Red', 'Blue', 'Green', 'Hue', ' Saturation', 'Value'))
        self.color_channel_index = self.get_color_channel_index(self.color_channel)
        
        # Nomalization or not
        # self.normalization = st.sidebar.radio('Perform normalization?', ('Yes','No'))
        # if self.normalization == "Yes":        
        #     self.color_channel_normalization = st.sidebar.selectbox('Select color channel for normalization?', ('Red', 'Blue', 'Green', 'Hue', ' Saturation', 'Value'))
        #     self.color_channel_normalization_index = self.get_color_channel_index(self.color_channel_normalization)
        # elif self.normalization == "No":
        #     self.color_channel_normalization_index = None
        
        
        # Choose between diffent algorithms
        self.algorithm = st.sidebar.radio('Choose algorithm', ('Algorithm 1', 'Algorithm 2'))
        
        self.placeholder_computation = st.empty()
        self.computation_button = self.placeholder_computation.button('Evaluation', key = 0)

        if self.computation_button:
                   
            with st.spinner('Evaluation'):

                # Algorithm 1- 1. Gaussian Blur, 2. Erosion, 3. Otsu thresholding, 4. Morphological closing, 5. Canny edge detection, 6. Connected components, 7. Circle detection
                if self.algorithm == 'Algorithm 1':
                    for uploaded_files in self.uploaded_file:
                        if (uploaded_files is not None) and (self.uploaded_file_avi_white is not None) and (self.uploaded_file_avi_black is not None):
               
                            time.sleep(1)
                            
                            self.algorithm_1(self.uploaded_file_path, self.uploaded_file_path_white, self.uploaded_file_path_black)
                 
        # **********************************************************
   
   def algorithm_1(self, uploaded_file_path, uploaded_file_path_white, uploaded_file_path_black):
    
      auto_circle = fwc.CircleDetection(uploaded_file_path, uploaded_file_path_white, uploaded_file_path_black)
      dict_values = auto_circle.convert_video_to_images()
      t = dict_values['Time']
      
      signal = dict_values['Signal']
      slope = dict_values['Slope']
      date = dict_values['Date']
      replicate = dict_values['Replicate']
      device_no = dict_values['Device']
      #***************************************************************** 
      
      p_signal = figure(
      title='Date {}, Device_number {}, Replicate {}_6'.format(
          date, device_no, replicate),
      x_axis_label='Frames',
      y_axis_label='Signal')
      p_signal.line(t[1:], signal[1:], legend_label='Trend', line_width=2)

      st.bokeh_chart(p_signal, use_container_width=True)
      #***************************************************************** 
      
      p_slope = figure(
      title='Date {}, Device_number {}, Replicate {}_6'.format(
          date, device_no, replicate),
      x_axis_label='Frames',
      y_axis_label='Slope')
      p_slope.line(t[1:], slope[1:], legend_label='Trend', line_width=2)

      st.bokeh_chart(p_slope, use_container_width=True)
      #***************************************************************** 
      
      return None

   def display_graph(self, path):
       st.image(path)
           
   def total_number_of_frames(path_avi):
       
       cap = cv2.VideoCapture(path_avi)
           
       frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) 

       cap.release()     
              
       return int(frame_count)
   
   def save_uploadedfile(self, uploadedfile):
      
        with open(os.path.join("temp", uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())
        
         
        return os.path.join("temp", uploadedfile.name)
        

   def get_color_channel_index(self, color_channel):
       
        if self.color_channel == 'Red':
            self.color_channel_index = 2
        elif self.color_channel == 'Blue':
            self.color_channel_index = 1
        elif self.color_channel == 'Green':
            self.color_channel_index = 0
        elif self.color_channel == 'Hue':
            self.color_channel_index = 2
        elif self.color_channel == 'Saturation':
            self.color_channel_index = 1
        elif self.color_channel == 'Value':
            self.color_channel_index = 0
            
        
   def load_image(self, path, frame_number = 1):
            
       # Opens the Video file
       vidcap = cv2.VideoCapture(path)
       vidcap.set(1, frame_number)
       success,image = vidcap.read()
           
       vidcap.release()
               
       if(success):
           return image
       else:
           st.write("Error. Could not load image frame.")
           
           return None
        
if __name__ == '__main__':
   streamlit_generator()
   
   