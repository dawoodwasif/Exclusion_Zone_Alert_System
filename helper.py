from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube

import settings

import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np

first_frame = True

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, dz_box, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """
    W, H = (720, int(720*(9/16)))
    
    # Resize the image to a standard size
    image = cv2.resize(image, (W, H))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker, classes=0)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf, classes=0)
        
    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    
    ##############
    x3, y3, x4, y4, x5, y5, x6, y6 = dz_box
    points = np.array([[x3, y3], [x4, y4], [x5, y5], [x6, y6]], np.int32)
    points = points.reshape((-1, 1, 2))

    # Fill color (Light yellow with reduced alpha)
    fill_color = (0, 255, 255)
    alpha = 0.1

    # Create a black background image
    filled_polygon = np.zeros_like(res_plotted)

    # Draw the filled polygon on the black background
    cv2.fillPoly(filled_polygon, [points], fill_color)

    # Blend the filled polygon with the original image
    cv2.addWeighted(filled_polygon, alpha, res_plotted, 1 - alpha, 0, res_plotted)

    # Outline color (the same color as the bounding box)
    outline_color = (0,255,255)
    thickness = 2  # You can adjust the thickness as needed

    # Draw the polygon outline on the image
    cv2.polylines(res_plotted, [points], isClosed=True, color=outline_color, thickness=thickness)

    ##############

    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def play_youtube_video(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_youtube = st.sidebar.text_input("YouTube Video url")

    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detect Objects'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    # vid_cap = cv2.VideoCapture(source_rtsp)
                    # time.sleep(0.1)
                    # continue
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    
    st.write("Draw your polygon on the canvas below:")
    
    
    bg_video = source_webcam
    print(bg_video)
    drawing_mode = "polygon"

    stroke_width =  3
    stroke_color = "#000000"
    bg_color = "#eee"
    DZ_BOX = 10,24,45,52,42,21,100,220
    
    #bg_video = st.sidebar.file_uploader("Background video:", type=["mp4"])  # Accept mp4 video files
    realtime_update = True

    # Initialize video capture
    video_capture = None
    frame_width = 0
    frame_height = 0
    
    
    # Check if it's the first frame using session state
    if "first_frame_webcam" not in st.session_state:
     
        if bg_video==0:

            # Open the temporary video file
            video_capture = cv2.VideoCapture(bg_video)

            # Check if the video file was successfully opened
            #global first_frame
            if video_capture.isOpened():
                # Read the first frame
                ret, frame = video_capture.read()
                video_capture.release()
                st.session_state.first_frame_webcam = frame

                # if ret:
                #     # Extract frame dimensions
                #     frame_width, frame_height = (720, int(720*(9/16)))

    frame_width, frame_height = (720, int(720*(9/16)))
    canvas_width = frame_width
    canvas_height = frame_height

    # Convert BGR frame to RGB format
    frame_rgb = cv2.cvtColor(st.session_state.first_frame_webcam, cv2.COLOR_BGR2RGB)
    canvas_background_image = Image.fromarray(frame_rgb)
 

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=canvas_background_image,
        update_streamlit=realtime_update,
        height=canvas_height,
        width=canvas_width,
        drawing_mode=drawing_mode,
    )


    flag = False 
    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)
        object_data = canvas_result.json_data['objects']
        
        if len(object_data):
            path_data = object_data[0]['path']
            # Initialize variables for coordinates
            X1, Y1, X2, Y2, X3, Y3, X4, Y4 = 0, 0, 0, 0, 0, 0, 0, 0
            print(path_data)
            
            if len(path_data) == 5: ## additional z item
                st.success("Polygon successfully selected!")
                flag = True
                
                
            else:    
                st.error("Please select exacly 4 coordinates for Danger Zone!")
                flag = False

            # Iterate through the path data
            for item in path_data:
                if len(item) == 3:
                    command = item[0]
                    x = item[1]
                    y = item[2]
                else:
                    continue
                
                if command == 'M':
                    # First point
                    X1, Y1 = x, y
                elif command == 'L':
                    if X2 == 0 and Y2 == 0:
                        # Second point
                        X2, Y2 = x, y
                    elif X3 == 0 and Y3 == 0:
                        # Third point
                        X3, Y3 = x, y
                    else:
                        # Fourth point (if there are more than 3 points)
                        X4, Y4 = x, y

            # Print the extracted coordinates
            print("X1:", X1, "Y1:", Y1)
            print("X2:", X2, "Y2:", Y2)
            print("X3:", X3, "Y3:", Y3)
            print("X4:", X4, "Y4:", Y4)
        
    if flag and canvas_result.json_data is not None:
        ### display coordinates
        coordinates = [(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)]
        st.write("### Coordinates of Selected Polygon:")
        df = pd.DataFrame(coordinates, columns=["X", "Y"])
        df.index = df.index + 1
        st.table(df)
        
        ### display object dataframe
        st.write("### Dataframe of Canvas Object:")
        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")
        st.dataframe(objects)  
        
        DZ_BOX = X1, Y1, X2, Y2, X3, Y3, X4, Y4
        
        if st.button("Upload Polygon"):
            st.success("Coordinates successfuly uploaded")
    
    
    
    
    if st.sidebar.button('Detect Objects'):
        try:
            dz_box = DZ_BOX
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             dz_box,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)
        
        if "first_frame_vid" in st.session_state:
            del st.session_state.first_frame_vid
    

    st.write("Draw your polygon on the canvas below:")
   
    bg_video = str(settings.VIDEOS_DICT.get(source_vid))
    drawing_mode = "polygon"

    stroke_width =  3
    stroke_color = "#000000"
    bg_color = "#eee"
    DZ_BOX = 0,0,0,0,0,0,0,0
    
    #bg_video = st.sidebar.file_uploader("Background video:", type=["mp4"])  # Accept mp4 video files
    realtime_update = True

    # Initialize video capture
    video_capture = None
    frame_width = 0
    frame_height = 0

    if "first_frame_vid" not in st.session_state:

        if bg_video:

            # Open the temporary video file
            video_capture = cv2.VideoCapture(bg_video)

            # Check if the video file was successfully opened
            if video_capture.isOpened():
                # Read the first frame
                ret, frame = video_capture.read()
                video_capture.release()
                st.session_state.first_frame_vid = frame
            
    frame_width, frame_height = (720, int(720*(9/16)))
    canvas_width = frame_width
    canvas_height = frame_height


    # Convert BGR frame to RGB format
    frame_rgb = cv2.cvtColor(st.session_state.first_frame_vid , cv2.COLOR_BGR2RGB)
    canvas_background_image = Image.fromarray(frame_rgb)
  

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=canvas_background_image,
        update_streamlit=realtime_update,
        height=canvas_height,
        width=canvas_width,
        drawing_mode=drawing_mode,
    )


    flag = False 
    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)
        object_data = canvas_result.json_data['objects']
        
        if len(object_data):
            path_data = object_data[0]['path']
            # Initialize variables for coordinates
            X1, Y1, X2, Y2, X3, Y3, X4, Y4 = 0, 0, 0, 0, 0, 0, 0, 0
            print(path_data)
            
            if len(path_data) == 5: ## additional z item
                st.success("Polygon successfully selected!")
                flag = True
                
                
            else:    
                st.error("Please select exacly 4 coordinates for Danger Zone!")
                flag = False

            # Iterate through the path data
            for item in path_data:
                if len(item) == 3:
                    command = item[0]
                    x = item[1]
                    y = item[2]
                else:
                    continue
                
                if command == 'M':
                    # First point
                    X1, Y1 = x, y
                elif command == 'L':
                    if X2 == 0 and Y2 == 0:
                        # Second point
                        X2, Y2 = x, y
                    elif X3 == 0 and Y3 == 0:
                        # Third point
                        X3, Y3 = x, y
                    else:
                        # Fourth point (if there are more than 3 points)
                        X4, Y4 = x, y

            # Print the extracted coordinates
            print("X1:", X1, "Y1:", Y1)
            print("X2:", X2, "Y2:", Y2)
            print("X3:", X3, "Y3:", Y3)
            print("X4:", X4, "Y4:", Y4)
        
    if flag and canvas_result.json_data is not None:
        ### display coordinates
        coordinates = [(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)]
        st.write("### Coordinates of Selected Polygon:")
        df = pd.DataFrame(coordinates, columns=["X", "Y"])
        df.index = df.index + 1
        st.table(df)
        
        ### display object dataframe
        st.write("### Dataframe of Canvas Object:")
        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")
        st.dataframe(objects)  
    
        DZ_BOX = X1, Y1, X2, Y2, X3, Y3, X4, Y4
        
        if st.button("Upload Polygon"):
            st.success("Coordinates successfuly uploaded")
            ########################
                        
            #######################
        

    if st.sidebar.button('Detect Video Objects'):
        try:
            dz_box = DZ_BOX
            print(dz_box)
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             dz_box,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
