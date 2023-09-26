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

from utils import calculate_iou

first_frame = True

# Initialize
breach_frequency = {}
prev_active_breachers = set()


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


def _display_detected_frames(conf, model, st_frame, image, dz_box, is_display_tracking=None, tracker=None, first_frame_flag = False):
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
    
    
    ############## Draw horizontal line at bottom 10% #################
    IDs = res[0].boxes.id
    XYXYs = res[0].boxes.xyxy
    
    for i in range(len(XYXYs)):
        x_min, y_min, x_max, y_max = map(int, XYXYs[i].tolist())
        box_height = y_max - y_min
        line_y = int(y_max - 0.1 * box_height)  # Calculate the y-coordinate for the line
        
        # Draw a horizontal line inside the bounding box
        cv2.line(res_plotted, (x_min, line_y), (x_max, line_y), (0, 0, 255), 2)
    
    #####################################################################
    
    ################## Draw Danger Zone Polygon #########################
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

    ################################################################
    
    ############## Find Breachers in the Danger Zone ################
    
    # Calculate IoU for each box with the polygon
    iou_threshold = 0.01
    polygon = [(x3, y3), (x4, y4), (x5, y5), (x6, y6)]

    
    iou_values = [calculate_iou(box, polygon) for box in XYXYs]

    try:
        # Filter IDs that exceed the IoU threshold
        intersected_ids = [int(IDs[i]) for i in range(len(IDs)) if iou_values[i] > iou_threshold]
    except:
        intersected_ids = []
    
    #################################################################
    
    
    #################### Count and Display Breachers ################
    
    # Initialize
    global breach_frequency, prev_active_breachers 
    
    if first_frame_flag: 
        breach_frequency = {}
        prev_active_breachers = set()

    # For each frame, update the active_breachers list
    active_breachers = intersected_ids  # This should be updated for each frame

    # Convert current list to a set for efficient operations
    current_set = set(active_breachers)

    # Find new entries: people who are currently active but weren't in the previous frame
    new_entries = current_set - prev_active_breachers

    # Update the breach frequency for each new entry
    for person_id in new_entries:
        breach_frequency[person_id] = breach_frequency.get(person_id, 0) + 1

    # Set the current active breachers as the previous for the next iteration
    prev_active_breachers = current_set

    # Display the entry counts for each person
    for id, count in breach_frequency.items():
        print(f"Person {id}: Entered {count} times.")
  

    # Display the entry counts on the image frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    font_thickness = 1
    x, y = 10, 20  # Position to display the text

    for id, count in breach_frequency.items():
        text = f'ID: {id}, Count: {count}'
        cv2.putText(res_plotted, text, (x, y), font, font_scale, font_color, font_thickness)
        y += 20  # Increase the y-coordinate for the next text line

    # Display total count at the end
    total_count = sum(breach_frequency.values())
    total_text = f'Total Count: {total_count}'
    cv2.putText(res_plotted, total_text, (x, y), font, font_scale, font_color, font_thickness+1)
    y += 20  # Increase the y-coordinate for the next text line

    # Write statistics to a text file
    with open("stats.txt", "w") as f:
        f.write("Unique IDs: " + str(len(breach_frequency)) + "\n")
        f.write("Total Count: " + str(total_count) + "\n")
        f.write("ID Frequencies:\n")
        for id, count in breach_frequency.items():
            f.write(f'ID: {id}, Count: {count}\n')

    #################################################################


    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


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
    
    st.write("### Select Danger Zone:")
    st.write("Draw your polygon on the canvas below:")
    
    
    bg_video = source_webcam
    print(bg_video)
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
        st.write("### Real Time Detection:")
        if "first_frame_detection" in st.session_state:
            del st.session_state.first_frame_detection
        try:
            dz_box = DZ_BOX
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                first_frame_flag = False
                if "first_frame_detection" not in st.session_state:
                        st.session_state.first_frame_detection = True
                        first_frame_flag = True
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             dz_box,
                                             is_display_tracker,
                                             tracker,
                                             first_frame_flag
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
        
        if "first_frame_detection" in st.session_state:
            del st.session_state.first_frame_detection
    
    st.write("### Select Danger Zone:")
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
            st.write("### Real Time Detection:")
            dz_box = DZ_BOX
            print(dz_box)
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                first_frame_flag = False
                success, image = vid_cap.read()
                
                if "first_frame_detection" not in st.session_state:
                        st.session_state.first_frame_detection = True
                        first_frame_flag = True
    
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             dz_box,
                                             is_display_tracker,
                                             tracker,
                                             first_frame_flag,

                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
