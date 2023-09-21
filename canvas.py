import streamlit as st
from streamlit_javascript import st_javascript

import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2
import tempfile

def select_polygon(bg_video):
    coordinates = []

    #st.set_page_config(layout='wide')

    # Specify canvas parameters in application
    # drawing_mode = st.sidebar.selectbox(
    #     "Drawing tool:", ("polygon", "freedraw", "line", "rect", "circle", "transform", "point")
    # )
    
    drawing_mode = "polygon"

    stroke_width =  3
    stroke_color = "#000000"
    bg_color = "#eee"
    
    print(stroke_width,stroke_color, bg_color)

    #bg_video = st.sidebar.file_uploader("Background video:", type=["mp4"])  # Accept mp4 video files
    realtime_update = True

    # Initialize video capture
    video_capture = None
    frame_width = 0
    frame_height = 0

    if bg_video:
        # Save the uploaded video to a temporary file
        # with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        #     temp_file.write(bg_video.read())
        #     temp_filename = temp_file.name

        # Open the temporary video file
        video_capture = cv2.VideoCapture(bg_video)

        # Check if the video file was successfully opened
        if video_capture.isOpened():
            # Read the first frame
            ret, frame = video_capture.read()

            if ret:
                # Extract frame dimensions
                frame_height, frame_width, _ = frame.shape
                
                print(type(frame))
                print(frame.shape)
    else:
        frame_height, frame_width = 100, 100

    # bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    # realtime_update = st.sidebar.checkbox("Update in realtime", True)


    # if bg_image:
    #     img = Image.open(bg_image)
    #     w, h = img.size
    # else:
    #     w, h = 100, 100

    # if option == 'Default Width':
    #     inner_width = 100
    # elif option == 'Update Canvas Width':
    #     #inner_width = st_javascript("""await fetch("http://localhost:8501/").then(function(response) {
    #     #    return window.innerWidth;
    #     #}) """)
    #     inner_width = 848

    # st.markdown(f"Inner width was: {inner_width}")

    # canvas_width = inner_width
    # canvas_height = h * (canvas_width / w)

    canvas_width = frame_width
    canvas_height = frame_height

    if frame_width > 100 and frame_height > 100:
        # Convert BGR frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        canvas_background_image = Image.fromarray(frame_rgb)
    else:
        canvas_background_image = None

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
        key = "c21"
    )


    flag = False 
    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)
        object_data = canvas_result.json_data['objects']
        
        if len(object_data):
            print(object_data)
            path_data = object_data[0]['path']
            # Initialize variables for coordinates
            X1, Y1, X2, Y2, X3, Y3, X4, Y4 = 0, 0, 0, 0, 0, 0, 0, 0
            print(path_data)
            
            if len(path_data) == 5: ## additional z item
                st.success("Polygon successfully seleected!")
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

    if st.button("Upload Coordinates"):
        return coordinates
    else:
        return []

    
            


    # if st.button("Upload Coordinates"):
    #     print("After")
    #     return coordinates