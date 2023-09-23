# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper


#--------------------------------Web Page Designing------------------------------
hide_menu_style = """
    <style>
        MainMenu {visibility: hidden;}
        
        
         div[data-testid="stHorizontalBlock"]> div:nth-child(1)
        {  
            border : 2px solid #doe0db;
            border-radius:5px;
            text-align:center;
            color:black;
            font-weight:bold;
            padding: 25px;
            
        }
        
        div[data-testid="stHorizontalBlock"]> div:nth-child(2)
        {   
            border : 2px solid #doe0db;
            border-radius:5px;
            text-align:center;
            font-weight:bold;
            color:black;
            padding: 25px;
            
        }
    </style>
    """

main_title = """
            <div>
                <h1 style="color:white;
                text-align:center; font-size:35px;
                margin-top:-95px;">
                Danger Zone Alert System</h1>
            </div>
            """
    
sub_title = """
            <div>
                <h6 style="color:dodgerblue;
                text-align:center;
                margin-top:-40px;">
                Using YOLOv8 Human Detection and Tracking </h6>
            </div>
            """
#--------------------------------------------------------------------------------



# Setting page layout
st.set_page_config(
    page_title="Danger Zone Alert System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(hide_menu_style, 
                unsafe_allow_html=True)

st.markdown(main_title,
                unsafe_allow_html=True)

st.markdown(sub_title,
                unsafe_allow_html=True)


# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence (%)", 0, 100, 30)) / 100

iou_threshold = float(st.sidebar.slider(
    "Select IoU Threshold (%)", 25, 100, 5)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is not None:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is not None:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
