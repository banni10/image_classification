import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from mlp import *
from cnn import *
from classify import *
# from your_image_classification_script import classify_image
import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from io import BytesIO


# Load MNIST dataset using PyTorch
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
data_samples = 60000
split_ratio = 0.5
model = "cnn"

# Function to visualize MNIST samples
def visualize_samples(num_samples):
    st.write(f"Visualizing {num_samples} MNIST samples")
    images = []
    labels = []

    for i in range(num_samples):
        image, label = train_dataset[i]
        image = image.squeeze().numpy()
        image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Scale to range [0, 1]
        images.append(image)
        labels.append(label)
    st.image(images, width=100, caption=[f"Label = {i}" for i in labels])

# Function to display table
def generate_data(data_samples, split_ratio):
    data = {
    "Train Data": [split_ratio*data_samples],
    "Test Data": [(1-split_ratio)*data_samples]
}
    df = pd.DataFrame(data)
    df = df.astype('int')
    st.table(df)
    

# Function for Home Page
def home():
    st.title("Handwritten Digits Classification Project")
    st.image("https://www.altexsoft.com/media/2019/12/https-lh5-googleusercontent-com-lirteaajnd3hx43l.gif")


# Function for About Page
def about():
    st.title("Welcome to our MNIST Handwritten Digits Classification project!")
    st.write(" This application aims to showcase the fascinating world of computer vision and machine learning by demonstrating the recognition and classification of handwritten digits using the famous MNIST dataset. The MNIST dataset consists of a vast collection of handwritten digits (0-9), widely used as a benchmark in the field of machine learning.")
    st.write("Our app utilizes state-of-the-art machine learning techniques and neural networks to analyze and classify these handwritten digits. With the power of deep learning, it can predict the digit drawn on the canvas with impressive accuracy.")
    st.write("Explore the intricacies of image recognition and witness firsthand how machine learning algorithms decipher handwritten digits. Whether you're new to machine learning or an enthusiast, this app provides an interactive platform to understand the magic behind digit classification.")
    st.write("Join us on this exciting journey into the realm of machine learning and witness the wonder of technology in action!")
    st.image("img_ml.webp")

# Function for Dataset Page
def dataset():
    st.title("MNIST")
    st.write("MNIST is a dataset containing 70,000 handwritten digits (0-9) as 28x28 pixel grayscale images. It's a fundamental resource for training and testing image classification models in machine learning and computer vision.")
    st.subheader("No. Of Samples")
    # Slider to adjust the number of samples
    num_samples = st.slider("Select number of samples", min_value=1, max_value=20, value=10)

    # Button to visualize the selected number of samples
    if st.button("Visualize Data"):
        visualize_samples(num_samples)

    st.title("Train-Test Split")
    st.write("Select the total number of data samples you want to use. Also specify the ratio int which these samples should be divided for training and testing.")
    st.subheader("Data Samples")
    # Slider to adjust the number of samples
    global data_samples
    data_samples = st.slider("Select number of data samples", min_value=1000, max_value=60000, value=10000)

    st.subheader("Split Ratio")
    # Slider to adjust the split ratio
    global split_ratio
    split_ratio = st.slider("Select split ratio", min_value=0.0, max_value=1.0, value=0.5)

    # Button to generate data
    if st.button("Generate Data"):
        generate_data(data_samples, split_ratio)
    

# Function for Dataset Page
def model1():
    st.title("Design A Neural Network")
    st.write("You can choose the number of layers, the kind of layers and the type of model you want to train your model.")
    st.subheader("Model Design")
    # Main menu options
    main_options = ["MLP", "CNN"]
    # Display the main menu
    selected_option = st.selectbox("Select an option", main_options)
    global model
    model = selected_option
    # submenu options
    submenu_options = {
        "MLP": {
            "Linear": ["1","2","3"],
            "Activations": ["ReLU", "Sigmoid"]
        },
        "CNN": {
            "Convolutional": ["1","2","3"],
            "Linear": ["1","2","3"],
            "Activations": ["ReLU", "Sigmoid"]
        }
    }

    # Display the cascading menu
    # selected_option = st.selectbox("Select an option", list(main_options.keys()))

    if selected_option:
        # st.write(f"You selected: {selected_option}")
        if selected_option in submenu_options:
            submenu_1 = st.selectbox(f"Select a submenu for {selected_option}", list(submenu_options[selected_option].keys()))

            if submenu_1:
                with st.expander(f"{submenu_1} Submenu"):
                    sub_option = st.selectbox("Select a sub-option", submenu_options[selected_option][submenu_1])
    
    if st.button("Design"):
        st.write("Model:",selected_option)
        st.write(submenu_1, ":", sub_option)
        if selected_option=='MLP':
            st.image("mlp1.webp",caption="An example")
        if selected_option=='CNN':
            st.image("cnn1.webp",caption="An example")



# Function for Dataset Page
def train1():
    st.title("Train Your Model")
    st.subheader("Train the Network")
    st.write("Specify the number of epochs, batch size, learning rate for training.")
    st.write("TRAIN PARAMETERS")
    num_epochs = st.slider("EPOCHS", min_value=1, max_value=50, value=10)
    batch_size = st.slider("BATCH SIZE", min_value=1, max_value=100, value=50)
    learning_rate = st.slider("LEARNING RATE",  min_value=0.001, max_value=0.010, value=0.005, step=0.001, format="%f")
    # Initialize empty lists to store loss and accuracy values
    epoch_loss = []
    epoch_accuracy = []
    # Button to generate data
    if st.button("Train Model"):
        global data_samples
        global split_ratio
        global model
        # st.write(model)
        st.write("working..")
        if model == 'MLP':
            epoch_loss, epoch_accuracy = train_mlp_model(data_samples, split_ratio,num_epochs,learning_rate, batch_size)
        elif model == 'CNN':
            epoch_loss, epoch_accuracy = train_cnn_model(data_samples, split_ratio,num_epochs,learning_rate, batch_size)     
        else:
            exit('Error: unrecognized model')
        
        for i in range(num_epochs):
            # Assuming epoch_accuracy and epoch_loss are lists containing values for each epoch
            st.write(f"EPOCH {i}:   Training Accuracy: {epoch_accuracy[i]*100:.2f}%,    Training Loss: {epoch_loss[i]:.2f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.plot(range(1, num_epochs + 1), epoch_loss, marker='o', linestyle='-', color='b')
        ax1.set_title('Loss per Epoch')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_xticks(range(1, num_epochs + 1))
        ax2.plot(range(1, num_epochs + 1), epoch_accuracy, marker='o', linestyle='-', color='r')
        ax2.set_title('Accuracy per Epoch')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_xticks(range(1, num_epochs + 1))
        # Display the Matplotlib plot in Streamlit using the st.pyplot() function
        st.pyplot(fig)

def upload_image():
    st.write("Upload image")
    # Upload image through Streamlit
    file_up = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    global model
    # Check if an image is uploaded
    if file_up is not None:
        image = Image.open(file_up)
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform image classification
        if st.button("Predict"):
            image = Image.open(file_up).convert('L')  # Open image in grayscale mode
            if model == 'MLP':
                result = classify_mlp_image(image)
            elif model == 'CNN':
                result = classify_cnn_image(image)
            else:
                exit('Error: unrecognized model')
            
            # Display classification results
            st.write("Classification Result:", result)

def draw_image():
    

    # Use a checkbox to toggle the display of the canvas
    display_canvas = st.checkbox("Display Canvas")

    # Only render the canvas if the checkbox is checked
    if display_canvas:
        # Specify canvas parameters in application
        drawing_mode = st.sidebar.selectbox(
            "Drawing tool:",
            ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
        )
        stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
        if drawing_mode == "point":
            point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
        stroke_color = st.sidebar.color_picker("Stroke color hex: ")
        bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
        bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
        realtime_update = st.sidebar.checkbox("Update in realtime", True)
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=Image.open(bg_image) if bg_image else None,
            update_streamlit=True,
            height=150,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == "point" else 0,
            display_toolbar=st.sidebar.checkbox("Display toolbar", True),
            key="full_app",
        )
        global model
        # Display the canvas
        if canvas_result.image_data is not None:
            # Create an image from the canvas data
            image = Image.fromarray(canvas_result.image_data.astype("uint8"))
            # Save the image to BytesIO buffer
            img_buffer = BytesIO()
            image.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            # Set the converted image for classification
            converted_image = image
            # image = st.image(canvas_result.image_data)
            # Perform image classification
            if st.button("Predict Image"):
                image = converted_image.convert('L')
                if model == 'MLP':
                    result = classify_mlp_image(image)
                elif model == 'CNN':
                    result = classify_cnn_image(image)
                else:
                    exit('Error: unrecognized model')
            
                # Display classification results
                st.write("Classification Result:", result)
        # if canvas_result.json_data is not None:
        #     objects = pd.json_normalize(canvas_result.json_data["objects"])
        #     for col in objects.select_dtypes(include=["object"]).columns:
        #         objects[col] = objects[col].astype("str")
        #     st.dataframe(objects)

# Function for Dataset Page
def predict1():
    st.title("Make Predictions On Your Data")
    st.subheader("Test Data")
    menu = ["UPLOAD IMAGE", "DRAW IMAGE"]
    tabs = {"UPLOAD IMAGE": upload_image, "DRAW IMAGE": draw_image}
    tab1, tab2 = st.tabs(menu)
    
    with tab1:
        tabs["UPLOAD IMAGE"]()
    with tab2:
        tabs["DRAW IMAGE"]()


    

# Streamlit app code
def main():
    
    st.write("")
    menu = ["Home", "About", "Dataset", "Model", "Train", "Predict"]
    tabs = {"Home": home, "About": about, "Dataset": dataset, "Model": model1, "Train": train1, "Predict": predict1}
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(menu)
    
    with tab1:
        tabs["Home"]()
    with tab2:
        tabs["About"]()
    with tab3:
        tabs["Dataset"]()
    with tab4:
        tabs["Model"]()
    with tab5:
        tabs["Train"]()
    with tab6:
        tabs["Predict"]()

if __name__ == "__main__":
    main()



