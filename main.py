import cv2
import torch
from torchvision import transforms
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, Label, Button, Frame
from src.train_model import DigitRecognizer

# Load the pre-trained model
model = DigitRecognizer()
model.load_state_dict(torch.load("./models/final_model.pth"))
model.eval()  # Set the model to evaluation mode

# Define the transformation for preprocessing
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tensor_image = transform(frame).unsqueeze(0)  # Add batch dimension
    return tensor_image

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert to grayscale
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(image_path)
    tensor_image = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor_image

# Function to visualize the prediction
def view_classify(img, ps):
    ''' Function to view an image and its predicted classes. '''
    ps = ps.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), nrows=2)
    ax1.imshow(img, cmap='gray')
    ax1.axis('off')
    ax2.barh(np.arange(10), ps, color="blue")
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    return fig

def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    # Preprocess the image
    image_tensor = preprocess_image(file_path)

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        predicted_digit = output.argmax(dim=1).item()
        probabilities = torch.exp(output).squeeze()

    # Display the predicted digit
    result_label.config(text=f"Predicted Digit: {predicted_digit}")

    # Display the image and class probabilities using the new plot
    #fig = view_classify(image_tensor, probabilities)
    #canvas = FigureCanvasTkAgg(fig, master=frame)
    #canvas_widget = canvas.get_tk_widget()
    #canvas_widget.pack()
    #canvas.draw()


# Function to handle live camera feed
def start_camera_feed():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Convert the captured frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess the frame
        image_tensor = preprocess_frame(frame_rgb)

        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            predicted_digit = output.argmax(dim=1).item()
            probabilities = torch.exp(output).squeeze()

        # Overlay the predicted digit on the frame
        cv2.putText(
            frame,
            f"Predicted Digit: {predicted_digit}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Show the frame
        cv2.imshow("Live Digit Recognition", frame)

        # Exit the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI setup
root = Tk()
root.title("Live Digit Recognition")

frame = Frame(root)
frame.pack(pady=20)

# Create and place widgets
upload_btn = Button(frame, text="Upload Image", command=upload_and_predict)
upload_btn.pack(pady=10)
result_label = Label(root, text="Predicted Digit: ", font=("Helvetica", 16))
result_label.pack(pady=10)

Label(frame, text="Press the button below to start live digit detection", font=("Helvetica", 16)).pack(pady=10)

Button(frame, text="Start Live Feed", command=start_camera_feed, font=("Helvetica", 14)).pack(pady=10)

Button(frame, text="Quit", command=root.quit, font=("Helvetica", 14)).pack(pady=10)

root.mainloop()
