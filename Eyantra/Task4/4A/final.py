import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

def get_subimage(image, x1, y1, x2, y2):
    subimage = image[y1:y2, x1:x2]
    if subimage.size == 0:
        return None
    # subimage = cv2.rotate(subimage, cv2.ROTATE_90_CLOCKWISE)
    # cv2.imshow(f"Sub-Image", subimage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return subimage

def preprocess_image(image):
    if isinstance(image, np.ndarray):  # Check if the input is a NumPy array
        image = Image.fromarray(image)  # Convert NumPy array to PIL Image

    image = image.convert('RGB')
    image = np.array(image)
        
    image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_CUBIC)
    # image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_CUBIC)
    # image = cv2.resize(image, (182, 182), interpolation=cv2.INTER_CUBIC)
    # image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
   	])
    input_tensor = transform(image)
        
    input_batch = input_tensor.unsqueeze(0) 
    return input_batch


def classify_subimage(subimage, model):
    if subimage is None:
        return None, None

    # Preprocess the subimage for classification
    subimage = preprocess_image(subimage)

    # Perform classification using the pre-trained model
    with torch.no_grad():
        output = model(subimage)

    # Get the predicted class and confidence
    confidence, predicted_class = torch.max(output.data, 1)

    return confidence.item(), predicted_class.item()

# Map the class labels to their corresponding folder names
class_to_label = {
    0: "combat",
    1: "humanitarianaid",
    2: "militaryvehicles",
    3: "fire",
    4: "destroyedbuilding"
}

if __name__ == '__main__':
    # Load the pre-trained classification model
    pretrained_model = torch.load('trained_model_final.pth')  # Adjust the model file name accordingly
    pretrained_model.eval()

    # Open a connection to the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(2)
    # Set frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Adjust as needed
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Adjust as needed


    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Extract subimages
        subimages = [
            # military vehicle
            {'box': (475*1920//1280 - 0, 630*1080//720 - 0, 533*1920//1280 - 0, 688*1080//720 + 0)},
            # aid
            {'box': (826*1920//1280 - 5, 482*1080//720 - 0, 884*1920//1280 - 0, 539*1080//720 - 0)},
            # fire
            {'box': (831*1920//1280 - 3, 333*1080//720 - 0, 888*1920//1280 - 4, 390*1080//720 - 0)},
            # destroyed building
            {'box': (466*1920//1280 - 3, 331*1080//720 - 0, 524*1920//1280 - 0, 388*1080//720 - 3)},
            # combat
            {'box': (477*1920//1280 - 0, 91*1080//720 - 0, 537*1920//1280 - 5, 149*1080//720 - 0)},
        ]

        # Classify and draw bounding boxes on the original frame
        for subimage_info in subimages:
            subimg = get_subimage(frame, *subimage_info['box'])
            confidence, prediction = classify_subimage(subimg, pretrained_model)

            if confidence is not None and prediction is not None:
                predicted_class_name = class_to_label[prediction]
                
            cv2.imshow(f"Sub-Image", subimg)
            print(predicted_class_name, confidence)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
            # cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
