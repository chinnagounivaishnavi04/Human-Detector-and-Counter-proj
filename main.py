import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLO model
net = cv2.dnn.readNet('C:/Users/chinn/OneDrive/Desktop/vaishu/yolo/yolov4.weights', 
                       'C:/Users/chinn/OneDrive/Desktop/vaishu/yolo/yolov4.cfg')

# Load COCO class names
with open("C:/Users/chinn/OneDrive/Desktop/vaishu/yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get the output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Initialize webcam
cap = cv2.VideoCapture(0)  # Change to 0 for webcam or specify video file path

plt.ion()  # Turn on interactive mode for matplotlib

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width, _ = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Process the outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]  # Get scores for each class
            class_id = np.argmax(scores)  # Get the class with the highest score
            confidence = scores[class_id]  # Get the confidence for the class
            if confidence > 0.5:  # Confidence threshold
                # Get the bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and count humans
    human_count = 0
    for i in indices.flatten():  # Flatten the indices for iteration
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        
        if label == "person":
            human_count += 1  # Count the number of humans
            color = (0, 255, 0)  # Green color for humans
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidences[i], 2)), 
                        (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)

    print(f"Human Count: {human_count}")  # Print the count of detected humans

    # Convert BGR frame to RGB for displaying with matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use Matplotlib to display the frame
    plt.imshow(frame_rgb)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.pause(0.01)  # Pause to allow for updating the plot

    # Exit the loop if needed; you can define your own way to exit
    if plt.fignum_exists(1) == False:  # Check if the figure is still open
        break

# Release the video capture
cap.release()
plt.close()  # Close the plot
