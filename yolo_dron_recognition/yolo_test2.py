import cv2
import numpy as np
import time

# Open the camera
cap = cv2.VideoCapture(0)

# Get the start time for FPS calculation
start_time = time.time()
frame_id = 0

# Main loop
while True:
    # Take a frame from the camera
    ret, frame = cap.read()

    # If a frame is not received, end the loop
    if not ret:
        break

    # Flip the frame horizontally (mirror view) and resize it
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (416, 416))
    
    frame_w = frame.shape[1]
    frame_h = frame.shape[0]
    # Convert the frame image into a format that the model can process (blob)
    frame_blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    
    # set labels
    labels = ["iha"]

    # Bounding box colours
    colors = ["0,0,255", "0,0,255", "255,0,0", "255,255,0", "0,255,0"]
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    colors = np.array(colors)
    colors = np.tile(colors, (18, 1))

    #Install YOLOv4 model
    model = cv2.dnn.readNetFromDarknet("yolov4-custom.cfg", "yolov4-custom_last.weights")
    layers = model.getLayerNames()
    output_layer = [layers[i-1] for i in model.getUnconnectedOutLayers()]

    # Feed the model with frame data
    model.setInput(frame_blob)
    
    # Get detection layers
    detection_layers = model.forward(output_layer)
    
    # Create empty lists for detection results
    id_list = []
    boxes_list = []
    confidences_list = []
    
    # Process detection results
    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]
            
            # Process detections above a certain confidence threshold (0.20)
            if confidence > 0.20:
                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([frame_w, frame_h, frame_w, frame_h])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")

                start_x = int(box_center_x - (box_width) / 2)
                start_y = int(box_center_y - (box_height) / 2)

                id_list.append(predicted_id)
                confidences_list.append(float(confidence))
                boxes_list.append([start_x, start_y, int(box_width), int(box_height)])
    
    # Select detections with the highest confidence with Non-Maximum Suppression (NMS)
    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)
    
    # Process selected detections and draw bounding boxes
    for max_id in max_ids:
        max_class_id = max_id
        box = boxes_list[max_class_id]

        start_x = box[0]
        start_y = box[1]
        box_width = box[2]
        box_height = box[3]

        predicted_id = id_list[max_class_id]
        label = labels[predicted_id]
        confidence = confidences_list[max_class_id]

        end_x = start_x + box_width
        end_y = start_y + box_height

        box_color = colors[predicted_id]
        box_color = [int(each) for each in box_color]
        
        # Labelling and bounding box drawing
        label = "{}: {:.2f}%".format(label, confidence * 100)

        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
        #cv2.rectangle(frame, (start_x - 1, start_y), (end_x + 1, end_y - 30), box_color, 2)
        cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Calculate FPS and print it on the screen
    frame_id += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame image
    cv2.imshow("Detector", frame)

     # End the loop when "q" is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
