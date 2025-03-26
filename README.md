# hardhat_detection_on_construction_site

Use YOLO to detect people, vest and different color hardhats at a construction site

## Project Overview

This project leverages the YOLO (You Only Look Once) object detection algorithm to identify and classify safety equipment on a construction site. The primary goal is to detect people, safety vests, and hardhats of different colors to ensure compliance with safety regulations.

## Jupyter Notebook

The Jupyter Notebook included in this project provides a step-by-step guide on how to preprocess the data, train the YOLO model, and evaluate its performance. Key sections of the notebook include:

- **Data Preparation**: Instructions on how to collect and annotate images for training.
- **Model Training**: Details on configuring and training the YOLO model using the annotated dataset.
- **Evaluation**: Methods to assess the accuracy and performance of the trained model.

## ammotate_video.py Script

The `ammotate_video.py` script is designed to apply the trained YOLO model to video footage. This script processes each frame of the video to detect and annotate people, vests, and hardhats. Key features of the script include:

- **Video Processing**: Efficiently handles video input and processes frames in real-time.
- **Detection and Annotation**: Utilizes the trained YOLO model to detect objects and annotate them with bounding boxes and labels.
- **Output**: Generates an annotated video highlighting detected objects, which can be used for safety audits and compliance checks.

## Conclusion

## Future Work

This project demonstrates the application of advanced computer vision techniques to enhance safety monitoring on construction sites and has the possibility to reduce yearly accidents that impact the family of these workers.

Future work could include:

- **Improving Model Accuracy**: Enhancing the YOLO model's accuracy by incorporating more diverse training data and experimenting with different model architectures.
- **Real-time Alerts**: Developing a real-time alert system to notify site managers immediately when safety violations are detected.
- **Integration with Site Management Systems**: Integrating the detection system with existing site management software for seamless monitoring and reporting.
- **Expanding Detection Capabilities**: Extending the model to detect additional safety equipment and other potential hazards on construction sites.

By continuing to refine and expand this project, we can further improve safety standards and compliance on construction sites.
