# AI Smart Traffic Violation Detection using YOLOv8

This project detects vehicles violating traffic signals using computer vision and deep learning. The system identifies vehicles crossing a stop line during a red signal and captures violation images automatically.

## Technologies Used

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- NumPy

## Features

- Real-time vehicle detection
- Traffic signal violation detection
- Virtual stop-line monitoring
- Automatic violation image capture

## Project Structure

traffic-violation-detection
│
├── videos
│   └── traffic.mp4
│
├── violations
│
├── detect.py
│
├── requirements.txt
│
└── README.md

## Installation

Install dependencies:

pip install ultralytics opencv-python numpy

## Run the Project

python detect.py

## Output

- Detects vehicles such as cars, trucks, buses, and motorcycles
- Identifies vehicles crossing the stop line during red signal
- Saves violation evidence images

## Applications

- Smart traffic monitoring
- Traffic law enforcement
- Intelligent transportation systems

## Author

Magesh K
B.Tech Artificial Intelligence and Data Science