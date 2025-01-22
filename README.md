# Resistor Sorter

A computer vision system that automatically identifies and sorts resistors based on their color bands. The system uses a Raspberry Pi with camera to capture images of resistors, processes them to identify the color bands, calculates the resistance value, and can sort them into bins.

![https://www.youtube.com/watch?v=I8Fsi_Wz-K8](rsort.mp4)

## Components

- Image capture system using PiCamera2
- Computer vision pipeline for resistor detection and color band identification
- Color classification system using RGB color space transformation
- Servo-controlled sorting mechanism
- Training/evaluation tools for improving color detection

## Key Files

- `capture.py` - Controls the camera and captures resistor images
- `identify.py` - Core computer vision pipeline for detecting resistors and their color bands
- `eval.py` - Tools for evaluating and improving the color detection system
- `utils.py` - Shared utilities and helper functions
- `run.py` - Main script that ties everything together for automated sorting

## How It Works

1. The camera continuously captures images, showing a preview with targeting overlay
2. When a resistor is detected:
   - Finds the endpoints of the resistor body
   - Isolates and normalizes the resistor image
   - Detects color band positions
   - Classifies the colors using a trained color space transformation
   - Calculates the resistance value
3. If valid resistance is detected, activates servo system to sort into appropriate bin

## Setup

Requires:
- Raspberry Pi with camera module
- Servo motors for sorting mechanism
- Python 3.7+
- OpenCV
- NumPy
- picamera2

## Usage

1. Training/improving color detection:
```
python eval.py
```

2. Capturing new training images:
```
python capture.py
```

3. Running the automated sorter:
```
python run.py
```

## Color Codes

The system recognizes the standard resistor color codes:
- Black (0)
- Brown (1) 
- Red (2)
- Orange (3)
- Yellow (4)
- Green (5)
- Blue (6)
- Purple (7)
- Gray (8)
- White (9)
- Gold (-1, tolerance)
- Silver (-2, tolerance)
