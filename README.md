# ASL Digit Recognition

This project attemps to perform live digit recognition on ASL digit symbols. The project uses OpenCV to access live feed from the webcam and performs ASL digit recognition within an adjustable region of interest (ROI) on the live feed. The recognition is done with the use of a trained Convolutional Neural Network (CNN). The dataset used to train this CNN is custom made and consists of binary images of every ASL digit symbol from 0-9, inclusive.

## Getting started

### Prerequisites

1. [Python](https://www.python.org/)
2. [Pip](https://pip.pypa.io/en/stable/installation/)

#### Installing packages

1. TensorFlow
   ```
   pip install tensorflow
   ```
2. OpenCV
   ```
   pip install opencv-python
   ```
3. numpy
   ```
   pip install numpy
   ```

### Running the project

1. Clone the repo
   ```
   git clone https://github.com/hamza-mughees/Sign-Language-Detection.git
   ```
2. Navigae to the repo
   ```
   cd Sign-Language-Detection
   ```
3. Run `program.py`
   ```
   python program.py
   ```