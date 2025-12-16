# Inspirit AI Machine Learning

## Topics Covered

Through Inspirit AI, I explored both theoretical foundations and hands-on implementations of machine learning:

* Linear Regression
* Logistic Regression
* Natural Language Processing

  * One-Hot Encoding
  * Bag of Words
* Word Embeddings
* Neural Networks
* Convolutional Neural Networks (CNNs)

---

## Real-Time Object Detection (YOLOv3)

The primary project in this repository is a **real-time object detection pipeline** developed in late June–early July 2022.

### Highlights

* Live webcam inference
* YOLOv3 architecture implemented with **TensorFlow / Keras**
* Manual implementation of:

  * Bounding box decoding
  * Confidence thresholding
  * Non-Maximum Suppression (NMS)
* GPU acceleration using CUDA

---

## Model Weights (YOLOv3)

This project uses **YOLOv3 pre-trained weights** stored as a Keras `.h5` file.

> ⚠️ **Note:** The `yolo.h5` weights file is **not included** in this repository.

### Obtaining Weights

Official YOLOv3 weights can be accessed from the **YOLO / Darknet website**. These weights can then be converted to Keras (`.h5`) format using widely available conversion scripts.

Once obtained, place the file in your project directory:

```
yolo.h5
```

And update the model loading path in the code if necessary:

```python
darknet = tf.keras.models.load_model("path/to/yolo.h5")
```

---

## Environment Setup (Recommended: Miniconda)

I **strongly recommend using Miniconda** to manage your Python environment, especially for **GPU optimization**.

### Why Miniconda?

* Lightweight compared to Anaconda
* Cleaner dependency resolution for CUDA + cuDNN
* Easier control over Python and TensorFlow versions
* Reduced conflicts when enabling GPU acceleration

### Example Setup

```bash
conda create -n yolo-gpu python=3.9.2
conda activate yolo-gpu
conda install cudatoolkit=11.2 cudnn=8.1 -c conda-forge
pip install tensorflow numpy opencv-python pillow
```

This setup closely matches the environment used during development and helps ensure **stable GPU detection** by TensorFlow.

---

## System Requirements (GPU)

To run this project with GPU support:

* **Python:** 3.9.2
* **CUDA Toolkit:** 11.2
* **cuDNN:** 8.1.0
* **TensorFlow (GPU-compatible)**
* NVIDIA GPU

GPU availability is verified in code using:

```python
tf.test.gpu_device_name()
```

---

## Running the Program

```bash
python object_detection.py
```

* The webcam will activate and perform real-time object detection
* Press **ESC** to exit

---

## Dataset & Labels

* Uses the **COCO dataset** label set (80 object classes)
* Class probabilities and bounding boxes are decoded manually from YOLO outputs

---

## Recognition

* ⭐ **Most liked project** at the YoungWonks CCI Fair (Summer 2022)
* Developed as part of Inspirit AI’s CNN & Computer Vision curriculum

---

## License

This repository is shared for **educational and non-commercial purposes**.

YOLO, COCO, and related assets remain under their respective licenses.

---

## Credits

* **Inspirit AI** — instruction and curriculum
* **Joseph Redmon et al.** — YOLOv3 architecture
* TensorFlow, Keras, OpenCV open-source communities

---

## Author

**Tristan Ng**
Machine Learning • Computer Vision • AI Research
