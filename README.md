# Facial Recognition System utilizing Siamese Neural Networks

An interactive, real-time facial recognition (Face ID) system built with **TensorFlow / Keras** and **OpenCV**. The application implements a Siamese network architecture (one-shot learning) to perform face verification.

---

## 🚀 Features

* **Capture Face ID Dataset**: Captures positive and anchor images directly from your webcam.
* **Train Siamese Model**: Automatically processes images, sets up twin embeddings, and trains the model locally (`siamesemodel.h5`).
* **Real-Time Verification**: Compares live video feed frames to validation images and prints authorization status (`verified ✅` or `not verified ❌`).
* **Cross-Layout Keyboard Support**: Built-in support for both English (QWERTY) and Arabic keyboard layouts.
* **Safe configuration**: Excludes large model weights and personal face pictures from Git automatically.

---

## 📂 Project Structure

* `main.py`: The entry-point script containing the CLI menu interface, camera initialization, and event loops.
* `FaceID.py`: Backend logic for dataset capture, image preprocessing pipelines, model training, evaluation, and distance verification.
* `layer.py`: Custom L1 distance layer (`L1Dist`) for computing absolute difference between the twin embeddings.
* `requirements.txt`: Python packages required for the project.
* `.gitignore`: Excludes virtual environments, private face data, and the 155MB model file (which exceeds GitHub's 100MB limit).

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd FaceIDProject
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

Run the main application:
```bash
.venv\Scripts\python main.py
```

### 1. Set up Face ID (Option 1)
Select **`1`** from the menu. The camera will turn on and automatically capture:
* **100 Anchor images**
* **100 Positive images**

Then, it will train the Siamese Network on the captured dataset for 50 epochs and save the trained model as `siamesemodel.h5` in the root folder.

### 2. Real-Time Verification (Option 2)
Select **`2`** from the menu. 
* Click on the camera window to focus it.
* Press **`v`** (or **`ر`** on Arabic keyboards) to take a snapshot and verify your identity.
* Press **`q`** (or **`ض`** on Arabic keyboards, or **`ESC`**) to exit the video stream.
