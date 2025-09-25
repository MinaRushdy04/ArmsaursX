# 🤖 ArmsaurusX

**ArmsaurusX** is a cyberpunk-inspired robotic arm project — blending computer vision, inverse kinematics, and real-time control.  
The goal is to showcase advanced **software-driven intelligence** on top of modest mechanical hardware.

---

## 🚀 Features

- **3DOF Robotic Arm (RRR Manipulator)**
  - Forward & Inverse Kinematics implementation
  - Custom servo angle limits and safety constraints
  - Real-time position feedback and control

- **Computer Vision Integration**
  - YOLOv8 object detection
  - Real-time bounding boxes, labels, and confidence scores
  - Camera calibration and world coordinate mapping

- **Control Modules**
  - 🎮 Manual joystick control  
  - 🎥 Camera-guided automatic control
  - 👋 Gesture Control using MediaPipe
  - 🔗 Arduino integration (serial communication)  


- **Visualization**
  - 3D simulation of the arm with Matplotlib
  - Object detection visualization with confidence metrics

---

## 🛠️ Technologies Used
### **Python Backend & Vision System**
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** - Custom object detection model training
- **OpenCV** - Computer vision, camera calibration, and image processing
- **NumPy** - Mathematical computations and matrix operations
- **Matplotlib** - 3D visualization and robotic arm simulation
- **PySerial** - Arduino communication and servo control
- **Firebase Admin SDK** - Real-time database integration
### **Android Mobile Application**
- **Kotlin** - Primary programming language for real-time performance
- **Jetpack Compose** - Modern declarative UI framework
- **MediaPipe** - Real-time hand gesture recognition
### **Cloud & Real-Time Systems**
- **Firebase Realtime Database** - Cross-platform data synchronization


---

## 🌐 Firebase Database Structure
````
{
"servo_control": {
"servo1": 90, "servo2": 35, "servo3": 60, "servo4": 35,
"timestamp": 1694087025000
},
"robot_commands": {
"command_id": {
"gestureType": "Thumb_Up", "confidence": 0.89,
"robotCommand": "SERVO3_INCREMENT", "timestamp": 1694087030000}
}
````

---

## 👨‍💻 Author

- [**Omar Mostafa**](https://www.linkedin.com/in/omar-mostafa-227a10288/) - Remote Control (Mobile App + Firebase)
- [**Sama Mohamed**](https://www.linkedin.com/in/sama-mohammed-503915364/) - Manual Control (Joysticks)
- [**Mina Rushdy Rady**](https://www.linkedin.com/in/mina-rushdy-rady-73434725a/) - Inverse Kinematics Implementation
- [**Abdallah Fathy**](https://www.linkedin.com/in/abdallah-fathy-b800402a4/) - Mechanical Design & Fabrication
- [**Sandy Alaa**](https://www.linkedin.com/in/sandy-alaa-736b06252/) - Handled YOLOv8 model & Camera Calibration



