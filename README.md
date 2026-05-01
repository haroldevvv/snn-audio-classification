# SNN Audio Classification (UrbanSound8K)

A Streamlit-based web application for environmental sound classification using a **Spiking Neural Network (SNN)** built with PyTorch and snnTorch.

This project classifies short audio clips (≤ 5 seconds) into 10 urban sound categories such as sirens, dog barks, drilling, and more.

---

## Features

-  Upload WAV audio files and get instant predictions  
-  Spiking Neural Network (SNN) for temporal audio modeling  
-  MFCC feature extraction using Librosa  
-  Spike encoding for biologically inspired processing  
-  Confidence score using softmax probabilities  
-  Deployable via Streamlit Cloud  

---

## Model Overview

The model is a **3-layer fully connected Spiking Neural Network**:

- **Input:** MFCC features (40 coefficients)  
- **Hidden Layers:**
  - FC (256) + Leaky LIF  
  - FC (128) + Leaky LIF  
- **Output:**
  - FC (10 classes) + Leaky LIF  
- Output spikes are summed across time for classification  

---

## Dataset

- **UrbanSound8K**
- 10 sound classes:
  - Air Conditioner  
  - Car Horn  
  - Children Playing  
  - Dog Bark  
  - Drilling  
  - Engine Idling  
  - Gunshot  
  - Jackhammer  
  - Siren  
  - Street Music  

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/haroldevvv/snn-audio-classification.git
cd snn-audio-classification

### 2. Install dependencies
```bash
pip install -r requirements.txt

### 3. Run locally
```bash
streamlit run app.py

## Developer
Harold Salvador


