import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa
import snntorch as snn


sample_rate = 16000
n_mfcc = 40
max_len = 120   
device = "cpu"

CLASS_NAMES = {
    "air_conditioner": "Air Conditioner Noise",
    "car_horn": "Car Horn",
    "children_playing": "Children Playing",
    "dog_bark": "Dog Barking",
    "drilling": "Drilling Sound",
    "engine_idling": "Engine Idling",
    "gun_shot": "Gunshot",
    "jackhammer": "Jackhammer",
    "siren": "Emergency Siren",
    "street_music": "Street Music"
}

CLASS_LABELS = list(CLASS_NAMES.keys())


class SNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(n_mfcc, 256)
        self.lif1 = snn.Leaky(beta=0.9)

        self.fc2 = nn.Linear(256, 128)
        self.lif2 = snn.Leaky(beta=0.9)

        self.fc3 = nn.Linear(128, 10)
        self.lif3 = snn.Leaky(beta=0.9)

    def forward(self, x):
        x = x.permute(2, 0, 1)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk3_rec = []

        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)

        return torch.stack(spk3_rec)


@st.cache_resource
def load_model():
    model = SNNModel().to(device)
    model.load_state_dict(torch.load("models/snn_urbansound8k.pth", map_location=device))
    model.eval()
    return model


model = load_model()


def extract_mfcc(file):
    y, sr = librosa.load(file, sr=sample_rate)

    y = librosa.util.fix_length(y, size=sample_rate * 5)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_len:
        pad = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)))
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc


def spike_encode(x):
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)

    n_features, time_steps = x.shape
    spike_times = (1 - x) * (time_steps - 1)
    spike_times = spike_times.astype(int)

    spikes = np.zeros((n_features, time_steps), dtype=np.float32)

    for i in range(n_features):
        for j in range(time_steps):
            t = spike_times[i, j]
            spikes[i, t] = 1.0

    return spikes


def predict(file):
    mfcc = extract_mfcc(file)
    spikes = spike_encode(mfcc)

    spikes = torch.tensor(spikes).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(spikes)
        output = output.sum(0)

        probs = torch.softmax(output, dim=1)
        pred = probs.argmax(1).item()
        confidence = probs[0][pred].item()

    label = CLASS_LABELS[pred]
    display = CLASS_NAMES[label]

    return display, confidence


st.title("🔊 SNN Audio Classification")

uploaded_file = st.file_uploader("Upload a WAV file (≤ 5 sec)", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Predict"):
        display, confidence = predict(uploaded_file)

        st.success(f"Predicted Sound: {display}")
        st.info(f"Confidence: {confidence:.2%}")