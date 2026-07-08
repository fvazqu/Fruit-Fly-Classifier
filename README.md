# Fruit Fly Maturity Classifier 🪰

An automated computer vision system developed to classify and count fruit fly pupae eye color maturity (Mature vs. Immature). This project utilizes the **YOLOv8** architecture and a **Streamlit** web interface to provide a localized, research-ready tool for entomological analysis.

## 📋 Overview
This project provides a "Human-in-the-loop" tool for researchers to automate the labor-intensive process of eye-color classification. The system detects individual pupae in high-density images, classifies their developmental stage based on eye pigmentation, and logs the results for historical analysis.

### Key Features
*   **High-Accuracy Detection:** Optimized for multi-specimen images (20+ flies per image) using head-centric localization.
*   **Persistent Storage:** Local SQLite database integration to track historical counts and trends.
*   **Real-time Visualization:** Instant bounding box overlays with class predictions.
*   **Data Export:** Capability to download detection history as CSV for external reporting and statistical analysis.

## 🚀 Getting Started

### 1. Prerequisites
*   Python 3.9+
*   NVIDIA GPU (recommended for faster inference)
*   Streamlit

### 2. Installation
```bash
# Clone the repository
git clone [https://github.com/your-username/fruit-fly-classifier.git](https://github.com/your-username/usda-fly-classifier.git)
cd fruit-fly-classifier

# Install dependencies
pip install -r requirements.txt
