# Brain Tumor Classification and Care

## Project Overview
This project focuses on two main areas:

1. **Brain Tumor Classification**: Using machine learning models to classify brain tumors based on MRI images.
2. **Care by Chatbot**: Providing support to patients during recovery and healing through an AI-powered chatbot.

---

## Features

### 1. Brain Tumor Classification
- Utilizes two models:
  - **PyTorch Model**: Pretrained ResNet50 architecture.
  - **TensorFlow Model**: Custom-trained EfficientNetB0 architecture.
- Both models are deployed in a web interface where users can upload MRI images to receive predictions from both models.
- The TensorFlow model shows higher accuracy compared to the PyTorch model.

#### Deployment Workflow
- Combines both models in a unified deployment file for evaluation.
- Results are displayed through a website built using Flask.

### 2. Care by Chatbot
- AI chatbot built with **Llama 3** for assisting patients during the recovery phase.
- Uses the **Llama3:8b-instruct-q3_K_L** model, which can run locally without requiring internet connectivity.
- To enhance performance on older devices, the chatbot utilizes:
  - **Open-WebUI**: An interface for running local models.
  - **Groq API**: Accelerates processing through external API calls.

---

## Installation and Setup

### Prerequisites
- Python 3.11
- Docker Desktop (optional for Open-WebUI installation)

### Steps to Install Dependencies
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Install the required Python modules:
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset and Models
- **Dataset**: To install the dataset, go to this link:
  [Brain Tumor Classification MRI Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri?)
- **Models**:
  - **First Model (TensorFlow)**: [Download TensorFlow Model](https://drive.google.com/file/d/1nCbXHx2LMmgRByJ8mG7OtPP0ZsSJUje-/view?usp=sharing)
  - **Second Model (PyTorch)**: [Download PyTorch Model](https://drive.google.com/file/d/1LbFYQWl-gsi9tKo6SDNIwRTd0yi88GAJ/view?usp=sharing)

After downloading the models, place them into the `assets` folder in the project directory to ensure they can be accessed by the deployment script.

---

## How to Run the Project
1. **Install Dependencies**: Ensure all required modules are installed using:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Deployment Script**: Start the deployment server by running:
   ```bash
   python deploy.py
   ```
3. **Start Open-WebUI**: In a terminal, start the Open-WebUI interface using:
   ```bash
   open-webui serve
   ```
4. **Launch the Website**:
   - Navigate to the `Home Page` folder in the project directory.
   - Open `index.html` in your browser.
5. **Interact with the Website**:
   - Use the classification section to upload MRI images and view predictions.
   - Access the chatbot section for patient care support.

---

## Setting Up Open-WebUI (Optional for Chatbot)
#### Method 1: Using Docker
```bash
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway   -v open-webui:/app/backend/data --name open-webui --restart always   ghcr.io/open-webui/open-webui:main
```
#### Method 2: Using Python
```bash
pip install open-webui
open-webui serve
```

### Using Groq API for Improved Performance
1. Create an API key at [Groq Console](https://console.groq.com/keys).
2. Configure Open-WebUI:
   - Go to **Settings > Admin Settings > Connections**.
   - Add a new API connection with:
     - **API Base URL**: `https://api.groq.com/openai/v1`
     - **API Key**: Your generated key.
   - Save the configuration.

---

## Project Structure
- `deploy.py`: Deployment script for model evaluation and integration.
- `KAN.py`: Implements a training methodology using the Kolmogorov-Arnold Network (KAN), leveraging B-spline basis functions and a KAN layer for neural network optimization. This approach facilitates the approximation of multivariate functions with reduced computational complexity, making it particularly useful for scenarios requiring efficient modeling.
- `TFModel_Training.py`: TensorFlow model training script.
- `requirements.txt`: List of required Python modules.
- `Home Page/index.html`: Main web page to access chatbot and classification functionalities.

---

## Notes
- Ensure all dependencies are installed before running the scripts.
- For more details on Open-WebUI, refer to their [official documentation](https://docs.openwebui.com/).
- For older devices, using the Groq API is highly recommended to improve response times.

---

## License
This project is open-source and available under the MIT License.

## Acknowledgments
- PyTorch and TensorFlow communities for providing robust frameworks.
- Open-WebUI and Groq teams for their incredible tools.
- Patients and medical professionals who inspired the care chatbot feature.
