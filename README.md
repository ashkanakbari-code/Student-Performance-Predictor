# Student Performance Predictor (PyTorch)

A deep learning project built with **PyTorch** to predict student performance from input features using a custom multi-layer feedforward neural network.

## 🚀 Features

- **Data Preprocessing:** Scales features using `StandardScaler` and splits the dataset into training and testing sets with `train_test_split`.
- **Neural Network Architecture:** Implements a deep feedforward neural network using `torch.nn.Module` and `torch.nn.Sequential`, with ReLU activation functions.
- **Optimization:** Trains the model with the Adam optimizer and Mean Squared Error (`MSELoss`).
- **Loss Visualization:** Tracks and plots training loss across epochs using Matplotlib.
- **Evaluation:** Evaluates the trained model on unseen test data and compares sample predictions with actual values.

## 📂 Project Structure

```text
Student-Performance-Predictor/
├── data/               # Dataset files
├── .venv/              # Python virtual environment (not committed)
├── main.py             # Training, evaluation, and plotting
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## 📦 Requirements & Installation

Make sure Python is installed. It is recommended to use a virtual environment.

### 1. Clone the repository

```bash
git clone <repository-url>
cd Student-Performance-Predictor
```

Replace `<repository-url>` with your repository's URL.

### 2. Create and activate a virtual environment (optional)

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 🏃‍♂️ How to Run

Run the main script to train the neural network, display the training loss curve, and evaluate performance on unseen test data:

```bash
python main.py
```

## 🛠️ Built With

- [Python](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)
- [pandas](https://pandas.pydata.org/) (if used for loading or handling the dataset)

## 📝 Notes

- Ensure the dataset files are placed in the expected location inside the `data/` directory.
- The exact dataset path, input features, target variable, and model configuration depend on the implementation in `main.py`.
- Do not commit the `.venv/` directory or other generated files. Add them to `.gitignore`.
- Model performance depends on the dataset, preprocessing, and training configuration.
