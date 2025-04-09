# AI-Driven Stock Price Forecasting with Sentiment Analysis

This project integrates deep learning and sentiment analysis to forecast stock prices more effectively. By combining historical stock data with financial news sentiment, it delivers context-aware predictions using ARIMA, LSTM, and XGBoost models. A user-friendly Flask web interface allows users to interact with the system, view sentiment insights, and visualize predicted stock price trends.

---

## Key Features

- Real-time stock data fetched using `yfinance` API
- News articles fetched via News API and sentiment analyzed using VADER
- Forecasts generated using ARIMA, LSTM, and XGBoost
- Visualization of actual vs predicted prices
- Web-based dashboard for user interaction and insights

---

## Technology Stack

| Component           | Technology                      |
|--------------------|----------------------------------|
| Frontend           | HTML, CSS, JavaScript            |
| Backend            | Python, Flask                    |
| Machine Learning   | TensorFlow/Keras, XGBoost        |
| Sentiment Analysis | VADER, NLTK                      |
| Time Series        | statsmodels (ARIMA)              |
| Data Sources       | yfinance, News API               |
| Deployment         | Flask (local or hosted)          |

---

## Project Structure

```
.
├── app/                  # Flask app with routes and templates
├── models/               # Trained models and architecture
├── utils/                # Data handling, preprocessing scripts
├── tests/                # Evaluation and test scripts
├── requirements.txt      # Project dependencies
├── .gitignore
└── README.md
```

---

## Hardware and Software Requirements

**Hardware:**
- Processor: Intel i5/i7 or equivalent
- RAM: 8 GB or higher
- Storage: 100 GB HDD/SSD
- GPU: Optional (NVIDIA GPU with CUDA for LSTM training)

**Software:**
- Python 3.7 or higher
- IDE: VS Code / Jupyter Notebook / Colab

**Libraries:**
- Data: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Modeling: scikit-learn, TensorFlow, XGBoost, statsmodels
- NLP: NLTK, VADER, TextBlob
- APIs: yfinance, News API
- Deployment: Flask

---

## Setup Instructions

1. Clone the repository:
```bash
git clone git@github.com:Sarada-Sudarshan/ai-driven-stock-forecasting.git
cd ai-driven-stock-forecasting
```
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the application:
```bash
app.py run
```
Visit `http://127.0.0.1:5000/` in your browser.

---

## Future Enhancements

- Integration of transformer-based models (e.g., BERT, GPT)
- Real-time stock sentiment stream from social media
- Cloud deployment on platforms like AWS or Heroku
- Addition of portfolio management module
