# 📈 Stock Price Forecasting Project

This project provides two main approaches for forecasting stock prices:

1. **CatBoost-based stock price forecasting** (`catboost_forecast.ipynb`)
2. **Fine-tuned LLM model for stock forecasting** (`model_test.ipynb`)

---

## 🚀 Getting Started

### **🔹 Install Dependencies**
Before running any notebook, install the required dependencies using:

```bash
pip install -r requirements.txt
```

---

## 📊 **1️⃣ CatBoost Stock Price Forecasting**
**File:** `catboost_forecast.ipynb`

This notebook trains a **CatBoost regression model** using historical stock data and engineered features (such as SMA, RSI, and MFI) to predict future stock prices.

### **How to Use**
1. Open `catboost_forecast.ipynb` in Jupyter Notebook or VS Code.
2. Ensure you have the dataset files:
   - `stock_data_train.json`
   - `stock_data_test.json`
3. Run all cells in the notebook to train and evaluate the CatBoost model.

### **Expected Output**
- The model will provide **predictions** for stock prices.
- Evaluation metrics such as **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)** will be displayed.
- The notebook will generate **dynamic visualizations** comparing actual vs. predicted prices.

---

## 🤖 **2️⃣ Fine-tuned LLM Model for Stock Forecasting**
**File:** `model_test.ipynb`

This notebook utilizes a **fine-tuned large language model (LLM)** to analyze stock trends and generate stock price predictions.

### **How to Use**
1. Open `model_test.ipynb` in Jupyter Notebook or VS Code.
2. Load the fine-tuned LLM model.
3. Provide historical stock data (`stock_data_finetune.json`).
4. Run all cells to generate stock price predictions.

### **Run LLM Model for Stock Price Forecasting**
To use the fine-tuned LLM model for stock prediction, run the following code inside `model_test.ipynb`:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

input_text = "User: Here is stock data for AAPL. Last 10 days price changes: [2.14, -0.12, -1.2, 0.0, 0.4, 1.38, -1.41, 1.34, 0.11, 0.32]%. Volume changes: [-22.77, -9.04, 9.59, -3.83, 20.22, -7.5, 6.68, -6.76, -18.96, -2.88]%. RSI values: [37.07, 33.37, 31.02, 37.37, 39.83, 44.72, 35.98, 41.52, 47.25, 56.67]. MFI values: [38.38, 38.46, 38.38, 45.79, 52.28, 52.7, 46.04, 52.35, 59.01, 61.82]. Recent news: [{'headline': 'Is There Now An Opportunity In Apple Inc. (NASDAQ:AAPL)?', 'days_ago': 9, 'snippet': 'Nov 11, 2024 · Apple Inc. (NASDAQ:AAPL) received a lot of attention from a substantial price movement on the NASDAQGS over the last few months, increasing to US$236 at one point, …'}, {'headline': 'Analysts Upgrade Apple Stock with Bullish Targets, Pushing Price ...', 'days_ago': 7, 'snippet': 'Nov 13, 2024 · This week, Apple Inc. (AAPL, Financial) gained confidence from analysts, who revised their recommendations and price targets for the stock, suggesting that the Street is …'}, {'headline': 'Bank of America Reiterates Buy on Apple Inc. (AAPL), Citing …', 'days_ago': 6, 'snippet': 'Nov 14, 2024 · In this article, we are going to take a look at where Apple Inc. (NASDAQ:AAPL) stands against the other AI stocks that are making waves.'}, {'headline': 'Barclays Highlights Key Semiconductor Trends Including Nvidia …', 'days_ago': 2, 'snippet': "Nov 18, 2024 · Barclays analysts gave a review of the chip industry and talked about Nvidia's (NVDA, Financials) Blackwell ramp, Apple's (AAPL, Financials) plans to cut iPhone …"}]. What is the expected price change in 3 days?"

inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### **Expected Output**
- The model will analyze historical stock data (price changes, RSI, MFI, volume trends, and news headlines).
- It will generate a **stock price prediction** for the next 3 days based on this data.

---

## 📂 **Project Files**
| File | Description |
|------|------------|
| `catboost_forecast.ipynb` | CatBoost-based stock price prediction model |
| `model_test.ipynb` | Fine-tuned LLM model for stock trend forecasting |
| `train_data_generator.ipynb` | Generates synthetic training data for experiments |
| `stock_data_train.json` | Training dataset for CatBoost |
| `stock_data_test.json` | Test dataset for CatBoost |
| `stock_data_finetune.json` | Dataset used for fine-tuning the LLM |
| `requirements.txt` | List of required dependencies |


---

## 📜 **License**
This project is licensed under the **MIT License**.

