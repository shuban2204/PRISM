# Product Price Prediction Pipeline

This project implements a multi-stage machine learning pipeline to predict product prices based on their catalog descriptions.

The pipeline consists of:
1.  **Price Preprocessing**: The target `price` variable is transformed using Box-Cox, winsorized to handle outliers, and binned into 14 ordinal classes.
2.  **RoBERTa Classifier**: A `roberta-base` model is fine-tuned to classify the `price_class` based on the `catalog_content`. The predictions from this model serve as a powerful feature for the next stage.
3.  **Feature Engineering**: Text-based features like item value, unit, and pack quantity are extracted from the `catalog_content`.
4.  **Ensemble Model**: A LightGBM regressor is trained on the RoBERTa predictions and the engineered features to predict the final price.
5.  **Evaluation**: The model's performance is measured using the Symmetric Mean Absolute Percentage Error (SMAPE) metric.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd price_prediction
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Place your data:**
    Put `train.csv` and `test.csv` inside the `data/` directory.

## How to Run

Execute the main script to run the entire pipeline from start to finish:

```bash
python -m src.main