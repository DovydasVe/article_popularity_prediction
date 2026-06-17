# Online News Popularity Predictor

With the ever-increasing availability of media across different sources and formats, the battle for consumer attention is at its peak. Online readers typically dedicate only a few seconds to evaluate content before moving elsewhere. In this project, we implement machine learning models to predict the most popular news articles *before publication* so media outlets can strategically place their best-performing content on their front pages or recommend them to maximize interactions and ad revenue.

We utilize the **Mashable Online News Popularity Dataset** from the UCI Machine Learning Repository, applying three different algorithms under four preprocessing strategies (12 distinct pipelines) and evaluate them using a customized business-oriented scoring metric.

---

## Methodology & Strategy

To identify "super-popular" articles with high click-through potential, we define the target class as the **top 10% of articles** by share count (creating a 90/10 class imbalance). We train **three machine learning algorithms**:
1. **XGBoost** (Extreme Gradient Boosting)
2. **Random Forest**
3. **CatBoost** (Categorical Boosting)

Each algorithm is trained under **four distinct conditions**:
*   **Baseline Classification:** Standard training on the imbalanced 90/10 target.
*   **Undersampling:** Class balancing using random undersampling of the majority class.
*   **SMOTE:** Class balancing via Synthetic Minority Over-sampling Technique.
*   **Regression:** Training a regressor to predict the raw share count, then ranking predictions to select the top-performing articles.

---

## Evaluation Metrics

To align validation with real-life news curation, we developed two custom evaluation strategies:
1.  **Custom Cross-Validation (CV) Score:** Evaluates the top-k precision, where \(k\) is the number of actual popular articles in the validation set.
2.  **Daily Simulation Test set:** Simulates 100 days of news publication, selecting the **top 3 most confident predictions** each day to output:
    *   **Precision@3:** The average percentage of correctly identified popular articles.
    *   **At Least One Metric:** The percentage of days (out of 100) where the model successfully selected at least one popular article in its top 3.

---

## Key Results

| Model | Pipeline / Strategy | Best CV Score | Test Precision | At Least One Metric (Days / 100) |
| :--- | :--- | :---: | :---: | :---: |
| **XGBoost** | **Baseline Classification** | 0.262 | **0.280** (Best) | **63** (Best) |
| **XGBoost** | Undersampling | 0.255 | 0.247 | 57 |
| **XGBoost** | SMOTE | 0.249 | 0.237 | 57 |
| **XGBoost** | Regression | 0.243 | 0.240 | 56 |
| **CatBoost** | **Baseline Classification** | **0.268** (Best) | 0.250 | 62 |
| **CatBoost** | Undersampling | 0.264 | 0.270 | 59 |
| **CatBoost** | SMOTE | 0.259 | 0.237 | 57 |
| **CatBoost** | Regression | 0.251 | 0.260 | 61 |
| **Random Forest**| Baseline Classification | 0.261 | 0.230 | 58 |
| **Random Forest**| Undersampling | 0.261 | 0.247 | 58 |
| **Random Forest**| SMOTE | 0.255 | 0.220 | 56 |
| **Random Forest**| Regression | 0.244 | 0.240 | 56 |

### Main Takeaways:
*   **Top Performer:** The **XGBoost Baseline Classification** model achieved the highest test precision (**0.280**), meaning 2.8 out of 10 articles it curates are popular. It placed at least one popular article on the front page in **63 out of 100 days**.
*   **Ensemble Robustness:** Resampling techniques (Undersampling & SMOTE) did not yield significant improvements and sometimes reduced model performance, indicating that tree-based ensemble models natively handle class imbalance well and benefit from the complete feature distributions of the original dataset.
*   **Classification vs. Regression:** Classification models consistently (but marginally) outperformed regression models at curating top articles.

---

## Project Structure

*   `data/`: Contains the Mashable Online News Popularity dataset (`OnlineNewsPopularity.csv` and metadata).
*   `data_preprocessing.ipynb`: Exploratory data analysis, handling feature collinearity, data cleaning, and class definitions.
*   `pipeline.ipynb`: Machine learning pipelines, hyperparameter grid search, model evaluation, test simulation, and performance visualization.
*   `utils.py`: Helper functions for making top-3 selections, evaluating prediction hit lists, and visualizing grid search parameter grids.
*   `Report.pdf`: The final academic paper detailing methodology, theoretical background, and extensive result analysis.

---

## Setup & Requirements

To run the pipeline and preprocessing notebooks, you will need Python 3.8+ and the following packages:
```bash
pip install numpy pandas scikit-learn xgboost catboost imbalanced-learn matplotlib seaborn
```
