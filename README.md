# Formula 1 Ranking Model

## Overview
This Formula 1 Ranking Model leverages the XGBoost XGBRanker to predict Formula 1 race outcomes based on qualifying and telemetry data. It collects real-time and historical data with the [FastF1 API](https://github.com/theOehrly/Fast-F1) and uses Optuna for hyperparameter optimization. The model is evaluated using metrics such as Normalized Discounted Cumulative Gain (NDCG), Spearman's Rank Correlation, and Mean Reciprocal Rank (MRR). 

## Technologies Used
- Python
    - Additional libraries: pandas, NumPy, Matplotlib
- XGBoost Ranker
- FastF1 API
- Optuna
- scikit-learn
    - Used for model evaluation and train/test splitting
- rankeval
    - Used for model evaluation
- joblib
    - Used for model and metadata persistence

## Data Used
This project uses data collected from the FastF1 API, which provides access to telemetry, timing, and weather information. The dataset includes sessions from 5 Formula 1 seasons (2019-2024). The exact code used to create this dataset can be found in the src/qualifying-data.ipynb notebook. The final dataset can be found in data/data.csv

The model uses data from qualifying sessions as input features and actual race results as the target variable. Each data point in the dataset corresponds to a driver for a specific qualifying session and includes the following information:
- Weather Information
    - Average air and track temperatures
    - Humidity, wind speed, pressure, and rainfall
- Telemetry Information
    - Speed, Throttle, RPM
    - Number of DRS activations and gear changes
- Performance Information
    - Average lap times and sector times
- Session Information
    - Driver number and team name
    - Season
    - Final qualifying position (corresponds to where the driver will start the race)

As mentioned above, the target variable for the model is the final race position, which allows the model to learn how qualifying, telemetry, and weather data impacts race outcomes. 

Note: To avoid rate limits with the FastF1 API, data is collected year-by-year. Team name standardization is applied to account for team name rebranding (ex. Alpha Tauri to Racing Bulls). 

## Training the Model
The model used in this project is an **XGBoost Ranker (XGBRanker)** trained to predict the final race order based on qualifying, telemetry, and weather data. Since race outcomes are ordinal, a learning-to-rank approach was chosen to capture the relative position of drivers for each race. 

The code used to train the model can be found in the src/model.ipynb notebook. These were the main steps involved in training the model:
1. **Train-Test Split**
    - A GroupShuffleSplit was used to ensure that data from the same session was kept together in either the train or test set. This was to ensure that there would be no data leakage and the model would be able to evaluate race sessions effectively.
2. **Model Configuration**
    - The specific parameters used can be found in the code. 
    - The model is trained with the **rank:pairwise** objective, essentially comparing pairs of drivers. This specific method focuses on predicting the correct order of drivers rather than their exact finishing positions. In a use case such as Formula 1, where only the top 10 drivers (out of 20 drivers) earn points, predicting the relative position of drivers is more impactful than exact placement The training data is also well-aligned with this method as it allows the model to compare qualifying and telemetry data among drivers. 
    - The **histogram-based tree method** was used for this model as it is faster and uses less memory compared to the exact algorithm. 
3. **Hyperparameter Optimization**
    - Hyperparameters such as max_depth, learning_rate, and subsample were tuned using Optuna.
4. **Training**
    - The model was trained by grouping each qualifying session to compare drivers within the same race. 

## Metrics and Model Performance
The model was evaluated using Normalized Discounted Cumulative Gain (NDCG), Spearman's Rank Correlation, Mean Reciprocal Rank, and Top-K Precision/Recall/F1 scores. These metrics were chosen as they are widely used for ranking and recommendation systems, and are well-suited to evaluate ranking models.

**Normalized Discounted Cumulative Gain (NDCG)**
Measures the quality ranking by giving more weight to the correct placement of top drivers.
    - **Average NDCG: 0.922**
    - A score close to 1.0 indicates that the model is ranking the top drivers accurately.

**Spearman's Rank Correlation**
Evaluates how well the predicted ranking order matches the actual finishing order.
    - **Average SRC: 0.613**
    - A value above 0.6 indicates a moderate to strong correlation between the predicted and actual order. 

**Mean Reciprocal Rank (MRR)**
Calculates the average of reciprocal ranks of the winner.
    - **Mean Reciprocal Rank: 0.7**
    - A score of 0.70 means that, on average, the actual race winner was ranked highly by the model, usually among the top 2 predictions. In a Formula 1 context, this indicates a strong performance by the model. Predicting the race winner is crucial because of the significant point different between the winner and the rest of the drivers. 

**Top-K Precision / Recall / F1 Score**
Measures how well the model predicts the top drivers (Winner, Top 3, Top 10)
    - **Winner Precision/Recall/F1: 0.538**
    - **Top 3 Precision/Recall/F1: 0.675**
    - **Top 10 Precision/Recall/F1: 0.774**
    - These values indicate that the model correctly predicts the winner over 53% of the time, and correctly predicts the top 10 over 77% of the time. This level of accuracy indicates a strong performance for a multi-class ranking problem with 20 drivers and multiple external factors like weather, crashes, and driver errors.
    - Note: It is expected for the precision and recall to be equivalent as the scores are computed with a fixed Top-K set of drivers.

## Using the Model for 2025 Predictions
The model has been trained on the 2019-2024 seasons. The 2025 season brought in numerous changes including 6 new rookie drivers, changes in regulations, etc. 

In order to predict 2025 races outcomes, use the src/predictions_2025.ipynb notebook. 

## Installing
1. Clone the repository
```bash
git clone https://github.com/yourusername/job-search-chatbot.git
cd F1Project
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create dataset using [this notebook](src/qualifying-data.ipynb). Or, use the [existing dataset](data/data.csv)

4. Train the model using [this notebook](src/model.ipynb). Or, load the existing [model](src/f1_xgbranker.pkl)

5. Make predictions on the 2025 season using this [notebook](src/predictions_2025.ipynb)!


## Limitations & Future Improvements
While the model demonstrates stong performance on 2019-2024 data, there are several limitations and opportunities for future improvement.
    - **Formula 1 Uncertainty**: There is so much data that Formula 1 race engineers utilize to make predictions and inform racing strategy. Though this model uses a variety of data across qualifying, telemetry, and weather factors, there are racing circumstances that cannot be captured including crashes, driver errors, safety cars, unexpected weather changes, etc. 
    - **Changes with 2025 Season**: The 2025 Formula 1 season has introduced numerous changes in driver lineups, car performance, and regulations that were not seen in the training data from 2019-2024. This introduces additional uncertainty when making predictions on currect races.

Future Improvements:
    - Implement automated retraining after each race to continuously improve predictions and introduce 2025 data to the model.
    - Incorporate additional data from FastF1 including tire compound data, pit strategy, etc.

## License
This project is licensed under the MIT license