# Predict Fuel Efficiency

This project aims to predict the fuel efficiency (miles per gallon) of vehicles using regression models. The dataset used is the Auto MPG dataset from the UCI Machine Learning Repository.

## Dataset
The dataset contains various attributes of vehicles, such as cylinders, displacement, horsepower, weight, acceleration, model year, and origin. The 'mpg' column represents the target variable.

## Data Preprocessing
- Download the dataset from the UCI repository and load it using pandas' `read_csv()` function.
- Handled missing values in the 'horsepower' column by replacing them with the median value.
- Created new features: 'displacement_per_cylinder' and 'power_to_weight_ratio'.

## Data Augmentation
- Implemented the `augment_data()` function to generate synthetic data by interpolating values between instances with the same 'mpg' value and similar 'cylinders'.
- Added noise to the mean values of attributes and rounded them to a reasonable precision.
- Calculated additional synthetic features: 'power_to_weight_ratio' and 'displacement_per_cylinder'.

## Model Building and Evaluation
- Utilized Linear Regression, Ridge Regression, Lasso Regression, Random Forest Regressor, and Gradient Boosting Regressor models.
- Split the combined data into training and testing sets.
- Evaluated the models' performance using the R-squared metric.
- Performed hyperparameter tuning for the Gradient Boosting Regressor using RandomizedSearchCV.

## Normalization
- Applied standard scaling to selected columns using StandardScaler to normalize the data.

## Results
- Provided the train and test accuracy results for each model in a tabular format.
- Compared the performance of Random Forest Regressor and Gradient Boosting Regressor with Random Forest as the base estimator.
