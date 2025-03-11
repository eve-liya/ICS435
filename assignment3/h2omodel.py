import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import numpy as np

# Initialize H2O
h2o.init()

# Load train.csv and test.csv
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Convert to H2OFrame
train_h2o = h2o.H2OFrame(train_df)
test_h2o = h2o.H2OFrame(test_df)

# Define the target column (first column in train.csv) and convert it to categorical for binary classification
target = train_h2o.columns[0]
train_h2o[target] = train_h2o[target].asfactor()

# Set predictors (all other columns)
features = train_h2o.columns[1:]

# Run AutoML with enhanced parameters:
# - max_runtime_secs: 7200 seconds (2 hours)
# - max_models: Allow up to 100 models
# - stopping_metric: Use AUC to monitor improvement
# - stopping_tolerance: Stop if improvement is less than 0.001
# - stopping_rounds: Stop after 10 rounds without improvement
# - nfolds: 5-fold cross validation for robust estimates
# - balance_classes: If classes are imbalanced, this can help
# - keep_cross_validation_predictions: Retain CV predictions for ensembling
# - sort_metric: Sort leaderboard by AUC
# - project_name: Name for this run (useful for tracking)
aml = H2OAutoML(
    max_models=500,
    nfolds=5,
    keep_cross_validation_predictions=True,
    sort_metric="AUC",
    seed=52,
    project_name="H2O_AutoML_Enhanced"
)
aml.train(x=features, y=target, training_frame=train_h2o)

# Display the leaderboard of models
lb = aml.leaderboard
print(lb)

# Get the best model
best_model = aml.leader

# Predict on test set using the best model
preds = best_model.predict(test_h2o)
# For binary classification, the H2OFrame 'preds' contains a column "p1" for the probability of class 1.
predictions = preds.as_data_frame(use_multi_thread=True)["p1"].values

# Create an ID array (if your test set doesn’t include an ID column, adjust if needed)
ids = np.arange(len(predictions))

# Combine IDs and predictions into a submission array
submission = np.column_stack((ids, predictions))

# Save the submission file with header "Id,Predicted"
np.savetxt(fname='500h2osubmission.csv', X=submission, header='Id,Predicted', delimiter=',', comments='')

# Optionally, shut down H2O to free resources
# h2o.shutdown(prompt=False)
