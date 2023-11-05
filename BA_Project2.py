import pandas as pd
import numpy as np
from sklearn import preprocessing
df = pd.read_csv('complaints_25Nov21.csv')
df.head()
df.describe()
df.info()

# Convert categorical variables
le = preprocessing.LabelEncoder()
X = df[['Product', 'Sub-product', 'Issue', 'State', 'Tags', 'Submitted via', 'Company response to consumer', 'Timely response?']].apply(le.fit_transform)
y = le.fit_transform(df['Consumer disputed?'])

from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

from imblearn.under_sampling import RandomUnderSampler

# Check the proportion and balance if necessary
if y_train.mean() < 0.3:
    undersampler = RandomUnderSampler(random_state=123)
    X_train, y_train = undersampler.fit_resample(X_train, y_train)

from xgboost import XGBClassifier

# Train the model
model_xgb = XGBClassifier(random_state=123)
model_xgb.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

# Predictions
y_pred = model_xgb.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Base-case cost calculation
base_case_cost = y_test.sum() * 600 + (len(y_test) - y_test.sum()) * 100

# Model cost calculation using confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
model_cost = tn * 100 + fp * (100 + 90) + fn * 600 + tp * (100 + 90)

# Adjust classification threshold to minimize cost
thresholds = np.linspace(0, 1, 100)
for threshold in thresholds:
    y_pred_threshold = (model_xgb.predict_proba(X_test)[:, 1] > threshold).astype(int)
    # Recalculate costs based on new predictions

# Quetion 1

# Calculate the number of disputes
num_disputes = sum(y_test)

# Calculate the total number of cases in the test set
total_cases = len(y_test)

# Calculate the proportion of disputes
proportion_disputes = num_disputes / total_cases

proportion_disputes

# Quetion 2

from imblearn.under_sampling import RandomUnderSampler

undersampler = RandomUnderSampler(random_state=123)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

# Calculate the number of disputes
num_disputes = sum(y_train_resampled)

# Calculate the total number of cases in the resampled train set
total_cases_resampled = len(y_train_resampled)

# Calculate the proportion of disputes
proportion_disputes_resampled = num_disputes / total_cases_resampled

proportion_disputes_resampled

# Quetion 3

undersampler = RandomUnderSampler(random_state=123)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

# Train the XGBClassifier model
model_xgb = XGBClassifier(random_state=123)
model_xgb.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = model_xgb.predict(X_test)

# Evaluate the model
report = classification_report(y_test, y_pred, output_dict=True)
recall_disputed_yes = report['1']['recall']  # Assuming '1' corresponds to 'Consumer disputed?' = 'Yes'

recall_disputed_yes

# Quetion 4

# Calculate the base cost for all complaints
base_cost = len(y_test) * 100

# Calculate the additional cost for disputed complaints
# Assuming '1' represents 'Yes' (disputed) in y_test
additional_cost_disputed = sum(y_test) * 500

# Calculate the total cost
total_cost = base_cost + additional_cost_disputed

total_cost

# Quetion 5

# Define the costs
cost_base = 100  # Base cost for resolving a complaint
cost_extra_diligence = 90  # Cost for extra diligence if a dispute is predicted
cost_dispute = 500  # Additional cost if a dispute occurs

# Calculate the total cost
total_cost = ((tn + fp + fn + tp) * cost_base) + ((tp + fp) * cost_extra_diligence) + (fn * cost_dispute)


total_cost

# Quetion 6

y_probs = model_xgb.predict_proba(X_test)[:, 1]  # get the probability of the positive class

# Define the costs
cost_base = 100
cost_extra_diligence = 90
cost_dispute = 500

# Initialize the minimum cost to a large number and the best threshold to None
min_total_cost = float('inf')
best_threshold = None

# Iterate over a range of possible thresholds
for threshold in np.linspace(0, 1, 101):
    # Apply the threshold to the predicted probabilities to create a binary prediction
    y_pred_threshold = (y_probs >= threshold).astype(int)

    # Calculate the confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()

    # Calculate the total cost for this threshold
    total_cost = ((tn + fp + fn + tp) * cost_base) + (fp * cost_extra_diligence) + (fn * cost_dispute)

    # If this total cost is lower than the minimum cost, update the minimum cost and best threshold
    if total_cost < min_total_cost:
        min_total_cost = total_cost
        best_threshold = threshold
min_total_cost

# Quetion 7

y_probs = model_xgb.predict_proba(X_test)[:, 1]

# Define the costs
cost_base = 100  # Base cost for resolving a complaint
cost_extra_diligence = 90  # Cost for extra diligence if a dispute is predicted
cost_dispute = 500  # Additional cost if a dispute occurs

# Initialize the minimum cost to a large number and the best threshold to None
min_total_cost = float('inf')
best_threshold = None

# Evaluate thresholds from 0 to 1 in increments of 0.01
for threshold in np.linspace(0, 1, 101):
    # Apply the threshold to the predicted probabilities to create binary predictions
    y_pred_threshold = (y_probs >= threshold).astype(int)

    # Calculate the confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()

    # Calculate the total cost for this threshold
    total_cost = (tn + fp) * cost_base + (fp) * cost_extra_diligence + (fn) * cost_dispute

    # If this total cost is lower than the minimum cost, update the minimum cost and best threshold
    if total_cost < min_total_cost:
        min_total_cost = total_cost
        best_threshold = threshold

# Output the best threshold
best_threshold































