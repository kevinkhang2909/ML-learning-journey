SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model.
It connects optimal credit allocation with local explanations using the classic Shapley values from game theory 
and their related extensions

- SHAP is a method for explaining individual predictions and answers the question of how much does each feature 
contribute to this prediction
- SHAP values are measures of feature importance
- SHAP values can be negative and positive and show the magnitude of prediction relative to the average of all 
predictions. The absolute magnitude indicates the strength of the feature for a particular individual prediction
- The average of absolute magnitudes of SHAP values per feature indicates the global importance of the feature