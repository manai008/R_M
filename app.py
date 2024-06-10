import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm

st.title("Online Linear Regression Calculator")

# Input for number of predictors
num_predictors = st.number_input("Number of Predictors", min_value=1, value=1)

# Generate input fields for predictors
predictors = []
for i in range(num_predictors):
    predictor_values = st.text_input(f"X{i+1} Values (comma-separated)", key=f"x{i+1}")
    predictors.append(predictor_values)

y_values = st.text_input("Y Values (comma-separated)")

if st.button("Calculate"):
    try:
        # Convert input values to arrays
        predictors = [list(map(float, p.split(','))) for p in predictors]
        y_values = list(map(float, y_values.split(',')))

        # Ensure the lengths of all arrays match
        if len(set(len(p) for p in predictors)) != 1 or len(predictors[0]) != len(y_values):
            st.error("All input arrays must have the same length.")
        else:
            # Perform linear regression
            X = np.column_stack(predictors)
            X = sm.add_constant(X)
            y = np.array(y_values)
            
            model = sm.OLS(y, X).fit()

            # Display results
            st.subheader("Model Summary")
            st.text(model.summary().as_text())

            st.subheader("ANOVA")
            anova = sm.stats.anova_lm(model, typ=2)
            st.write(anova)

            st.subheader("Coefficients")
            st.write(pd.DataFrame({
                "Coefficient": model.params,
                "P-value": model.pvalues,
                "Std Err": model.bse
            }))

            st.subheader("Model Equation")
            equation = "y = " + " + ".join([f"{coef}*X{i}" for i, coef in enumerate(model.params)])
            st.text(equation)

            st.subheader("Correlation Matrix")
            corr_matrix = np.corrcoef(X.T)
            st.write(pd.DataFrame(corr_matrix))
    except Exception as e:
        st.error(f"An error occurred: {e}")
