
# Importing necessary libraries
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Setting Plotly template
pio.templates.default = "plotly_white"

# Reading data from CSV
data = pd.read_csv('retail_price.csv')

# Displaying the first few rows of the data
print(data.head())

# Checking for missing values
print(data.isnull().sum())

# Displaying descriptive statistics
print(data.describe())

# Creating a histogram of total prices
fig = px.histogram(data, x='total_price', nbins=20, title='Distribution of Total Price')
fig.show()

# Creating a box plot of unit prices
fig = px.box(data, y='unit_price', title='Box Plot of Unit Price')
fig.show()

# Creating a scatter plot of quantity vs. total price with a trendline
fig = px.scatter(data, x='qty', y='total_price', title='Quantity vs Total Price', trendline="ols")
fig.show()

# Creating a bar chart of average total price by product category
fig = px.bar(data, x='product_category_name', y='total_price', title='Average Total Price by Product Category')
fig.show()

# Creating box plots of total price by weekday and holiday
fig = px.box(data, x='weekday', y='total_price', title='Box Plot of Total Price by Weekday')
fig.show()

fig = px.box(data, x='holiday', y='total_price', title='Box Plot of Total Price by Holiday')
fig.show()

# Calculating the correlation matrix and creating a heatmap
numeric_data = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
fig = go.Figure(go.Heatmap(x=correlation_matrix.columns,
                           y=correlation_matrix.columns,
                           z=correlation_matrix.values))
fig.update_layout(title='Correlation Heatmap of Numerical Features')
fig.show()

# Creating a new column 'comp_price_diff' and bar chart of average competitor price difference by product category
data['comp_price_diff'] = data['unit_price'] - data['comp_1']
avg_price_diff_by_category = data.groupby('product_category_name')['comp_price_diff'].mean().reset_index()
fig = px.bar(avg_price_diff_by_category, x='product_category_name', y='comp_price_diff',
             title='Average Competitor Price Difference by Product Category')
fig.update_layout(xaxis_title='Product Category', yaxis_title='Average Competitor Price Difference')
fig.show()

# Splitting data into training and testing sets for the regression model
X = data[['qty', 'unit_price', 'comp_1', 'product_score', 'comp_price_diff']]
y = data['total_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training a decision tree regression model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Visualizing predicted vs. actual retail prices
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', marker=dict(color='blue'),
                         name='Predicted vs. Actual Retail Price'))
fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)],
                         mode='lines', marker=dict(color='red'), name='Ideal Prediction'))
fig.update_layout(title='Predicted vs. Actual Retail Price',
                  xaxis_title='Actual Retail Price',
                  yaxis_title='Predicted Retail Price')
fig.show()
