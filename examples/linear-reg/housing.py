#https://www.kaggle.com/faressayah/linear-regression-house-price-prediction#4.-LASSO-Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RANSACRegressor, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures



############################################################
def cross_val(model):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

if __name__ == "__main__":
    sns.set_style("whitegrid")
    plt.style.use("fivethirtyeight")

    USAhousing = pd.read_csv('../../exploratory/USA_Housing.csv')
    # print(USAhousing.head())
    # print(USAhousing.info())
    # print(USAhousing.describe())
    # print(USAhousing.columns)

    # sns.pairplot(USAhousing).savefig("pairplot.png")
    # sns.histplot(USAhousing['Price'])#.savefig("priceplot.png")
    # sns.heatmap(USAhousing.corr(), annot=True)#.savefig("heatmap.png")

    X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                'Avg. Area Number of Bedrooms', 'Area Population']]
    y = USAhousing['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    pipeline = Pipeline([
        ('std_scalar', StandardScaler())
    ])

    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)

    lin_reg = LinearRegression(normalize=True)
    lin_reg.fit(X_train,y_train)

    # print the intercept
    print(lin_reg.intercept_)

    coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])
    print(coeff_df)

    pred = lin_reg.predict(X_test)

    plt.scatter(y_test, pred)

    sns.displot((y_test - pred), bins=50);

    #Linear
    test_pred = lin_reg.predict(X_test)
    train_pred = lin_reg.predict(X_train)

    # print('Test set evaluation:\n_____________________________________')
    # print_evaluate(y_test, test_pred)
    # print('====================================')
    # print('Train set evaluation:\n_____________________________________')
    # print_evaluate(y_train, train_pred)

    results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, test_pred) , cross_val(LinearRegression())]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])

    #Robust Regression
    model = RANSACRegressor(base_estimator=LinearRegression(), max_trials=100)
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)

    # print('Test set evaluation:\n_____________________________________')
    # print_evaluate(y_test, test_pred)
    # print('====================================')
    # print('Train set evaluation:\n_____________________________________')
    # print_evaluate(y_train, train_pred)

    results_df_2 = pd.DataFrame(data=[["Robust Regression", *evaluate(y_test, test_pred) , cross_val(RANSACRegressor())]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
    results_df = results_df.append(results_df_2, ignore_index=True)

    #Ridge Regression
    model = Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)

    results_df_2 = pd.DataFrame(data=[["Ridge Regression", *evaluate(y_test, test_pred) , cross_val(Ridge())]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
    results_df = results_df.append(results_df_2, ignore_index=True)

    #Lasso Regression
    model = Lasso(alpha=0.1, 
                precompute=True, 
    #               warm_start=True, 
                positive=True, 
                selection='random',
                random_state=42)
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)

    results_df_2 = pd.DataFrame(data=[["Lasso Regression", *evaluate(y_test, test_pred) , cross_val(Lasso())]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
    results_df = results_df.append(results_df_2, ignore_index=True)

    #Elastic Net
    model = ElasticNet(alpha=0.1, l1_ratio=0.9, selection='random', random_state=42)
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)

    results_df_2 = pd.DataFrame(data=[["Elastic Net Regression", *evaluate(y_test, test_pred) , cross_val(ElasticNet())]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
    results_df = results_df.append(results_df_2, ignore_index=True)

    #Polynomial Regression
    poly_reg = PolynomialFeatures(degree=2)

    X_train_2_d = poly_reg.fit_transform(X_train)
    X_test_2_d = poly_reg.transform(X_test)

    lin_reg = LinearRegression(normalize=True)
    lin_reg.fit(X_train_2_d,y_train)

    test_pred = lin_reg.predict(X_test_2_d)
    train_pred = lin_reg.predict(X_train_2_d)

    results_df_2 = pd.DataFrame(data=[["Polynomail Regression", *evaluate(y_test, test_pred), 0]], 
                                columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
    results_df = results_df.append(results_df_2, ignore_index=True)
    print(results_df)