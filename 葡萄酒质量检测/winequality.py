import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# 加载数据集
data = pd.read_csv('D:\BaiduNetdiskDownload\winequality-red.csv', sep=';')
print(data.head())
print(data.shape)   #[1599 rows x 12 columns]

# 训练集/测试集
y = data.quality
x = data.drop('quality', axis=1)
print(x)    # #[1599 rows x 11 columns]
print(y)
X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    random_state=123,
                                                    stratify=y)
print(X_train.shape)  #(1279, 11)
print(y_train.shape)  #(1279, 1)
print(X_test.shape)   #(320, 11)
print(y_test.shape)   #(320, 1)

# 运行线性回归模型
linreg = LinearRegression()
linreg.fit(X_train,y_train)
print(linreg.intercept_)  #[ 26.93130076593354]
print(linreg.coef_)       #[ 4.28178608e-02 -1.12214812e+00 -2.65534169e-01  1.26822261e-02-1.79327314e+00  5.36365135e-03 -3.23265154e-03 -2.35540727e+01-2.77765613e-01  9.16315949e-01  2.87309248e-01]

#模型拟合测试集
y_pred = linreg.predict(X_test)
from sklearn import metrics
#scikit-learn计算MSE RMSE
print("MSE:",metrics.mean_squared_error(y_test, y_pred))    #MSE: 0.45001367789267305
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  #RMSE:  0.6708305880717375


from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(linreg,x , y, cv=10)

print("MSE:",metrics.mean_squared_error(y, predicted))  #MSE: 20.7892840922
print("RMSE:",np.sqrt(metrics.mean_squared_error(y, predicted)))    #RMSE: 4.55952673994

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()