import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from cvxopt import matrix
from cvxopt import solvers
# 生成Gaussian数据集
# 随机种子确保每次随机生成的数据一直
np.random.seed(0)

# 生成X1和X2的样本点
mean1 = np.array([-3, -3])
mean2 = np.array([3, 3])
cov = np.array([[2, -1], [-1, 2]])
X1 = np.random.multivariate_normal(mean1, cov, 80)
X2 = np.random.multivariate_normal(mean2, cov, 80)

# 创建标签，-1表示X1样本点，+1表示X2样本点
y1 = -np.ones(80)
y2 = np.ones(80)

# 合并数据集
X = np.vstack((X1, X2))
Y = np.hstack((y1, y2))

# 划分数据集为训练集、验证集和测试集
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=0)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=0)

# 绘制Gaussian数据集
plt.figure(figsize=(8, 8))
plt.scatter(X1[:, 0], X1[:, 1], c='blue', label='Class -1')
plt.scatter(X2[:, 0], X2[:, 1], c='red', label='Class +1')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='best')
plt.title('Gaussian Dataset')
plt.show()

# 读入Moon数据集
X_moon, Y_moon = make_moons(n_samples=200, noise=0.1, random_state=0)

# 划分Moon数据集为训练集、验证集和测试集
X_train_moon, X_temp_moon, Y_train_moon, Y_temp_moon = train_test_split(X_moon, Y_moon, test_size=0.4,
                                                                        random_state=0)
X_val_moon, X_test_moon, Y_val_moon, Y_test_moon = train_test_split(X_temp_moon, Y_temp_moon, test_size=0.5,
                                                                    random_state=0)
X_train = X_train.T
X_val = X_val.T
X_test = X_test.T
X_train_moon = X_train_moon.T
X_val_moon = X_val_moon.T
X_test_moon = X_test_moon.T

# 绘制Moon数据集
plt.figure(figsize=(8, 8))
plt.scatter(X_moon[Y_moon == 0, 0], X_moon[Y_moon == 0, 1], c='blue', label='Class 0')
plt.scatter(X_moon[Y_moon == 1, 0], X_moon[Y_moon == 1, 1], c='red', label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='best')
plt.title('Moon Dataset')
plt.show()

# 绘制训练集和测试集
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=plt.cm.Paired, marker='o', label='Train Set')
plt.title('Train Set')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap=plt.cm.Paired, marker='o', label='Test Set')
plt.title('Test Set')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='best')

# 绘制分类曲线和间隔曲线
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# 对每个点进行分类预测
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制分类曲线
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# 绘制间隔曲线
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])

# 标识支持向量
support_vectors = X_train[svm_model.support_]
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], c='none', marker='o', edgecolors='k', s=80, label='Support Vectors')
plt.legend()

plt.show()
