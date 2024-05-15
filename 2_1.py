import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from cvxopt import matrix
from cvxopt import solvers

t_x = np.linspace(-5, 5, 1000)


# 线性核函数
def liner(x, z):
    return np.inner(x, z)


#  多项式
def poly(x, z, c, p):
    return (np.inner(x, z) + c) ** p


# 高斯
def Gaussian(x, z, l):
    distance = np.sum((x - z) ** 2)
    return np.exp(-distance / (l * l))


def periodic(x, z, p, l):

    return np.exp((np.sin(np.pi * np.linalg.norm(x - z) / p) ** 2) /((l ** 2) * -2))


def RBF(x, z, a, r):
    return a * np.exp(-r * np.linalg.norm(x - z))


def get_linear(x, z, z1):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    k = liner(x, z)
    k2 = liner(x, z1)
    plt.plot(x, k, color='black')
    plt.title('liner')
    plt.xlabel('x')
    plt.ylabel("k(x,1)")

    plt.subplot(1, 2, 2)
    plt.plot(x, k2, color='black')
    plt.title('liner')
    plt.xlabel('x')
    plt.ylabel("k(x,0)")
    plt.show()


def get_data():

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

    # # 绘制Gaussian数据集
    # plt.figure(figsize=(8, 8))
    # plt.scatter(X1[:, 0], X1[:, 1], c='blue', label='Class -1')
    # plt.scatter(X2[:, 0], X2[:, 1], c='red', label='Class +1')
    # plt.xlabel('X1')
    # plt.ylabel('X2')
    # plt.legend(loc='best')
    # plt.title('Gaussian Dataset')
    # plt.show()

    # 读入Moon数据集
    X_moon, Y_moon = make_moons(n_samples=200, noise=0.1, random_state=0)

    # 划分Moon数据集为训练集、验证集和测试集
    X_train_moon, X_temp_moon, Y_train_moon, Y_temp_moon = train_test_split(X_moon, Y_moon, test_size=0.4,
                                                                            random_state=0)
    X_val_moon, X_test_moon, Y_val_moon, Y_test_moon = train_test_split(X_temp_moon, Y_temp_moon, test_size=0.5,
                                                                        random_state=0)

    # # 绘制Moon数据集
    # plt.figure(figsize=(8, 8))
    # plt.scatter(X_moon[Y_moon == 0, 0], X_moon[Y_moon == 0, 1], c='blue', label='Class 0')
    # plt.scatter(X_moon[Y_moon == 1, 0], X_moon[Y_moon == 1, 1], c='red', label='Class 1')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.legend(loc='best')
    # plt.title('Moon Dataset')
    # plt.show()

    return X_train, Y_train, X_test, Y_test, X_val, Y_val, X_train_moon, Y_train_moon, X_test_moon, Y_test_moon, X_val_moon, Y_val_moon


def train_svm(X, Y, kernel, kernel_params):
    K = kernel(X, X, *kernel_params)
    p = matrix(np.outer(Y, Y) * K)
    q = matrix(-np.ones(len(Y)))
    G = matrix(-np.identity(len(Y)))
    h = matrix(np.zeros(len(Y)))
    A = matrix(Y, (1, len(Y)))
    b = matrix(0.0)
    sol = solvers.qp(p, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return K, alphas


if __name__ == '__main__':
    # 多项式几何函数，p/c越大函数斜率越大
    #  l越大，高斯分布的方差越大，即图像开口越大
    # 周期核函数，p越大抖动的频率越小，l越带值越小
    # a控制函数图像高低，a越大函数值越大，r控制开口大小，r越大开口越小

    # 数据集
    X_train, Y_train, X_test, Y_test, X_val, Y_val, X_train_moon, Y_train_moon, X_test_moon, Y_test_moon, X_val_moon, Y_val_moon = get_data()
    # 定义不同的核函数和参数
    kernel_functions = [liner,poly, Gaussian, periodic, RBF]
    kernel_names = ['liner', 'Polynomial', 'Gaussian', 'Periodic', 'RBF']
    kernel_parameters = [
        [],
        [3, 2],
        [2],
        [1, 2],
        [1, 1]
    ]
    # for kernel_name, kernel, kernel_params in zip(kernel_names, kernel_functions, kernel_parameters):
    #     k1 = [kernel(x, 0, *kernel_params)for x in t_x]
    #     k2 = [kernel(x, 1, *kernel_params)for x in t_x]
    #
    #     plt.figure(figsize=(12, 4))
    #     plt.subplot(121)
    #     plt.plot(t_x, k1)
    #     plt.title(f'{kernel_name} - k(x, 1)')
    #
    #     plt.subplot(122)
    #     plt.plot(t_x, k2)
    #     plt.title(f'{kernel_name} - k(x, 0)')
    #
    #     plt.tight_layout()
    #     plt.show()
    # # 迭代不同的核函数和参数

    for kernel_name, kernel, kernel_params in zip(kernel_names, kernel_functions, kernel_parameters):
        print(f"Kernel: {kernel_name}")
        K, alphas = train_svm(X_train, Y_train, kernel, kernel_params)

        # 计算 b
        support_vector = np.where(alphas > 1e-5)[0]
        print(len(support_vector))
        # w = np.sum(alphas[support_vector] * Y_train[support_vector] * X_train[support_vector])
        # # 计算b
        # b = np.mean(Y_train[support_vector] - np.dot(K[support_vector], w))
        # # print(f"b: {b}\n")