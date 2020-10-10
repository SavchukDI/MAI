import math
import numpy as np
from matplotlib import pyplot as plt

# GLOBAL CONSTS
T1 = 20  # Температура слева
T2 = 800  # Температура справа
ALPHA1 = 25  # Коэффициент теплоотдачи слева
ALPHA2 = 46.5  # Коэффициент теплоотдачи справа
h = 150e-3  # Толщина стенки
A = 70
B = -100
C = -20
D = 0.05e-3
E = 1
n = 100  # Количество точек
dx = (h - 0) / n  # Шаг
print(f'dx = {dx}')

# FUNCTIONS
k = lambda x: A + B * x * h / (n - 1)
dk = lambda x: (k(x + dx) - k(x - dx)) / (2 * dx)
f = lambda x: C * math.exp(-D * (x * h / (n - 1) - h / E) ** 2)


def d(x):
    """
    Функция, задающая неоднородность в СЛАУ
    """
    if x == 0:
        return ALPHA1 * dx * T1
    elif x == n - 1:
        return ALPHA2 * dx * T2
    else:
        return 2 * f(x) * dx ** 2


def coeff():
    """
    Функция для расчета коэффициентов a, b, c
    """
    a, b, c = [], [], []

    # Расчет трехдиагональных элементов
    for i in range(n):
        if i == 0:
            a.append(0)
            b.append(ALPHA1 * dx + k(i))
            c.append(-k(i))
        elif i == n - 1:
            a.append(-k(i))
            b.append(ALPHA2 * dx + k(i))
            c.append(0)
        else:
            a.append(2 * k(i) - dx * dk(i))
            b.append(-4 * k(i))
            c.append(2 * k(i) + dx * dk(i))

    return a, b, c


def tm_alg():
    """
    Функция, реализующая метод прогонки
    """
    a, b, c = coeff()
    P, Q, X = [], [], [0] * n

    # Расчет "прогоночных" коэффициентов
    for i in range(n):
        if i == 0:
            P.append(-c[i] / b[i])
            Q.append(d(i) / b[i])
        else:
            P.append(-c[i] / (a[i] * P[i - 1] + b[i]))
            Q.append((d(i) - a[i] * Q[i - 1]) / (a[i] * P[i - 1] + b[i]))

    # Проверка устойчивости метода прогонки
    for i in P:
        if abs(i) > 1:
            return 0

    # Обратный ход метода прогонки
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            X[i] = Q[i]
        else:
            X[i] = P[i] * X[i + 1] + Q[i]

    return X


def derivative_t(T):
    """
    Функция, реализующая взятие производной
    """
    dt_sum, q = [], []

    for i in range(n):
        if i == 0:
            dt = (T[i + 1] - T[i]) / dx
        elif i == n - 1:
            dt = (T[i] - T[i - 1]) / dx
        else:
            dt = (T[i + 1] - T[i - 1]) / (2 * dx)
        dt_sum.append(dt)

    for i, j in enumerate(dt_sum):
        q.append(-k(i) * j)

    return q


def solver():
    """
    Функция для решения СЛАУ методом научной библиотеки Numpy
    """
    a, b, c = coeff()
    D = []
    M = np.zeros([n, n])
    j = 0

    for i in range(n):
        D.append(d(i))

    for i in range(n):
        if i == 0:
            M[i][j] = b[j]
            M[i][j + 1] = c[j]
        elif i == n - 1:
            M[i][j - 1] = a[j]
            M[i][j] = b[j]
        else:
            M[i][j - 1] = a[j]
            M[i][j] = b[j]
            M[i][j + 1] = c[j]
        j += 1

    res = np.linalg.solve(M, np.array(D))

    return res


def derivative_t_2(T):
    """
    Функция, реализующая взятие производной методом научной библиотеки Numpy
    """
    dt_sum, q = [], []
    dt_sum = np.gradient(T, dx)

    for i, j in enumerate(dt_sum):
        q.append(-k(i) * j)

    return q


def main():
    T = tm_alg()  # Стационарное распределение температуры

    if T == 0:
        print('Метод прогонки не устойчив!')
        return 0

    q = derivative_t(T)  # Тепловой поток

    T_2 = solver()  # Стационарное распределение температуры (Numpy)
    q_2 = derivative_t_2(T)  # Тепловой поток (Numpy)

    x = np.linspace(0, h, n)  # Лист точек

    KK, FF = [], []

    for i in range(n):
        KK.append(k(i))  # Значения функции k(x)
        FF.append(f(i))  # Значения функции f(x)

    # Сравнения двух методов решения
    delta_T = np.array(T) - T_2
    delta_q = np.array(q) - q_2

    # Построение графиков
    plt.figure(figsize=(9, 9))
    plt.subplot(2, 1, 1)
    plt.plot(x, KK, color='black')
    plt.title("k(x) = A + Bx. Переменный коэффициент теплопроводности")
    plt.ylabel('k(x), Вт/(м*C)')
    plt.xlabel('x, м')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(x, FF, color='black')
    plt.title("f(x) = C*exp{-D(x-h/E)^2}")
    plt.ylabel('f(x), Вт/м^3')
    plt.xlabel('x, м')
    plt.grid()
    plt.show()

    # Графики T и q
    plt.figure(figsize=(9, 9))
    plt.subplot(2, 1, 1)
    plt.plot(x, T, color='black')
    plt.title("Распределение температуры внутри плоской стенки")
    plt.ylabel('T(x), C')
    plt.xlabel('x, м')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(x, q, color='black')
    plt.title("Тепловой поток через стенку")
    plt.ylabel('q(x), Вт/м^2')
    # plt.title("Производная температуры")
    # plt.ylabel('dT(x)/dx, производная температуры')
    plt.xlabel('x, м')
    plt.grid()
    plt.show()

    # Графики T и q (Numpy)
    plt.figure(figsize=(9, 9))
    plt.subplot(2, 1, 1)
    plt.plot(x, T_2, color='black')
    plt.title("Распределение температуры внутри плоской стенки")
    plt.ylabel('T_2(x), C')
    plt.xlabel('x, м')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(x, q_2, color='black')
    plt.title("Тепловой поток через стенку")
    plt.ylabel('q_2(x), Вт/м^2')
    plt.xlabel('x, м')
    plt.grid()
    plt.show()

    # Графики delta_T и delta_q
    plt.figure(figsize=(9, 9))
    plt.subplot(2, 1, 1)
    plt.plot(x, delta_T, color='black')
    plt.title("Разность температур двух методов")
    plt.ylabel('delta_T(x), C')
    plt.xlabel('x, м')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(x, delta_q, color='black')
    plt.title("Разность теплового потока двух методов")
    plt.ylabel('delta_q(x), Вт/м^2')
    plt.xlabel('x, м')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
