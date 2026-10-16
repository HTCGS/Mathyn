import numpy as np

def factorial(x):
    result = 1
    for i in range(1, x+1):
        result *= i
    return result

def series_ratio_test(f, limval=99999):
    return limit(lambda x: np.abs(f(x+1) * np.power(f(x), -1)), limval)


def series_root_test(f, limval=99999):
    return limit(lambda x: np.power(np.abs(f(x+1) * np.power(f(x), -1)), 1/x), limval)


def series_integral_test(f, a=1, limval=99999):
    return simpsons_rule(f, a+1, limval + 1, 100) / simpsons_rule(f, a, limval, 100)


def fourier_series(f, l, n, x):
    a0 = 1/l * simpsons_rule(f, -l, l, 10)
    a1 = 1/l * simpsons_rule(lambda x: f(x)*np.cos(x), -l, l, 10)
    b1 = 1/l * simpsons_rule(lambda x: f(x)*np.sin(x), -l, l, 10)
    result = a0/2 + a1*np.cos(x) + b1*np.sin(x)
    for i in range(2, n+1):
        ai = 1/l * simpsons_rule(lambda x: f(x)*np.cos(i*x * (np.pi / l)), -l, l, 10)
        bi = 1/l * simpsons_rule(lambda x: f(x)*np.sin(i*x * (np.pi / l)), -l, l, 10)
        result += ai*np.cos(i*x) + bi*np.sin(i*x)
    return result

def least_squares(x, y):
    x_sum = x.sum()
    y_sum = y.sum()
    x2 = x**2
    x2_sum = x2.sum()
    xy = x*y
    xy_sum = xy.sum()
    n = len(x)
    det = x2_sum * n - x_sum*x_sum
    det_a = xy_sum * n - y_sum*x_sum
    det_b = x2_sum * y_sum - x_sum * xy_sum
    a = det_a / det
    b = det_b / det
    return lambda x: a*x + b

# def average_value2D(f, range_x, range_y, xn, yn):
#     start_x, end_x = range_x
#     start_y, end_y = range_y
#     delta_x = (end_x - start_x) / xn
#     delta_y = (end_y - start_y) / yn
#     delta_area = delta_x * delta_y
#     area = end_x * end_y
#     x = np.linspace(start_x + (delta_x/2), end_x - (delta_x/2), xn)
#     y = np.linspace(start_y + (delta_y/2), end_y - (delta_y/2), yn)
#     x, y = np.meshgrid(x, y)
#     z = f(x, y)
#     z_sum = 0
#     for i in range(len(z)):
#         z_sum += z[i].sum()
#     return (1 / area) * (delta_area * z_sum)

def simple_euler_method(f, a, b, y0, n):
    h = (b - a) / n
    x = np.arange(a, b+h, h)
    y = [y0]
    for i in range(1, len(x)):
        yi = y[i - 1] + h*f(x[i - 1], y[i-1])
        y.append(yi)
    return [x, np.array(y)]


def improved_euler_method(f, a, b, y0, n):
    h = (b - a) / n
    x = np.arange(a, b+h, h)
    y = [y0]
    for i in range(1, len(x)):
        yi = y[i - 1] + h*f(x[i - 1] + (h/2), y[i-1]+(h/2)*f(x[i - 1], y[i-1]))
        y.append(yi)
    return [x, np.array(y)]


def runge_kutta_method(f, a, b, y0, n):
    h = (b - a) / n
    x = np.arange(a, b+h, h)
    y = [y0]
    for i in range(1, len(x)):
        k1 = f(x[i-1], y[i-1])
        k2 = f(x[i-1] + (h/2), y[i-1] + (h*k1*0.5))
        k3 = f(x[i-1] + (h/2), y[i-1] + (h*k2*0.5))
        k4 = f(x[i-1] + h, y[i-1] + (h*k3))
        yi = y[i - 1] + (h/6)*(k1+2*k2+2*k3+k4)
        y.append(yi)
    return [x, np.array(y)]
