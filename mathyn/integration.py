def left_point_rule(f, a, b, n):
    h = (b - a) / n
    x = np.arange(a, b, h)
    y = f(x)
    s = y.sum() * h
    return s

def right_point_rule(f, a, b, n):
    h = (b - a) / n
    x = np.arange(a+h, b+h, h)
    y = f(x)
    s = y.sum() * h
    return s

def mid_point_rule(f, a, b, n):
    h = (b - a) / n
    x = np.arange(a+(h/2), b+(h/2), h)
    y = f(x)
    s = y.sum() * h
    return s

def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    if h == 0:
        return 0
    x = np.arange(a, b + h, h)
    y = f(x)
    s = h * ((y[0] + y[-1])/2 + y[1:-1].sum())
    return s

def simpsons_rule(f, a, b, n=10):
    h = (b - a) / (2 * n)
    if h == 0:
        return 0
    x = np.arange(a, b + h, h)
    y = f(x)
    even = y[::2]
    odd = y[1::2]
    s = (h / 3) * (y[0] + y[-1] + 2*even[1:-1].sum() + 4*odd.sum())
    return s

def double_integral(f, range, f_lower, f_upper, axis=0, xn=10, yn=10):
    if axis == 1:
        f = swap_functions_parameters_2D(f)
    start_x, end_x = range
    delta_x = (end_x - start_x) / xn
    x = np.linspace(start_x + (delta_x/2), end_x - (delta_x/2), xn)
    z_sum = 0
    # print(f(1, 1))
    # print(f(0, 0))
    # print(f(1, 0))
    # print(f(2, 0))
    # print(f(0, -1))
    # print(f(0, 0))
    # print(f(0, 1))
    # print(f(0, 2))
    for i in x:
        start_y = f_lower(i)
        end_y = f_upper(i)
        delta_y = (end_y - start_y) / yn
        delta_area = delta_x * delta_y
        y = np.linspace(start_y + (delta_y/2), end_y - (delta_y/2), yn)
        f_sum = 0
        for j in y:
            f_sum += f(i, j)
        f_sum *= delta_area
        z_sum += f_sum
    return z_sum

def triple_integral(f, range_x, fy_lower, fy_upper, fz_lower, fz_upper, axis="xyz", xn=10, yn=10, zn=10):
    if axis != "xyz":
        f = swap_functions_parameters_3D(f, axis)
    start_x, end_x = range_x
    delta_x = (end_x - start_x) / xn
    x = np.linspace(start_x + (delta_x/2), end_x - (delta_x/2), xn)
    result = 0
    for i in x:
        start_y = fy_lower(i)
        end_y = fy_upper(i)
        delta_y = (end_y - start_y) / yn
        y = np.linspace(start_y + (delta_y/2), end_y - (delta_y/2), yn)
        f_sum = 0
        for j in y:
            start_z = fz_lower(i, j)
            end_z = fz_upper(i, j)
            delta_z = (end_z - start_z) / zn
            delta_area = delta_x * delta_y * delta_z
            z = np.linspace(start_z + (delta_z/2), end_z - (delta_z/2), zn)
            for k in z:
                f_sum += f(i, j, k)
            f_sum *= delta_area
            result += f_sum
    return result

def arc_length(f, a, b, n):
    dx = (b - a) / (2 * n)
    l = simpsons_rule(lambda x: np.sqrt(1 + (derivative(f, x, dx)**2)), a, b, n)
    return l


def average_value(f, a, b, n):
    value = 1/(b-a) * simpsons_rule(f, a, b, n)
    return value


def surface_area_of_revolution_about_y(f, a, b, n):
    dx = (b - a) / (2 * n)
    area = simpsons_rule(lambda x: 2 * x * np.pi * np.sqrt(1 + (derivative(f, x, dx)**2)), a, b, n)
    return area


def surface_area_of_revolution_about_x(f, a, b, n):
    dx = (b - a) / (2 * n)
    area = simpsons_rule(lambda x: 2 * f(x) * np.pi * np.sqrt(1 + (derivative(f, x, dx)**2)), a, b, n)
    return area


def volume_of_revolution(f, a, b, n):
    volume = simpsons_rule(lambda x: np.pi * np.power(f(x), 2), a, b, n)
    return volume


def volume_of_revolution_between_curves(f1, f2, a, b, n):
    volume = simpsons_rule(lambda x: np.pi * (np.power(f1(x), 2) - np.power(f2(x), 2)), a, b, n)
    return volume


def moments_of_system(f1, f2, a, b, n):
    Mx = simpsons_rule(lambda x: 0.5 * (np.power(f1(x), 2) - np.power(f2(x), 2)), a, b, n)
    My = simpsons_rule(lambda x: x * (f1(x) - f2(x)), a, b, n)
    return (Mx, My)


def center_of_mass(f1, a, b, n):
    area = simpsons_rule(lambda x: f1(x), a, b, n)
    Mx, My = moments_of_system(f1, lambda x: 0, a, b, n)
    x = My / area
    y = Mx / area
    return (x, y)


def center_of_mass_between_curves(f1, f2, a, b, n):
    area = simpsons_rule(lambda x: f1(x) - f2(x), a, b, n)
    Mx, My = moments_of_system(f1, f2, a, b, n)
    x = My / area
    y = Mx / area
    return (x, y)

def average_value_2D(f, range, f_lower, f_upper, axis=0, xn=10, yn=10):
    return double_integral(f, range, f_lower, f_upper, axis, xn, yn) / double_integral(lambda x, y: 1, range, f_lower, f_upper, axis, xn, yn)


def surface_area_2D(f, range, f_lower, f_upper, axis=0, xn=10, yn=10):
    return double_integral(lambda x, y: np.sqrt(1 + partial_derivative_2D_by_x(f, [x, y])**2 +
                                                partial_derivative_2D_by_y(f, [x, y])**2), range, f_lower, f_upper, axis, xn, yn)


def center_of_mass_2D(f, range, f_lower, f_upper, axis=0, xn=10, yn=10):
    m = double_integral(f, range, f_lower, f_upper, axis, xn, yn)
    lx = double_integral(lambda x, y: x * f(x, y), range, f_lower, f_upper, axis, xn, yn)
    ly = double_integral(lambda x, y: y * f(x, y), range, f_lower, f_upper, axis, xn, yn)
    x = lx / m
    y = ly / m
    return [x, y]


def center_of_mass_3D(f, range_x, fy_lower, fy_upper, fz_lower, fz_upper, xn=10, yn=10, zn=10):
    m = triple_integral(f, range_x, fy_lower, fy_upper, fz_lower, fz_upper, xn, yn, zn)
    lx = triple_integral(lambda x, y, z: x * f(x, y, z), range_x, fy_lower, fy_upper, fz_lower, fz_upper, xn, yn, zn)
    ly = triple_integral(lambda x, y, z: y * f(x, y, z), range_x, fy_lower, fy_upper, fz_lower, fz_upper, xn, yn, zn)
    lz = triple_integral(lambda x, y, z: z * f(x, y, z), range_x, fy_lower, fy_upper, fz_lower, fz_upper, xn, yn, zn)
    x = lx / m
    y = ly / m
    z = lz / m
    return [x, y, z]

