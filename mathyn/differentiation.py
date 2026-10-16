def limit(f, x, e=0.001):
    return (f(x-e) + f(x + e)) / 2

def derivative_newtons_method(f, x, dx=0.01): return (f(x+dx) - f(x)) / dx


def derivative(f, x, dx=0.01): return (f(x + dx) - f(x - dx))/(2 * dx)


def derivative_function(f, dx=0.01): return lambda x: derivative(f, x, dx)


def derivative_by_points(points, dx):
    derivative = []
    for i in range(1, len(points)-1):
        result = (points[i+1] - points[i - 1]) / (2 * dx)
        derivative.append(result)
    return np.array(derivative)


def higher_order_derivative(points, dx, order):
    derivative = points
    for i in range(0, order):
        derivative = derivative_by_points(derivative, dx)
    return derivative

def partial_derivative_2D_by_x(f, point, dx=0.01):
    x, y = point
    return (f(x+dx, y) - f(x-dx, y)) / (2*dx)


def partial_derivative_2D_by_y(f, point, dy=0.01):
    x, y = point
    return (f(x, y + dy) - f(x, y - dy)) / (2*dy)


def partial_derivative_3D_by_x(f, point, dx=0.01):
    x, y, z = point
    return (f(x+dx, y, z) - f(x-dx, y, z)) / (2*dx)


def partial_derivative_3D_by_y(f, point, dy=0.01):
    x, y, z = point
    return (f(x, y + dy, z) - f(x, y - dy, z)) / (2*dy)


def partial_derivative_3D_by_z(f, point, dz=0.01):
    x, y, z = point
    return (f(x, y, z + dz) - f(x, y, z - dz)) / (2*dz)


def directional_derivative(f, point, vector):
    return directional_derivative_3D(lambda x, y, z: f(x, y), [point[0], point[1], 0], [vector[0], vector[1], 0])


def directional_derivative_3D(f, point, vector):
    dfdx = partial_derivative_3D_by_x(f, point)
    dfdy = partial_derivative_3D_by_y(f, point)
    dfdz = partial_derivative_3D_by_z(f, point)
    cosa, cosb, cosy = direction_cosines_3D(vector)
    return dfdx*cosa + dfdy*cosb + dfdz*cosy


def gradient_vector(f, point):
    return gradient_vector_3D(lambda x, y, z: f(x, y), [point[0], point[1], 0])[:-1]


def gradient_vector_3D(f, point):
    dfdx = partial_derivative_3D_by_x(f, point)
    dfdy = partial_derivative_3D_by_y(f, point)
    dfdz = partial_derivative_3D_by_z(f, point)
    return np.array([dfdx, dfdy, dfdz])


def partial_derivatives_2D_by_points(points, dx):
    dfdx = np.array([])
    dfdy = np.array([])

    for i in range(points.shape[0]):
        dfdxi = derivative_by_points(points[i], dx)
        dfdx = np.append(dfdx, dfdxi)
    dfdx = dfdx.reshape(points.shape[0], -1)

    for j in range(points.shape[1]):
        dfdyj = derivative_by_points(points[:, j], dx)
        dfdy = np.append(dfdy, dfdyj)
    dfdy = dfdy.reshape(points.shape[1], -1).T
    return [dfdx, dfdy]
