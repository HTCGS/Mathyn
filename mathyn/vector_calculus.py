def line_by_point_and_slope(point, slope):
    px, py = point
    return lambda x: py + slope*(x - px)


def line_by_point_and_vector(point, vector):
    px, py = point
    vx, vy = vector
    if vx == 0 or vy == 0:
        return lambda x: 0
    return lambda x: py + (vy * ((x - px) / vx))


def line_by_two_points(p1, p2, x):
    vector = [p2[0] - p1[0], p2[1] - p1[1]]
    return line_by_point_and_vector(p1, vector, x)


def line_by_point_and_normal_vector(point, normal):
    px, py = point
    nx, ny = normal
    if nx == 0 or ny == 0:
        return lambda x: 0
    return lambda x: py - ((nx * (x - px)) / ny)


def distance_between_point_and_line(point, a, b, c):
    px, py = point
    num = abs(a*px + b*py + c)
    den = np.sqrt(a*a + b*b)
    return num/den


def distance_between_point_and_plane(point, a, b, c, d):
    px, py, pz = point
    num = abs(a*px + b*py + c*pz + d)
    den = np.sqrt(a*a + b * b + c*c)
    return num/den


def distance_between_parallel_planes(a, b, c, d1, d2):
    return abs(d2-d1) / np.sqrt(a*a + b*b + c*c)


def angle_between_planes(p1, p2):
    a1, b1, _ = p1
    a2, b2, _ = p2
    tg = (a1*b2 - b1*a2) / (a1*a2 + b1*b2)
    return np.arctan(tg)


def angle_between_vectors_3D(v1, v2):
    return np.arccos(vector_dot_3D(v1, v2) / (vector_magnitude_3D(v1) * vector_magnitude_3D(v2)))


def angle_between_vectors(v1, v2):
    v1x, v1y = v1
    v2x, v2y = v2
    return angle_between_vectors_3D([v1x, v1y, 0], [v2x, v2y, 0])


def vector_dot_3D(v1, v2):
    v1x, v1y, v1z = v1
    v2x, v2y, v2z = v2
    return v1x*v2x+v1y*v2y+v1z*v2z


def vector_dot(v1, v2):
    v1x, v1y = v1
    v2x, v2y = v2
    return vector_dot_3D([v1x, v1y, 0], [v2x, v2y, 0])


def vector_cross_3D(v1, v2):
    v1x, v1y, v1z = v1
    v2x, v2y, v2z = v2
    i = (v1y*v2z) - (v2y*v1z)
    j = -((v1x*v2z) - (v2x*v1z))
    k = (v1x*v2y) - (v2x*v1y)
    return np.array([i, j, k])


def vector_cross(v1, v2):
    v1x, v1y = v1
    v2x, v2y = v2
    return vector_cross_3D([v1x, v1y, 0], [v2x, v2y, 0])


def vector_magnitude_3D(v):
    vx, vy, vz = v
    return np.sqrt(vx**2+vy**2+vz**2)


def vector_magnitude(v):
    vx, vy = v
    return vector_magnitude_3D([vx, vy, 0])


def projection_on_vector(v, p):
    return projection_on_vector_3D([v[0], v[1], 0], [p[0], p[1], 0])


def projection_on_vector_3D(v, p):
    vx, vy, vz = v
    px, py, pz = p
    return (vx*px + vy*py + vz*pz) / np.sqrt(px**2 + py**2 + pz**2)

def direction_cosines(vector):
    return direction_cosines_3D([vector[0], vector[1], 0])[:-1]


def direction_cosines_3D(vector):
    x, y, z = vector
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    cosa = x/magnitude
    cosb = y/magnitude
    cosy = z/magnitude
    return np.array([cosa, cosb, cosy])

def tangent_plane(f, point):
    px, py, pz = point
    dfdx = partial_derivative_3D_by_x(f, point)
    dfdy = partial_derivative_3D_by_y(f, point)
    return lambda x, y: dfdx[0]*(x - px) + dfdy[0]*(y - py) + pz


def normal_plane(f, point):
    px, py, pz = point
    dfdx = partial_derivative_3D_by_x(f, point)
    dfdy = partial_derivative_3D_by_y(f, point)
    return lambda x, y: 1/dfdx[0]*(x - px) + 1/dfdy[0]*(y - py) + pz


def normal_vector_3D(f, point):
    return gradient_vector_3D(f, point)


def normal_line_2D(f):
    dfdx = partial_derivative_2D_by_x(f)


def tangent_line_of_curve(f, a, dx=0.01):
    return lambda x: f(a) + derivative(f, a, dx) * (x - a)


def normal_line_of_curve(f, a, dx=0.01):
    df = derivative(f, a, dx)
    if abs(df) == 0:
        return lambda x: f(a) * x
    return lambda x: f(a) + (-1 * np.power(derivative(f, a, dx), -1)) * (x - a)


def vector_function(fi, fj, fk):
    return lambda t: np.array([fi(t), fj(t), fk(t)], dtype=object)


def derivative_of_vector_function(vectors, dx):
    i = derivative_by_points(vectors[0], dx)
    j = derivative_by_points(vectors[1], dx)
    k = derivative_by_points(vectors[2], dx)
    return np.array([i, j, k])


def derivative_of_vector_function_at_point(f, point, dx=0.01):
    t = np.linspace(point - dx, point + dx, 3)
    vectors = f(t)
    return derivative_of_vector_function(vectors, dx)


def unit_tangent_vector(fi, fj, fk, point):
    di = derivative(fi, point)
    dj = derivative(fj, point)
    dk = derivative(fk, point)
    magnitude = vector_magnitude_3D([di, dj, dk])
    return [di / magnitude, dj / magnitude, dk / magnitude]


def unit_normal_vector(fi, fj, fk, point):
    f = vector_function(fi, fj, fk)
    t = np.linspace(point - 0.02, point + 0.02, 5)
    vectors = f(t)
    dfdt = derivative_of_vector_function(vectors, 0.01)
    didt, djdt, dkdt = dfdt
    magnitude_f = vector_magnitude_3D([didt[1], djdt[1], dkdt[1]])
    dfdt = dfdt / magnitude_f
    d2fdt = derivative_of_vector_function(dfdt, 0.01)
    d2idt, d2jdt, d2kdt = d2fdt
    magnitude_t = vector_magnitude_3D([d2idt[0], d2jdt[0], d2kdt[0]])
    return [d2idt[0] / magnitude_t, d2jdt[0] / magnitude_t, d2kdt[0] / magnitude_t]


def binormal_vector(fi, fj, fk, point):
    tangent = unit_tangent_vector(fi, fj, fk, point)
    normal = unit_normal_vector(fi, fj, fk, point)
    return vector_cross_3D(tangent, normal)


def tangential_comp_of_acceleration(fv, point, dx=0.01):
    t = np.linspace(point - (2*dx), point + (2*dx), 5)
    vectors = fv(t)
    dfdt = derivative_of_vector_function(vectors, dx)
    d2fdt = derivative_of_vector_function(dfdt, dx)
    return vector_dot_3D([dfdt[0][1], dfdt[1][1], dfdt[2][1]], [d2fdt[0][0], d2fdt[1][0], d2fdt[2][0]]) / vector_magnitude_3D([dfdt[0][1], dfdt[1][1], dfdt[2][1]])


def normal_comp_of_acceleration(fv, point, dx=0.01):
    t = np.linspace(point - (2*dx), point + (2*dx), 5)
    vectors = fv(t)
    dfdt = derivative_of_vector_function(vectors, dx)
    d2fdt = derivative_of_vector_function(dfdt, dx)
    return vector_magnitude_3D(vector_cross_3D([dfdt[0][1], dfdt[1][1], dfdt[2][1]], [d2fdt[0][0], d2fdt[1][0], d2fdt[2][0]])) / vector_magnitude_3D([dfdt[0][1], dfdt[1][1], dfdt[2][1]])


def arc_length_of_vector_function(fi, fj, fk, a, b, n=10):
    return simpsons_rule(lambda t: np.sqrt(derivative(fi, t)**2 + derivative(fj, t)**2 + derivative(fk, t)**2), a, b, n)


def curvature(points):
    dfdx = derivative_by_points(points, 0.01)
    d2fdx = derivative_by_points(dfdx, 0.01)
    dfdx = dfdx[1:-1]
    return d2fdx / np.power(np.sqrt(1 + dfdx*dfdx), 3)


def line_integral(f, a, b, line):
    return simpsons_rule(lambda x: f(x, line(x)) * np.sqrt(1 + derivative(line, x)**2), a, b)


def line_integral_3D(f, a, b, xt, yt, zt):
    return simpsons_rule(lambda t: f(xt(t), yt(t), zt(t)) *
                         np.sqrt(derivative(xt, t)**2 + derivative(yt, t)**2 + derivative(zt, t)**2), a, b)


def line_integral_by_parametric_curve(f, a, b, xt, yt):
    return simpsons_rule(lambda t: f(xt(t), yt(t)) * np.sqrt(derivative(xt, t)**2 + derivative(yt, t)**2), a, b)


def line_integral_of_2form_greens_rule(p, q, range, f_lower, f_upper, axis=0, xn=10, yn=10):
    return double_integral(lambda x, y: partial_derivative_2D_by_x(q, [x, y]) - partial_derivative_2D_by_y(p, [x, y]),
                           range, f_lower, f_upper, axis, xn, yn)


def surface_integral(f, surface, range, f_lower, f_upper, proj="xy", axis=0, xn=10, yn=10):
    result = 0
    if proj == "yz":
        result = double_integral(lambda y, z: f(surface(y, z), y, z) *
                                 np.sqrt(partial_derivative_2D_by_x(surface, [y, z])**2 +
                                         partial_derivative_2D_by_y(surface, [y, z])**2 + 1),
                                 range, f_lower, f_upper, axis, xn, yn)
    if proj == "xz":
        result = double_integral(lambda x, z: f(x, surface(x, z), z) *
                                 np.sqrt(partial_derivative_2D_by_x(surface, [x, z])**2 +
                                         partial_derivative_2D_by_y(surface, [x, z])**2 + 1),
                                 range, f_lower, f_upper, axis, xn, yn)
    if proj == "xy":
        result = double_integral(lambda x, y: f(x, y, surface(x, y)) *
                                 np.sqrt(partial_derivative_2D_by_x(surface, [x, y])**2 +
                                         partial_derivative_2D_by_y(surface, [x, y])**2 + 1),
                                 range, f_lower, f_upper, axis, xn, yn)

    # print(f(0, -1, 0))
    # print(f(1, -1, 0))
    # print(f(0, 0, 0))
    # print(f(0, 1, 0))
    # print(f(0, 2, 0))
    return result


def surface_integral_of_2form(p, q, r, surface, range, f_lower, f_upper, proj="xy", axis=0, xn=10, yn=10):
    # dsdx = partial_derivative_2D_by_x(surface, [-1.94, -0.40])
    # dsdy = partial_derivative_2D_by_y(surface, [-1.94, -0.40])
    # denominator = np.sqrt(dsdx**2 + dsdy**2 + 1)
    # i = -dsdx / denominator
    # j = -dsdy / denominator
    # k = 1 / denominator
    return surface_integral(lambda x, y, z:
                            p(x, y, z)*(-partial_derivative_2D_by_x(surface, [x, y]) /
                                        np.sqrt(partial_derivative_2D_by_x(surface, [x, y])**2 + partial_derivative_2D_by_y(surface, [x, y])**2 + 1)) +
                            q(x, y, z)*(-partial_derivative_2D_by_y(surface, [x, y]) /
                                        np.sqrt(partial_derivative_2D_by_x(surface, [x, y])**2 + partial_derivative_2D_by_y(surface, [x, y])**2 + 1)) +
                            r(x, y, z)*(1 /
                                        np.sqrt(partial_derivative_2D_by_x(surface, [x, y])**2 + partial_derivative_2D_by_y(surface, [x, y])**2 + 1)),
                            surface, range, f_lower, f_upper, proj, axis, xn, yn)
    # return surface_integral(lambda x, y, z:
    #                         p(x, y, z)*(partial_derivative_2D_by_x(surface, [y, z]) /
    #                                     np.sqrt(partial_derivative_2D_by_x(surface, [y, z])**2 + partial_derivative_2D_by_y(surface, [y, z])**2 + 1)) +
    #                         q(x, y, z)*(partial_derivative_2D_by_y(surface, [y, z]) /
    #                                     np.sqrt(partial_derivative_2D_by_x(surface, [y, z])**2 + partial_derivative_2D_by_y(surface, [y, z])**2 + 1)) +
    #                         r(x, y, z)*(-1 /
    #                                     np.sqrt(partial_derivative_2D_by_x(surface, [y, z])**2 + partial_derivative_2D_by_y(surface, [y, z])**2 + 1)),
    #                         surface, range, f_lower, f_upper, proj, axis, xn, yn)


def surface_integral_greens_rule(p, q, r, range, f1_lower, f1_upper, f2_lower, f2_upper, axis="xyz", xn=10, yn=10, zn=10):
    return triple_integral(lambda x, y, z: partial_derivative_3D_by_x(p, [x, y, z]) + partial_derivative_3D_by_y(q, [x, y, z]) +
                           partial_derivative_3D_by_z(r, [x, y, z]), range, f1_lower, f1_upper, f2_lower, f2_upper, axis, xn, yn, zn)

