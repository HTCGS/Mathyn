import numpy as np
from mathyn.differentiation import *
from mathyn.linear_algebra import vector_magnitude_3D, vector_dot_3D, vector_cross_3D
from mathyn.integration import integral, double_integral

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
    return lambda x: f(a) + derivative_at_point(f, a, dx) * (x - a)


def normal_line_of_curve(f, a, dx=0.01):
    df = derivative_at_point(f, a, dx)
    if abs(df) == 0:
        return lambda x: f(a) * x
    return lambda x: f(a) + (-1 * np.power(derivative_at_point(f, a, dx), -1)) * (x - a)


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
    di = derivative_at_point(fi, point)
    dj = derivative_at_point(fj, point)
    dk = derivative_at_point(fk, point)
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
    return integral(lambda t: np.sqrt(derivative_at_point(fi, t)**2 + derivative_at_point(fj, t)**2 + derivative_at_point(fk, t)**2), a, b, n)


def curvature(points):
    dfdx = derivative_by_points(points, 0.01)
    d2fdx = derivative_by_points(dfdx, 0.01)
    dfdx = dfdx[1:-1]
    return d2fdx / np.power(np.sqrt(1 + dfdx*dfdx), 3)


def line_integral(f, a, b, line):
    return integral(lambda x: f(x, line(x)) * np.sqrt(1 + derivative_at_point(line, x)**2), a, b)


def line_integral_3D(f, a, b, xt, yt, zt):
    return integral(lambda t: f(xt(t), yt(t), zt(t)) *
                        np.sqrt(derivative_at_point(xt, t)**2 + derivative_at_point(yt, t)**2 + derivative_at_point(zt, t)**2), a, b)


def line_integral_by_parametric_curve(f, a, b, xt, yt):
    return integral(lambda t: f(xt(t), yt(t)) * np.sqrt(derivative_at_point(xt, t)**2 + derivative_at_point(yt, t)**2), a, b)


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

