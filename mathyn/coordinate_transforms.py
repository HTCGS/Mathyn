def convert_to_polar_function(f):
    return lambda r, phi: f(r*np.cos(phi), r*np.sin(phi))


def convert_to_cylindrical_function(f):
    return lambda r, phi, z: f(r*np.cos(phi), r*np.sin(phi), z)


def convert_to_spherical_function(f):
    return lambda r, t, phi: f(r*np.sin(t)*np.cos(phi), r*np.sin(t)*np.sin(phi), r*np.cos(t))


def swap_functions_parameters_2D(f):
    return lambda x, y: f(y, x)


def swap_functions_parameters_3D(f, axis):
    if axis == "xzy":
        return lambda x, y, z: f(x, z, y)
    if axis == "yxz":
        return lambda x, y, z: f(y, x, z)
    if axis == "yzx":
        return lambda x, y, z: f(y, z, x)
    if axis == "zxy":
        return lambda x, y, z: f(z, x, y)
    if axis == "zyx":
        return lambda x, y, z: f(z, y, x)
