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
