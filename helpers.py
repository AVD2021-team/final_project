from math import sqrt
import numpy as np
import transforms3d

def transform_world_to_ego_frame(pos, ego, ego_rpy):
    loc = np.array(pos) - np.array(ego)
    r = transforms3d.euler.euler2mat(ego_rpy[0], ego_rpy[1], ego_rpy[2]).T
    loc_relative = np.dot(r, loc)
    return loc_relative

# Using d = (v_f^2 - v_i^2) / (2 * a), compute the distance
# required for a given acceleration/deceleration.
def calc_distance(v_i, v_f, a):
    """Computes the distance given an initial and final speed, with a constant
    acceleration.

    args:
        v_i: initial speed (m/s)
        v_f: final speed (m/s)
        a: acceleration (m/s^2)
    returns:
        d: the final distance (m)
    """

    return (v_f * v_f - v_i * v_i) / 2 / a


# Using v_f = sqrt(v_i^2 + 2ad), compute the final speed for a given
# acceleration across a given distance, with initial speed v_i.
# Make sure to check the discriminant of the radical. If it is negative,
# return zero as the final speed.
def calc_final_speed(v_i, a, d):
    """Computes the final speed given an initial speed, distance travelled,
    and a constant acceleration.

    args:
        v_i: initial speed (m/s)
        a: acceleration (m/s^2)
        d: distance to be travelled (m)
    returns:
        v_f: the final speed (m/s)
    """
    pass

    temp = v_i * v_i + 2 * d * a
    if temp < 0:
        return 0.0000001
    else:
        return sqrt(temp)
