#!/usr/bin/env python3
import sys
import os
import numpy as np
import math

# Script level imports
sys.path.append(os.path.realpath(os.path.dirname(__file__)))
from behavioural_planner_state import BehaviouralPlannerState, FollowLaneState

sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), '..'))
from helpers import calc_distance

# State machine states
# FOLLOW_LANE = 0
# DECELERATE_TO_STOP = 1
# STAY_STOPPED = 2


class BehaviouralPlanner:
    def __init__(self, lookahead, lead_vehicle_lookahead, a_max):
        self._lookahead = lookahead
        self._follow_lead_vehicle_lookahead = lead_vehicle_lookahead
        self._state = FollowLaneState(self)
        self._follow_lead_vehicle = False
        self._obstacle_on_lane = False
        self._pedestrian_on_lane = True
        self._goal_state = [0.0, 0.0, 0.0]
        self._goal_index = 0
        self._stop_count = 0
        self._lookahead_collision_index = 0
        self._a_max = a_max
        self._emergency_brake_distance = 0

    @property
    def lookahead(self):
        return self._lookahead

    @property
    def pedestrian_on_lane(self):
        return self._pedestrian_on_lane

    @property
    def a_max(self):
        return self._a_max

    @property
    def emergency_brake_distance(self):
        return self._emergency_brake_distance

    @pedestrian_on_lane.setter
    def pedestrian_on_lane(self, pedestrian_on_lane):
        self._pedestrian_on_lane = pedestrian_on_lane

    @emergency_brake_distance.setter
    def emergency_brake_distance(self, emergency_brake_distance):
        self._emergency_brake_distance = emergency_brake_distance

    def set_lookahead(self, lookahead):
        self._lookahead = lookahead

    # Handles state transitions and computes the goal state.
    def transition_state(self, waypoints, ego_state, closed_loop_speed, pedestrian_states):
        self._state.transition_state(waypoints, ego_state, closed_loop_speed, pedestrian_states)

    def transition_to(self, state: BehaviouralPlannerState):
        """
        The Context allows changing the State object at runtime.
        """
        print(f"Behavioural Planner: Transition to {type(state).__name__}")
        self._state = state
        self._state.context = self

    # Gets the goal index in the list of waypoints, based on the lookahead and
    # the current ego state. In particular, find the earliest waypoint that has accumulated
    # arc length (including closest_len) that is greater than or equal to self._lookahead.
    def get_goal_index(self, waypoints, ego_state, closest_len, closest_index):
        """Gets the goal index for the vehicle. 
        
        Set to be the earliest waypoint that has accumulated arc length
        accumulated arc length (including closest_len) that is greater than or
        equal to self._lookahead.

        args:
            waypoints: current waypoints to track. (global frame)
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]: 
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
        returns:
            wp_index: Goal index for the vehicle to reach
                i.e. waypoints[wp_index] gives the goal waypoint
        """
        # Find the farthest point along the path that is within the
        # lookahead distance of the ego vehicle.
        # Take the distance from the ego vehicle to the closest waypoint into
        # consideration.
        arc_length = closest_len
        wp_index = closest_index

        # In this case, reaching the closest waypoint is already far enough for
        # the planner.  No need to check additional waypoints.
        if arc_length > self._lookahead:
            return wp_index

        # We are already at the end of the path.
        if wp_index == len(waypoints) - 1:
            return wp_index

        # Otherwise, find our next waypoint.
        while wp_index < len(waypoints) - 1:
            arc_length += np.sqrt((waypoints[wp_index][0] - waypoints[wp_index + 1][0]) ** 2 + (
                    waypoints[wp_index][1] - waypoints[wp_index + 1][1]) ** 2)
            if arc_length > self._lookahead:
                break
            wp_index += 1

        return wp_index % len(waypoints)

    # Checks to see if we need to modify our velocity profile to accommodate the
    # lead vehicle.
    def check_for_lead_vehicle(self, ego_state, lead_car_position, lead_car_speed):
        """Checks for lead vehicle within the proximity of the ego car, such
        that the ego car should begin to follow the lead vehicle.

        args:
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            lead_car_position: The [x, y] position of the lead vehicle.
                Lengths are in meters, and it is in the global frame.
        sets:
            self._follow_lead_vehicle: Boolean flag on whether the ego vehicle
                should follow (true) the lead car or not (false).
        """
        # Check lead car position delta vector relative to heading, as well as
        # distance, to determine if car should be followed.
        # Check to see if lead vehicle is within range, and is ahead of us.
        if not self._follow_lead_vehicle:
            # Compute the angle between the normalized vector between the lead vehicle
            # and ego vehicle position with the ego vehicle's heading vector.
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0], lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)
            # In this case, the car is too far away.
            if lead_car_distance > self._follow_lead_vehicle_lookahead + calc_distance(ego_state[3], lead_car_speed, -self._a_max):
                return

            lead_car_delta_vector = np.divide(lead_car_delta_vector,
                                              lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]),
                                  math.sin(ego_state[2])]
            # Check to see if the relative angle between the lead vehicle and the ego
            # vehicle lies within +/- 45 degrees of the ego vehicle's heading.
            if np.dot(lead_car_delta_vector,
                      ego_heading_vector) < (1 / math.sqrt(2)):
                return

            self._follow_lead_vehicle = True

        else:
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0], lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)

            # Add a 5m buffer to prevent oscillations for the distance check.
            if lead_car_distance > 10 + self._follow_lead_vehicle_lookahead + calc_distance(ego_state[3], lead_car_speed, -self._a_max):
                self._follow_lead_vehicle = False
                return
            # Check to see if the lead vehicle is still within the ego vehicle's
            # frame of view.
            lead_car_delta_vector = np.divide(lead_car_delta_vector, lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]), math.sin(ego_state[2])]
            if np.dot(lead_car_delta_vector, ego_heading_vector) > (1 / math.sqrt(2)):
                return

            self._follow_lead_vehicle = False

    # Compute the waypoint index that is closest to the ego vehicle, and return
    # it as well as the distance from the ego vehicle to that waypoint.
    @staticmethod
    def get_closest_index(waypoints, ego_state):
        """Gets closest index a given list of waypoints to the vehicle position.

        args:
            waypoints: current waypoints to track. (global frame)
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]:
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)

        returns:
            [closest_len, closest_index]:
                closest_len: length (m) to the closest waypoint from the vehicle.
                closest_index: index of the waypoint which is closest to the vehicle.
                    i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
        """
        closest_len = float('Inf')
        closest_index = 0

        for i in range(len(waypoints)):
            temp = (waypoints[i][0] - ego_state[0]) ** 2 + (waypoints[i][1] - ego_state[1]) ** 2
            if temp < closest_len:
                closest_len = temp
                closest_index = i
        closest_len = np.sqrt(closest_len)

        return closest_len, closest_index


# Checks if p2 lies on segment p1-p3, if p1, p2, p3 are collinear.
def point_on_segment(p1, p2, p3):
    if min(p1[0], p3[0]) <= p2[0] <= max(p1[0], p3[0]) and min(p1[1], p3[1]) <= p2[1] <= max(p1[1], p3[1]):
        return True
    else:
        return False


def compute_stop_distance(speed):
    """Computes the stop distance given the speed."""
    return(speed * 3.6 / 10) ** 2
