from abc import ABC, abstractmethod
from traffic_light_detector import TrafficLightState
import numpy as np
import sys
import os

# Script level imports
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), '..'))
from helpers import calc_distance, transform_world_to_ego_frame

# Distance from intersection where we spot the stop line
DIST_SPOT_INTER = 15 # meters
# Minimum distance from the intersection to which we want to stop for a RED light
DIST_STOP_INTER = 3.5 # meters

# Radius on the Y axis for the Intersection ahead check.
RELATIVE_DIST_INTER_Y = 3.5


class BehaviouralPlannerState(ABC):

    __slots__ = '_context','name'

    def __init__(self, context):
        self._context = context

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, context):
        self._context = context

    @abstractmethod
    def transition_state(self, waypoints, ego_state, closed_loop_speed):
        """Handles state transitions and computes the goal state.

        args:
            waypoints: current waypoints to track (global frame).
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
            closed_loop_speed: current (closed-loop) speed for vehicle (m/s)
        """
        pass

    def _get_new_goal(self, waypoints, ego_state):
        # First, find the closest index to the ego vehicle.
        closest_len, closest_index = self.context.get_closest_index(waypoints, ego_state)

        # Next, find the goal index that lies within the lookahead distance along the waypoints.
        goal_index = self.context.get_goal_index(waypoints, ego_state, closest_len, closest_index)
        while goal_index < (len(waypoints) - 1) and waypoints[goal_index][2] <= 0.1:
            goal_index += 1

        return goal_index

    def _get_intersection_goal(self, waypoints, ego_state):
        """
        Checks whether the vehicle is near an intersection and returns a waypoint index.
        Returns None if theres no intersection to be managed.
        The check is done 15 meters from the intersection. If so, the one that is located
        at least 3.5 meters from the intersection is chosen as the target waypoint so it is possible
        to stop at an acceptable distance.

        Args:
            waypoints: list of the waypoints on the path
            ego_state: (x, y, yaw, current_speed) of the vehicle

        Returns:
            goal_index: index of the waypoint target
        """

        # We get the closest index and the goal index, so to get the current path
        closest_len, closest_index = self.context.get_closest_index(waypoints, ego_state)
        goal_index = self.context.get_goal_index(waypoints, ego_state, closest_len, closest_index)

        # We get the intersections positions in the world frame
        intersection_lines = self.context.get_intersection_lines()

        # For each waypoint from the closest to the goal we want to check if there
        # is an intersection that we have to manage
        for i in range(closest_index, goal_index):
            for inter in intersection_lines:
                # We project the intersection onto the ego frame
                inter_loc_relative = transform_world_to_ego_frame(
                    [inter[0], inter[1], inter[2]],
                    [ego_state[0], ego_state[1], 0.0],
                    [0.0, 0.0, ego_state[2]]
                )
                # We calculate the distance between the current waypoint and the current intersection
                dist_spot = np.linalg.norm(np.array([waypoints[i][0] - inter[0], waypoints[i][1] - inter[1]]))
                # If this distance is smaller than DIST_SPOT_INTER, we spot the intersection
                if dist_spot < DIST_SPOT_INTER:
                    # But we also check if it is ahead of ego or behind. If ahead, we choose a stop waypoint.
                    if inter_loc_relative[0] > 0 and -RELATIVE_DIST_INTER_Y <= inter_loc_relative[1] <= RELATIVE_DIST_INTER_Y:
                        print(f"Intersection ahead. Position: {inter_loc_relative}")
                        for j in range(i, len(waypoints)):
                            dist_stop = np.linalg.norm(
                                np.array([waypoints[j][0] - inter[0], waypoints[j][1] - inter[1]]))

                            if dist_stop < DIST_STOP_INTER:
                                print(f"Stop Waypoint: {j - 1} {waypoints[j - 1]}")
                                return j - 1
                    # Otherwise we stop checking.
                    else:
                        print(f"Intersection behind, ignored. Position: {inter_loc_relative}")
                        return None
        return None


class FollowLaneState(BehaviouralPlannerState):
    """
    In this state, check if it is needed to stop for a light or for
    an obstacle, otherwise continue tracking the lane by finding the
    goal index in the waypoint list that is within the lookahead
    distance.
    """
    def __init__(self, context):
        super().__init__(context)
        self.name = "FollowLane"

    def transition_state(self, waypoints, ego_state, closed_loop_speed):
        # print("FOLLOW_LANE")
        self.context.emergency_brake_distance = calc_distance(closed_loop_speed, 0, -self.context.a_max)
        # First, find the closest index to the ego vehicle.
        goal_index = self._get_new_goal(waypoints, ego_state)

        if self.context.obstacle_on_lane:
            # Update goal with speed 0
            self.context.update_goal(waypoints, goal_index, 0)
            self.context.transition_to(EmergencyStopState(self.context))
        else:
            # Get the intersection goal if needed
            intersection_goal = None
            if self.context.tl_state == TrafficLightState.STOP:
                intersection_goal = self._get_intersection_goal(waypoints, ego_state)

            # Then set the right goal given the sensor data.
            if intersection_goal is not None:
                self.context.update_goal(waypoints, intersection_goal, 0)
                self.context.transition_to(DecelerateToStopState(self.context))
            else:
                self.context.update_goal(waypoints, goal_index)


class DecelerateToStopState(BehaviouralPlannerState):
    """
    In this state, check the state of the light and act consequently.
    If the light is Green or there is no light, transition to Follow Lane.
    If there is an obstacle on the lane, transition to Emergency Stop.
    """
    def __init__(self, context):
        super().__init__(context)
        self.name = "DecelerateToStop"

    def transition_state(self, waypoints, ego_state, closed_loop_speed):
        # print("DECELERATE_TO_STOP")
        if self.context.obstacle_on_lane:
            self.context.transition_to(EmergencyStopState(self.context))
        # If the traffic light is green or has disappeared, transition to Follow lane
        elif self.context.tl_state in (TrafficLightState.GO, TrafficLightState.NO_TL):
            self.context.transition_to(FollowLaneState(self.context))


class EmergencyStopState(BehaviouralPlannerState):
    """
    This state is activated when a very close pedestrian is detected along our trajectory.
    The goal is to stop immediately to avoid running over a pawn.
    This state remains until the pedestrian is no longer detected.
    """

    def __init__(self, context):
        super().__init__(context)
        self.name = "EmergencyStop"

    def transition_state(self, waypoints, ego_state, closed_loop_speed):
        if not self.context.obstacle_on_lane:
            self.context.transition_to(FollowLaneState(self.context))

