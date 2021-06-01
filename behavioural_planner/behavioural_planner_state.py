from abc import ABC, abstractmethod
from traffic_light_detector import TrafficLightState
import numpy as np
import sys
import os

# Script level imports
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), '..'))
from helpers import calc_distance, transform_world_to_ego_frame


# Stop speed threshold
STOP_THRESHOLD = 0.02
# Number of cycles before moving from stop sign.
STOP_COUNTS = 10

# Distance from intersection where we spot the stop line
DIST_SPOT_INTER = 15 # meters
DIST_STOP_INTER = 3.5 # meters

# Radius on the Y axis for the Intersection ahead check. The check on the X axis
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
    def transition_state(self, waypoints, ego_state, closed_loop_speed, pedestrian_states):
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
        variables to set:
            self._goal_index: Goal index for the vehicle to reach
                i.e. waypoints[self._goal_index] gives the goal waypoint
            self._goal_state: Goal state for the vehicle to reach (global frame)
                format: [x_goal, y_goal, v_goal]
            self._state: The current state of the vehicle.
                available states:
                    FollowLaneS         : Follow the global waypoints (lane).
                    DecelerateToStop     : Decelerate to stop.
                    StayStopped        : Stay stopped.
            self._stop_count: Counter used to count the number of cycles which
                the vehicle was in the StayStopped state so far.
        useful_constants:
            STOP_THRESHOLD  : Stop speed threshold (m). The vehicle should fully
                              stop when its speed falls within this threshold.
            STOP_COUNTS     : Number of cycles (simulation iterations)
                              before moving from stop sign.
        """
        pass

    def _get_new_goal(self, waypoints, ego_state):
        # First, find the closest index to the ego vehicle.
        closest_len, closest_index = self.context.get_closest_index(waypoints, ego_state)

        # Next, find the goal index that lies within the lookahead distance along the waypoints.
        goal_index = self.context.get_goal_index(waypoints, ego_state, closest_len, closest_index)
        while goal_index < len(waypoints) and waypoints[goal_index][2] <= 0.1:
            goal_index += 1

        # Update goal
        self.context._goal_index = goal_index
        self.context._goal_state = waypoints[goal_index]
        return goal_index

    def _get_intersection_goal(self, waypoints, ego_state):

        closest_len, closest_index = self.context.get_closest_index(waypoints, ego_state)
        goal_index = self.context.get_goal_index(waypoints, ego_state, closest_len, closest_index)

        intersection_lines = self.context.get_intersection_lines()

        for i in range(closest_index, goal_index):
            for inter in intersection_lines:
                car_loc_relative = transform_world_to_ego_frame([inter[0], inter[1], inter[2]],
                                                                [ego_state[0], ego_state[1], 0.0],
                                                                [0.0, 0.0, ego_state[2]])
                dist_spot = np.linalg.norm(np.array([waypoints[i][0] - inter[0], waypoints[i][1] - inter[1]]))
                if dist_spot < DIST_SPOT_INTER:
                    if car_loc_relative[0] > 0 and -RELATIVE_DIST_INTER_Y <= car_loc_relative[1] <= RELATIVE_DIST_INTER_Y:
                        print(f"Intersection ahead. Position: {car_loc_relative}")
                        for j in range(i, len(waypoints)):
                            dist_stop = np.linalg.norm(
                                np.array([waypoints[j][0] - inter[0], waypoints[j][1] - inter[1]]))

                            if dist_stop < DIST_STOP_INTER:
                                print(f"Stop Waypoint: {j - 1} {waypoints[j - 1]}")
                                return j - 1
                    else:
                        print(f"Intersection behind, ignored. Position: {car_loc_relative}")
                        return None
        return None


class FollowLaneState(BehaviouralPlannerState):
    """
    In this state, continue tracking the lane by finding the
    goal index in the waypoint list that is within the lookahead
    distance. Then, check to see if the waypoint path intersects
    with any stop lines. If it does, then ensure that the goal
    state enforces the car to be stopped before the stop line.
    You should use the get_closest_index(), get_goal_index(), and
    check_for_stop_signs() helper functions.
    Make sure that get_closest_index() and get_goal_index() functions are
    complete, and examine the check_for_stop_signs() function to
    understand it.
    """
    def __init__(self, context):
        super().__init__(context)
        self.name = "FollowLane"

    def transition_state(self, waypoints, ego_state, closed_loop_speed, pedestrian_states):
        # print("FOLLOW_LANE")
        self.context.emergency_brake_distance = calc_distance(closed_loop_speed, 0, -self.context.a_max)

        if self.context.pedestrian_on_lane:
            # First, find the closest index to the ego vehicle.
            goal_index = self._get_new_goal(waypoints, ego_state)
            # Update goal
            self.context.update_goal(waypoints, goal_index, 0)
            self.context.transition_to(EmergencyStopState(self.context))
        else:
            goal_index = self._get_new_goal(waypoints, ego_state)

            intersection_goal = None
            if self.context.tl_state == TrafficLightState.STOP:
                intersection_goal = self._get_intersection_goal(waypoints, ego_state)

            if intersection_goal is not None:
                self.context.update_goal(waypoints, intersection_goal, 0)
                self.context.transition_to(DecelerateToStopState(self.context))
            else:
                self.context.update_goal(waypoints, goal_index)


class DecelerateToStopState(BehaviouralPlannerState):
    """
    In this state, check if we have reached a complete stop. Use the
    closed loop speed to do so, to ensure we are actually at a complete
    stop, and compare to STOP_THRESHOLD.  If so, transition to the next
    state.
    """
    def __init__(self, context):
        super().__init__(context)
        self.name = "DecelerateToStop"

    def transition_state(self, waypoints, ego_state, closed_loop_speed, pedestrian_states):
        # print("DECELERATE_TO_STOP")
        if self.context.pedestrian_on_lane:
            self.context.transition_to(EmergencyStopState(self.context))
        # If the traffic light is green or has disappeared, transition to Follow lane
        elif self.context.tl_state in (TrafficLightState.GO, TrafficLightState.NO_TL):
            self.context.transition_to(FollowLaneState(self.context))


class StayStoppedState(BehaviouralPlannerState):
    """
    In this state, check to see if we have stayed stopped for at
    least STOP_COUNTS number of cycles. If so, we can now leave
    the stop sign and transition to the next state.
    """
    def __init__(self, context):
        super().__init__(context)
        self.name = "StayStopped"

    def transition_state(self, waypoints, ego_state, closed_loop_speed, pedestrian_states):
        # print("STAY_STOPPED")
        # We have stayed stopped for the required number of cycles.
        # Allow the ego vehicle to leave the stop sign. Once it has
        # passed the stop sign, return to lane following.
        # You should use the get_closest_index(), get_goal_index(), and
        # check_for_stop_signs() helper functions.
        # We've stopped for the required amount of time, so the new goal
        # index for the stop line is not relevant. Use the goal index
        # that is the lookahead distance away.

        goal_index = self._get_new_goal(waypoints, ego_state)
        self.context.update_goal(waypoints, goal_index)

        # If the stop sign is no longer along our path, we can now
        # transition back to our lane following state.

        self.context.transition_to(FollowLaneState(self.context))


class EmergencyStopState(BehaviouralPlannerState):

    def __init__(self, context):
        super().__init__(context)
        self.name = "EmergencyStop"

    def transition_state(self, waypoints, ego_state, closed_loop_speed, pedestrian_states):
        if not self.context.pedestrian_on_lane:
            self.context.transition_to(FollowLaneState(self.context))

