# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import traceback
import omni.graph.core as og
from isaacsim.core.nodes import BaseResetNode
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.wheeled_robots.controllers.ackermann_controller import AckermannController
from isaacsim.robot.wheeled_robots.ogn.OgnAckermannControllerDatabase import OgnAckermannControllerDatabase


class OgnAckermannControllerInternalState(BaseResetNode):
    def __init__(self):
        self.wheel_base = 0.0
        self.track_width = 0.0
        self.front_wheel_radius = 0.0
        self.back_wheel_radius = 0.0
        self.max_wheel_velocity = 1.0e20
        self.invert_steering = False
        self.max_wheel_rotation_angle = 6.28
        self.max_acceleration = 0.0
        self.max_steering_angle_velocity = 0.0

        self.node = None
        self.graph_id = None
        super().__init__(initialize=False)

    def initialize_controller(self) -> None:
        self.controller_handle = AckermannController(
            name="ackermann_controller",
            wheel_base=self.wheel_base,
            track_width=self.track_width,
            front_wheel_radius=self.front_wheel_radius,
            back_wheel_radius=self.back_wheel_radius,
            max_wheel_velocity=self.max_wheel_velocity,
            invert_steering=self.invert_steering,
            max_wheel_rotation_angle=self.max_wheel_rotation_angle,
            max_acceleration=self.max_acceleration,
            max_steering_angle_velocity=self.max_steering_angle_velocity,
        )

        self.initialized = True

    def forward(self, command: np.ndarray) -> ArticulationAction:
        return self.controller_handle.forward(command)

    def custom_reset(self):
        self.node.get_attribute("outputs:wheelAngles").set([0.0, 0.0])
        self.node.get_attribute("outputs:wheelRotationVelocity").set([0.0, 0.0, 0.0, 0.0])


class OgnAckermannController:
    @staticmethod
    def init_instance(node, graph_instance_id):
        state = OgnAckermannControllerDatabase.get_internal_state(node, graph_instance_id)
        state.node = node
        state.graph_id = graph_instance_id

    @staticmethod
    def internal_state():
        return OgnAckermannControllerInternalState()

    @staticmethod
    def _convert_to_numpy_array(data):
        """Convert tuple/list of numpy scalars or any sequence to numpy array."""
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return data
        # Convert tuple/list (including numpy scalars) to numpy array
        return np.array(data, dtype=np.float64)

    @staticmethod
    def compute(db) -> bool:
        state = db.per_instance_state

        try:
            if not state.initialized:

                state.wheel_base = db.inputs.wheelBase
                state.track_width = db.inputs.trackWidth
                state.front_wheel_radius = db.inputs.frontWheelRadius
                state.back_wheel_radius = db.inputs.backWheelRadius
                state.max_wheel_velocity = db.inputs.maxWheelVelocity
                state.invert_steering = db.inputs.invertSteering
                state.max_wheel_rotation_angle = db.inputs.maxWheelRotation
                state.max_acceleration = db.inputs.maxAcceleration
                state.max_steering_angle_velocity = db.inputs.maxSteeringAngleVelocity

                state.initialize_controller()

            actions = state.forward(
                [
                    db.inputs.steeringAngle,
                    db.inputs.steeringAngleVelocity,
                    db.inputs.speed,
                    db.inputs.acceleration,
                    db.inputs.dt,
                ]
            )

            if actions.joint_velocities is not None:
                try:
                    # Convert to numpy array for OmniGraph compatibility
                    converted_velocities = OgnAckermannController._convert_to_numpy_array(actions.joint_velocities)
                    db.outputs.wheelRotationVelocity = converted_velocities
                except Exception as vel_error:
                    # some input data from upstream may have data type mismatch, ignore for now
                    pass

            if actions.joint_positions is not None:
                try:
                    # Convert to numpy array for OmniGraph compatibility
                    converted_positions = OgnAckermannController._convert_to_numpy_array(actions.joint_positions)
                    db.outputs.wheelAngles = converted_positions
                except Exception as pos_error:
                    # some input data from upstream may have data type mismatch, ignore for now
                    pass

        except Exception as error:
            db.log_warning(f"Error in compute: {error}\n{traceback.format_exc()}")
            return False

        db.outputs.execOut = og.ExecutionAttributeState.ENABLED

        return True

    @staticmethod
    def release_instance(node, graph_instance_id):
        try:
            state = OgnAckermannControllerDatabase.get_internal_state(node, graph_instance_id)
        except Exception:
            state = None
            pass

        if state is not None:
            state.reset()
            state.initialized = False
