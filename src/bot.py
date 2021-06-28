from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import Sequence, ControlStep
from util.vec import Vec3
import random
import math
from bot_data.bestbot import *

# My Files to import
from custom_functions import *
from options import *

findbestbot()


# Neural Network Variables
# Layer one wieghts and biases (field inputs to layer 1 of neurons)
if bestbot_fitness == 0
    input_count = 89
    output_count = 40
    wieght_layer_one = []
    bias_layer_one = []
    for i in range(0,output_count):
        list = []
        for j in range(0,input_count):
            a = random.random()
            list.append(a)
        wieght_layer_one.append(list)
        b = random.random()
        bias_layer_one.append(b)

    # Layer two wieghts and biases (layer 1 neurons to layer 2 neurons)
    input_count = 40
    output_count = 40
    wieght_layer_two = []
    bias_layer_two = []
    for i in range(0,output_count):
        list = []
        for j in range(0,input_count):
            a = random.random()
            list.append(a)
        wieght_layer_two.append(list)
        b = random.random()
        bias_layer_two.append(b)

    # Layer three wieghts and biases (layer 2 neurons to Outputs)
    input_count = 40
    output_count = 8
    wieght_layer_three = []
    bias_layer_three = []
    for i in range(0,output_count):
        list = []
        for j in range(0,input_count):
            a = random.random()
            list.append(a)
        wieght_layer_three.append(list)
        b = random.random()
        bias_layer_three.append(b)

# Fitness Function Global Variables
boost_tracker_over_game = []
dist_from_tm8_one = []
dist_from_tm8_two = []
data_save_check = False


class MyBot(BaseAgent):


    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.boost_pad_tracker = BoostPadTracker()

    def initialize_agent(self):
        # Set up information about the boost pads now that the game is active and the info is available
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())


    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """
        This function will be called by the framework many times per second. This is where you can
        see the motion of the ball, etc. and return controls to drive your car.
        """
        # Keep our boost pad info updated with which pads are currently active
        self.boost_pad_tracker.update_boost_status(packet)
        # This is good to keep at the beginning of get_output. It will allow you to continue
        # any sequences that you may have started during a previous call to get_output.
        if self.active_sequence is not None and not self.active_sequence.done:
            controls = self.active_sequence.tick(packet)
            if controls is not None:
                return controls

        # 3v3 team and opposite team finding
        if self.index > 2:
            teamates_index = [3, 4, 5]
            teamates_index.remove(self.index)
            opponents_index = [0, 1, 2]
        else:
            teamates_index = [0, 1, 2]
            teamates_index.remove(self.index)
            opponents_index = [3, 4, 5]

        #--------Variables-------------
        # My car
        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_velocity = Vec3(my_car.physics.velocity)
        car_pitch = my_car.physics.rotation.pitch
        car_yaw = my_car.physics.rotation.yaw
        car_roll = my_car.physics.rotation.roll
        car_boost = my_car.boost
        car_wheel_contact = tfconvert(my_car.has_wheel_contact)
        car_super_sonic = tfconvert(my_car.is_super_sonic)
        car_double_jumped = tfconvert(my_car.double_jumped)

        # teamates
        tm8_one = packet.game_cars[teamates_index[0]]
        tm8_one_location = Vec3(tm8_one.physics.location)
        tm8_one_velocity = Vec3(tm8_one.physics.velocity)
        tm8_one_pitch = tm8_one.physics.rotation.pitch
        tm8_one_yaw = tm8_one.physics.rotation.yaw
        tm8_one_roll = tm8_one.physics.rotation.roll
        tm8_one_boost = tm8_one.boost
        tm8_one_demoed = tfconvert(tm8_one.is_demolished)

        tm8_two = packet.game_cars[teamates_index[1]]
        tm8_two_location = Vec3(tm8_two.physics.location)
        tm8_two_velocity = Vec3(tm8_two.physics.velocity)
        tm8_two_pitch = tm8_two.physics.rotation.pitch
        tm8_two_yaw = tm8_two.physics.rotation.yaw
        tm8_two_roll = tm8_two.physics.rotation.roll
        tm8_two_boost = tm8_two.boost
        tm8_two_demoed = tfconvert(tm8_two.is_demolished)

        # Opponents
        op1_car = packet.game_cars[opponents_index[0]]
        op1_location = Vec3(op1_car.physics.location)
        op1_velocity = Vec3(op1_car.physics.velocity)
        op1_pitch = op1_car.physics.rotation.pitch
        op1_yaw = op1_car.physics.rotation.yaw
        op1_roll = op1_car.physics.rotation.roll

        op2_car = packet.game_cars[opponents_index[1]]
        op2_location = Vec3(op2_car.physics.location)
        op2_velocity = Vec3(op2_car.physics.velocity)
        op2_pitch = op2_car.physics.rotation.pitch
        op2_yaw = op2_car.physics.rotation.yaw
        op2_roll = op2_car.physics.rotation.roll

        op3_car = packet.game_cars[opponents_index[2]]
        op3_location = Vec3(op3_car.physics.location)
        op3_velocity = Vec3(op3_car.physics.velocity)
        op3_pitch = op3_car.physics.rotation.pitch
        op3_yaw = op3_car.physics.rotation.yaw
        op3_roll = op3_car.physics.rotation.roll

        # Game ball Variables
        ball_velocity = Vec3(packet.game_ball.physics.velocity)
        ball_location = Vec3(packet.game_ball.physics.location)

        # Distances from an object to a place
        my_car_to_ball_dist = car_location.dist(ball_location)
        tm8_one_to_ball_dist = tm8_one_location.dist(ball_location)
        tm8_two_to_ball_dist = tm8_two_location.dist(ball_location)
        op1_car_to_ball_dist = op1_location.dist(ball_location)
        op2_car_to_ball_dist = op2_location.dist(ball_location)
        op3_car_to_ball_dist = op3_location.dist(ball_location)

        my_car_to_tm8_one_dist = car_location.dist(tm8_one_location)
        my_car_to_tm8_two_dist = car_location.dist(tm8_two_location)
        my_car_to_op1_dist = car_location.dist(op1_location)
        my_car_to_op2_dist = car_location.dist(op2_location)
        my_car_to_op3_dist = car_location.dist(op3_location)
        tm8_one_to_tm8_two_dist = tm8_one_location.dist(tm8_two_location)
        tm8_one_to_op1_dist = tm8_one_location.dist(op1_location)
        tm8_one_to_op2_dist = tm8_one_location.dist(op2_location)
        tm8_one_to_op3_dist = tm8_one_location.dist(op3_location)
        tm8_two_to_op1_dist = tm8_two_location.dist(op1_location)
        tm8_two_to_op2_dist = tm8_two_location.dist(op2_location)
        tm8_two_to_op3_dist = tm8_two_location.dist(op3_location)
        op1_to_op2_dist = op1_location.dist(op2_location)
        op1_to_op3_dist = op1_location.dist(op3_location)
        op2_to_op3_dist = op2_location.dist(op3_location)

        # Other Variables
        controls = SimpleControllerState()

        # Putting the Inputs into a list for the Nerual Network
        info = [
            car_location.x, car_location.y, car_location.z,
            car_velocity.x, car_velocity.y, car_velocity.z,
            car_pitch, car_yaw, car_roll, car_boost, car_wheel_contact, car_super_sonic, car_double_jumped,

            tm8_one_location.x, tm8_one_location.y, tm8_one_location.z,
            tm8_one_velocity.x, tm8_one_velocity.y, tm8_one_velocity.z,
            tm8_one_pitch, tm8_one_yaw, tm8_one_roll, tm8_one_boost, tm8_one_demoed,

            tm8_two_location.x, tm8_two_location.y, tm8_two_location.z,
            tm8_two_velocity.x, tm8_two_velocity.y, tm8_two_velocity.z,
            tm8_two_pitch, tm8_two_yaw, tm8_two_roll, tm8_two_boost, tm8_two_demoed,

            op1_location.x, op1_location.y, op1_location.z,
            op1_velocity.x, op1_velocity.y, op1_velocity.z,
            op1_pitch, op1_yaw, op1_roll,

            op2_location.x, op2_location.y, op2_location.z,
            op2_velocity.x, op2_velocity.y, op2_velocity.z,
            op2_pitch, op2_yaw, op2_roll,

            op3_location.x, op3_location.y, op3_location.z,
            op3_velocity.x, op3_velocity.y, op3_velocity.z,
            op3_pitch, op3_yaw, op3_roll,

            ball_location.x, ball_location.y, ball_location.z,
            ball_velocity.x, ball_velocity.y, ball_velocity.z,

            my_car_to_ball_dist, tm8_one_to_ball_dist, tm8_two_to_ball_dist,
            op1_car_to_ball_dist, op2_car_to_ball_dist, op3_car_to_ball_dist,

            my_car_to_tm8_one_dist, my_car_to_tm8_two_dist, my_car_to_op1_dist,
            my_car_to_op2_dist, my_car_to_op3_dist, tm8_one_to_tm8_two_dist,
            tm8_one_to_op1_dist, tm8_one_to_op2_dist, tm8_one_to_op3_dist,
            tm8_two_to_op1_dist, tm8_two_to_op2_dist, tm8_two_to_op3_dist,
            op1_to_op2_dist, op1_to_op3_dist, op2_to_op3_dist
        ]
        # Neural Network Calulation
        layer_two_inputs = hidden_layer_calculator(info, wieght_layer_one, bias_layer_one)
        layer_three_inputs = hidden_layer_calculator(layer_two_inputs, wieght_layer_two, bias_layer_two)
        controller_outputs = hidden_layer_calculator(layer_three_inputs, wieght_layer_three, bias_layer_three)

        # Fitness function information gathering
        boost_tracker_over_game.append(car_boost)
        dist_from_tm8_one.append(my_car_to_tm8_one_dist)
        dist_from_tm8_two.append(my_car_to_tm8_two_dist)

        # Saving Data to a file and calculate fitness
        if packet.game_info.is_match_ended == True:
            global data_save_check
            data_to_save =[
                wieght_layer_one,
                wieght_layer_two,
                wieght_layer_three,
                bias_layer_one,
                bias_layer_two,
                bias_layer_three
            ]
            datastoreandfitcal(packet, self, boost_tracker_over_game, dist_from_tm8_one, dist_from_tm8_two, my_car, data_to_save, data_save_check)
            data_save_check = True


        # Returning Outputs from the Neural Network
        controls.throttle = (controller_outputs[0]*2)-1
        controls.steer = (controller_outputs[1]*2)-1
        controls.pitch = (controller_outputs[2]*2)-1
        controls.yaw = (controller_outputs[3]*2)-1
        controls.roll = (controller_outputs[4]*2)-1

        if controller_outputs[5] > .5:
            controls.boost = True
        else:
            controls.boost = False

        if controller_outputs[6] > .5:
            controls.hand_break = True
        else:
            controls.hand_break = False

        if controller_outputs[7] > .5:
            controls.jump = True
        else:
            controls.jump = False

        return controls

    def begin_front_flip(self, packet):
        # Send some quickchat just for fun
        # self.send_quick_chat(team_only=False, quick_chat=QuickChatSelection.Information_IGotIt)

        # Do a front flip. We will be committed to this for a few seconds and the bot will ignore other
        # logic during that time because we are setting the active_sequence.
        self.active_sequence = Sequence([
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.2, controls=SimpleControllerState(jump=True, pitch=-1)),
            ControlStep(duration=1, controls=SimpleControllerState()),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)
