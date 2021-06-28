import math
from bot_data.bot0 import *
from bot_data.bot1 import *
from bot_data.bot2 import *
from bot_data.bot3 import *
from bot_data.bot4 import *
from bot_data.bot5 import *
from bot_data.bestbot import *
"""
Gives the Values of the neural network hidden layers
Make sure that Number of Inputs == the number of colums of wieght matrix
Make sure that Number of outputs == the Number of rows of the wieght and bias matrix
"""
def hidden_layer_calculator(inputs,wieght,bias):
    output = []
    count = 0
    for list in wieght:
        templist = []
        b = 0
        for i in range(0, len(list)):
            a = inputs[i]*list[i]
            templist.append(a)
        b = sum(templist) + bias[count]
        if b > 500:
            b = 1
        if b <-500:
            b = 0
        b = 1 / (1 + math.exp(-b))
        count += 1
        output.append(b)
    return output

# True and False Statements to numbers
def tfconvert(statement):
    if statement == True:
        output = 1
    else:
        output = 0
    return output

# Saves bot Data and calculates fitness function
def datastoreandfitcal(packet, self, boost_tracker_over_game, dist_from_tm8_one, dist_from_tm8_two, my_car, data, check):
    if check == True:
        return "already saved and calculated"

    if packet.teams[0].team_index == my_car.team:
        my_team_goals = packet.teams[0].score
        op_team_goals = packet.teams[1].score
    else:
        my_team_goals = packet.teams[1].score
        op_team_goals = packet.teams[0].score

    # Fitness Function Variables
    boost_avg = sum(boost_tracker_over_game)/len(boost_tracker_over_game)
    avg_dist_from_tm8_one = sum(dist_from_tm8_one)/len(dist_from_tm8_one)
    avg_dist_from_tm8_two = sum(dist_from_tm8_two)/len(dist_from_tm8_two)
    score = my_car.score_info.score
    goals = my_car.score_info.goals
    own_goals = my_car.score_info.own_goals
    assists = my_car.score_info.assists
    saves = my_car.score_info.saves
    shots = my_car.score_info.shots
    demolitions = my_car.score_info.demolitions

    # Rienforcement Wieghts (how much each part is worth to the score)
    rw = [
        1,      #boost average wieght for the fitness function
        .01,    #distance from teammate 1 wieght for the fitness function
        .01,    #distance from teammate 2 for the fitness function
        1,      #bots's score wieght for the fitness function
        100,    #bot's goals wieght for the fitness function
        25,     #bot's assits wieght for the fitness function
        50,     #bot's saves wieght for the fitness function
        1,      #bot's shots wieght for the fitness function
        10,     #bot's demolitions wieght for the fitness function
        -100,   #bot's own goals wieght for the fitness function
    ]
    # List of Rienforcements
    r = [
        boost_avg,
        avg_dist_from_tm8_one,
        avg_dist_from_tm8_two,
        score,
        goals,
        assists,
        saves,
        shots,
        demolitions,
        own_goals,
    ]
    # Rienforcement Function
    templist = []
    for i in range(0,len(r)):
        a = r[i]*rw[i]
        templist.append(a)
    fitness = sum(templist)
    bot_name = "bot" + str(self.index)
    file_name = "src/bot_data/bot" + str(self.index) + ".py"

    f = open(file_name,"w")
    f.write(bot_name + "_wieght_layer_one = " + str(data[0]) + "\n")
    f.write(bot_name + "_bias_layer_one = " + str(data[1]) + "\n")
    f.write(bot_name + "_wieght_layer_two = " + str(data[2]) + "\n")
    f.write(bot_name + "_bias_layer_two = " + str(data[3]) + "\n")
    f.write(bot_name + "_wieght_layer_three = " + str(data[4]) + "\n")
    f.write(bot_name + "_bias_layer_three = " + str(data[5]) + "\n")
    f.write(bot_name + "_fitness = " + str(fitness) + "\n")
    f.write(bot_name + "_num = " + str(self.index) + "\n")
    f.close()
    print("saved")
    return

# Finds the bot with the highest fitness and compares it to the best bots fitness and then replaces it if it is higher
def findbestbot():
    fitness_list = [
        bot0_fitness,
        bot1_fitness,
        bot2_fitness,
        bot3_fitness,
        bot4_fitness,
        bot5_fitness,
    ]
    # Finds Max from the stored bot Data
    m = max(fitness_list)
    bot_number = fitness_list.index(m)

    # Compares fitness to best bot and if better then replaces data
    if m > bestbot_fitness:
        data = []
        if bot_number == 0:
            data = [
                bot0_wieght_layer_one,
                bot0_bias_layer_one,
                bot0_wieght_layer_two,
                bot0_bias_layer_two,
                bot0_wieght_layer_three,
                bot0_bias_layer_three,
                bot0_fitness
            ]

        if bot_number == 1:
            data = [
                bot1_wieght_layer_one,
                bot1_bias_layer_one,
                bot1_wieght_layer_two,
                bot1_bias_layer_two,
                bot1_wieght_layer_three,
                bot1_bias_layer_three,
                bot1_fitness
            ]

        if bot_number == 2:
            data = [
                bot2_wieght_layer_one,
                bot2_bias_layer_one,
                bot2_wieght_layer_two,
                bot2_bias_layer_two,
                bot2_wieght_layer_three,
                bot2_bias_layer_three,
                bot2_fitness
            ]

        if bot_number == 3:
            data = [
                bot3_wieght_layer_one,
                bot3_bias_layer_one,
                bot3_wieght_layer_two,
                bot3_bias_layer_two,
                bot3_wieght_layer_three,
                bot3_bias_layer_three,
                bot3_fitness
            ]

        if bot_number == 4:
            data = [
                bot4_wieght_layer_one,
                bot4_bias_layer_one,
                bot4_wieght_layer_two,
                bot4_bias_layer_two,
                bot4_wieght_layer_three,
                bot4_bias_layer_three,
                bot4_fitness
            ]

        if bot_number == 5:
            data = [
                bot5_wieght_layer_one,
                bot5_bias_layer_one,
                bot5_wieght_layer_two,
                bot5_bias_layer_two,
                bot5_wieght_layer_three,
                bot5_bias_layer_three,
                bot5_fitness
            ]
        f = open("src/bot_data/bestbot.py", "w")
        f.write("bestbot_wieght_layer_one = " + str(data[0]) + "\n")
        f.write("bestbot_bias_layer_one = " + str(data[1]) + "\n")
        f.write("bestbot_wieght_layer_two = " + str(data[2]) + "\n")
        f.write("bestbot_bias_layer_two = " + str(data[3]) + "\n")
        f.write("bestbot_wieght_layer_three = " + str(data[4]) + "\n")
        f.write("bestbot_bias_layer_three = " + str(data[5]) + "\n")
        f.write("bestbot_fitness = " + str(data[6]) + "\n")
        f.close
        print("data is stored")

# def mutatebestbot():
