import random
import pandas as pd
import numpy as np

class Robot(object):  # 定义Robot类
    
    # 定义初始参数
    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.5):

        self.maze = maze
        self.valid_actions = self.maze.valid_actions  
        self.state = None  
        self.action = None  

        # Set Parameters of the Learning Robot
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # discount rate

        self.epsilon0 = epsilon0  
        self.epsilon = epsilon0  
        self.t = 0  # 步数

        self.Qtable = {}  # action value,注意它是一个 map
        self.reset()  





    def reset(self):
            """
            Reset the robot
            """
            self.state = self.sense_state()  #  获得当前智体状态
            self.create_Qtable_line(self.state)  # 创建 当前 state 的 action value 列表,

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the robot is learning its q table, or
        exceuting the testing procedure.
        """
        self.learning = learning
        self.testing = testing



    def update_parameter(self):
            """
            Some of the paramters of the q learning robot can be altered,
            update these parameters when necessary.
            """
          
            if self.testing:
                # TODO 1. No random choice when testing
                #测试阶段完全使用贪心策略
                self.epsilon = 0
            else:
                #self.t初始值为0,随着 step 增长,epsilon逐渐下降,智体从随机策略过度到贪心策略
                self.t += 1
                self.epsilon = 1.0/self.t


            return self.epsilon



    def sense_state(self):
            """
            Get the current state of the robot. In this
            """

            # TODO 3. Return robot's current state
            return self.maze.sense_robot()  



    def create_Qtable_line(self, state):
            """
            Create the qtable with the current state
            """
            # TODO 4. Create qtable with current state
            # Our qtable should be a two level dict,
            # Qtable[state] ={'u':xx, 'd':xx, ...}
            # If Qtable[state] already exits, then do
            # not change it.
            if state not in self.Qtable:  # Qtable是map,如果 map 中已经有了当前 state 的值,就什么都不做,否则创建一个新的.注意它不是一个 array,而是一个 map
                self.Qtable[state] = {'u':0.0, 'r':0.0, 'd':0.0, 'l':0.0}  # 否则，新增一个状态，并赋值为0，注意是float



    def choose_action(self):
            """
            Return an action according to given rules
            """
            def is_random_exploration():

                # TODO 5. Return whether do random choice
                # hint: generate a random number, and compare
                # it with epsilon
                random_num = random.random()
                flag = random_num < self.epsilon
                return flag
   
            if self.learning:
                #用一个随机数来决定到底是随机选择 action,还是贪心 action,这实际上是epsilon-greedy
                if is_random_exploration():
                    # TODO 6. Return random choose aciton
                    return random.choice(self.valid_actions)
                else:
                    # TODO 7. Return action with highest q value
                    action_value_dic = self.Qtable[self.state]
                    return max(action_value_dic, key=action_value_dic.get)
                
            elif self.testing:
                # TODO 7. choose action with highest q value
                action_value_dic = self.Qtable[self.state]
                return max(action_value_dic, key=action_value_dic.get)
            else:
                # TODO 6. Return random choose aciton
                return random.choice(self.valid_actions)




    def update_Qtable(self, r, action, next_state):
            """
            Update the qtable according to the given rule.
            """
            if self.learning:
                # TODO 8. When learning, update the q table according
                # to the given rules
                # q_old = self.Qtable[self.state][action]

                # #从当前 state 的 action 中选择最大的 actoin value
                # max_q = max(self.Qtable[next_state].values())
                # #根据贝尔曼方程预测新的 action value
                # q_new = r + self.gamma * max_q
                # self.Qtable[self.state][action] += self.alpha * (q_new - q_old)

                #Q learning formular
                self.Qtable[self.state][action] = self.Qtable[self.state][action] + self.alpha * ((r + self.gamma * max(self.Qtable[next_state].values())) - self.Qtable[self.state][action])
             



    def update(self):
            """
            Describle the procedure what to do when update the robot.
            Called every time in every epoch in training or testing.
            Return current action and reward.
            """
            self.state = self.sense_state() 
            self.create_Qtable_line(self.state) 

            action = self.choose_action() # 为该状态选择动作，取最大值或者随机
            reward = self.maze.move_robot(action) #  根据选择的动作，计算奖赏值


            next_state = self.sense_state() # 获得下一个状态
            self.create_Qtable_line(next_state) # 创建下一个状态的q值表

            if self.learning and not self.testing:
                self.update_Qtable(reward, action, next_state) #  更新action value 的值
                self.update_parameter() 
            return action, reward










