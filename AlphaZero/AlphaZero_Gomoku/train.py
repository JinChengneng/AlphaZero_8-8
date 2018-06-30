# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
""" 

from __future__ import print_function
import random
import numpy as np
import pickle  #import cPickle as pickle
from collections import defaultdict, deque
from game import Board, Game
#from policy_value_net import PolicyValueNet  # Theano and Lasagne
# 此处引入用Pytorch实现的策略价值网络
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer

import datetime


class TrainPipeline():
    def __init__(self, init_model=None):
        # params of the board and the game
        self.board_width = 8  #6
        self.board_height = 8 #6
        self.n_in_row = 5  #4
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params 
        self.learn_rate = 5e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0 # the temperature param
        self.n_playout = 400 # num of simulations for each move
        # c_puct是MCTS里用来控制exploration-exploit tradeoff的参数
        # 这个参数越大的话MCTS搜索的过程中就偏向于均匀的探索，越小的话就偏向于直接选择访问次数多的分支
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512 # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)        
        self.play_batch_size = 1 
        self.epochs = 5 # num of train_steps for each update
        self.kl_targ = 0.025
        # 检查当前策咯胜率的频率，当前设置为每50次训练后通过自我对弈评价当前策略
        # 如果找到更优策略，则保存当前策咯模型
        self.check_freq = 50 
        #训练迭代次数
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as the opponent to evaluate the trained policy
        #每次训练蒙特卡洛树搜索的次数，初始化为1000（后续训练过程中会不断增加）
        self.pure_mcts_playout_num = 1000  
        if init_model:
            # start training from an initial policy-value net
            policy_param = pickle.load(open(init_model, 'rb')) 
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, net_params = policy_param)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height) 
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1)

    def get_equi_data(self, play_data):
        """
        augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]"""
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1,2,3,4]:
                # rotate counterclockwise 
                equi_state = np.array([np.rot90(s,i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data
                
    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data_zip2list = list(play_data)  # add by haward
            self.episode_len = len(play_data_zip2list)
            # augment the data
            play_data = self.get_equi_data(play_data_zip2list)
            self.data_buffer.extend(play_data)
                        
    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]            
        old_probs, old_v = self.policy_value_net.policy_value(state_batch) 
        for i in range(self.epochs): 
            loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            # kl距离，也叫做相对熵，衡量的是相同事件空间里的两个概率分布的差异情况
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))  
            if kl > self.kl_targ * 4:   # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        # 通过比较新旧两个神经网络输出的KL散度（信息增益）来控制学习率，使得学习率快死增加然后逐渐减少
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
            
        explained_var_old =  1 - np.var(np.array(winner_batch) - old_v.flatten())/np.var(np.array(winner_batch))
        explained_var_new = 1 - np.var(np.array(winner_batch) - new_v.flatten())/np.var(np.array(winner_batch))        
        print("kl:{:.5f},lr_multiplier:{:.3f},loss:{},entropy:{},explained_var_old:{:.3f},explained_var_new:{:.3f}".format(
                kl, self.lr_multiplier, loss, entropy, explained_var_old, explained_var_new))
        return loss, entropy
        
    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing games against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player, pure_mcts_player, start_player=i%2, is_shown=0)
            win_cnt[winner] += 1
        #计算赢率，获胜积一分，平局积0.5分，失败不计分，再以总积分除以总比赛次数
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1])/n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(self.pure_mcts_playout_num, win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio
    
    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):                
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(i+1, self.episode_len))                
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()                    
                # check the performance of the current model and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    # 与纯蒙特卡洛树进行十局对弈，计算对弈胜率
                    win_ratio = self.policy_evaluate()
                    net_params = self.policy_value_net.get_policy_param() # get model params
                    pickle.dump(net_params, open('current_policy_8_8_5_new.model', 'wb'), pickle.HIGHEST_PROTOCOL) # save model param to file
                    if win_ratio > self.best_win_ratio: 
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        pickle.dump(net_params, open('best_policy_8_8_5_new.model', 'wb'), pickle.HIGHEST_PROTOCOL) # update the best_policy
                        #如果当前策咯价值网络胜率为1，则提高纯蒙特卡洛树的搜索次数，继续训练
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')
    

if __name__ == '__main__':
    #导入之前训练好的模型，此处为增量训练
    training_pipeline = TrainPipeline("best_policy_8_8_5_new.model")
    time_start = datetime.datetime.now()
    print("开始时间：" + time_start.strftime('%Y.%m.%d-%H:%M:%S'))
    training_pipeline.run()
    time_end = datetime.datetime.now()
    print("结束时间：" + time_end.strftime('%Y.%m.%d-%H:%M:%S'))
    