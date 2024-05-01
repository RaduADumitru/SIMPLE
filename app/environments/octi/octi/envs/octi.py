
import gym
import numpy as np

import config

from stable_baselines import logger

from octi_shared import *


ACTION_COUNT = 32 # 4 pods maximum, each with 8 directions
MAX_TURN_COUNT = 100

class Player():
    def __init__(self, id, token):
        self.id = id
        self.token = token
        

class Token():
    def __init__(self, symbol, number):
        self.number = number
        self.symbol = symbol
        
        
class OctiEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(OctiEnv, self).__init__()
        self.name = 'octi'
        self.manual = manual

        self.rows = 7
        self.cols = 6
        self.n_players = 2
        self.grid_shape = (self.rows, self.cols)
        self.num_squares = self.rows * self.cols
        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        # observation space: 18 layers. 1st layer current player pods, 2nd current N prongs,
        # 3rd layer current NE prongs, ... for each direction,
        # 10th layer opposing player pods, 11th opposing N prongs, ...
        self.observation_space = gym.spaces.Box(-1, 1, self.grid_shape + (18, ))
        self.verbose = verbose
        

    @property
    def observation(self):
        # if self.current_player.token.number == 1:
        #     position_1 = np.array([1 if x.number == 1 else 0  for x in self.board]).reshape(self.grid_shape)
        #     position_2 = np.array([1 if x.number == -1 else 0 for x in self.board]).reshape(self.grid_shape)
        #     position_3 = np.array([self.can_be_placed(i) for i,x in enumerate(self.board)]).reshape(self.grid_shape)
        # else:
        #     position_1 = np.array([1 if x.number == -1 else 0 for x in self.board]).reshape(self.grid_shape)
        #     position_2 = np.array([1 if x.number == 1 else 0 for x in self.board]).reshape(self.grid_shape)
        #     position_3 = np.array([self.can_be_placed(i) for i,x in enumerate(self.board)]).reshape(self.grid_shape)

        # out = np.stack([position_1, position_2, position_3], axis = -1)]
        # if red, turn tokens upside down so that observation is same format for both players
        if self.current_player.token.number == -1:
            observation_board = np.rot90(self.board.tokens, 2)
        else:
            observation_board = self.board.tokens
        # TODO: possible to optimize this?
        # alternate approaches: size 9 instead of 18. Current player value 1, opposing player value -1
        out = np.zeros(self.grid_shape + (18,))
        for row in range(self.rows):
            for col in range(self.cols):
                token = observation_board[row, col]
                if token.number == self.current_player.token.number:
                    out[row, col, 0] = 1
                    for direction in Direction:
                        out[row, col, 1 + direction.value] = token.has_prong(direction)
                elif token.number != TokenType.NONE.value:
                    out[row, col, 9] = 1
                    for direction in Direction:
                        out[row, col, 10 + direction] = token.has_prong(direction)
        return out

    @property
    def legal_actions(self):

        legal_actions = []
        board_tokens = self.board.tokens
        # if red, turn tokens upside down so that observation is same format for both players
        if self.current_player.token.number == -1:
            board_tokens = np.rot90(board_tokens, 2)
        for row in range(self.rows):
            for col in range(self.cols):
                token = board_tokens[row, col]
                if token.number == self.current_player.token.number:
                    for direction in Direction:
                        if not token.has_prong(direction):
                            # legal action: place prong in direction
                            legal_actions.append(1)
                        else:
                            # check if it is legal to move in direction
                            new_row = row + direction_to_position[direction][0]
                            new_col = col + direction_to_position[direction][1]
                            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                                if board_tokens[new_row, new_col].number == TokenType.NONE.value:
                                    # move into unoocupied square
                                    legal_actions.append(1)
                                else:
                                    new_row = row + 2 * direction_to_position[direction][0]
                                    new_col = col + 2 * direction_to_position[direction][1]
                                    if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                                        # attempt capture
                                        if board_tokens[new_row, new_col].number == TokenType.NONE.value:
                                            # capture ends in unoccupied square
                                            legal_actions.append(1)
                                        else:
                                            # capture ends in occupied square
                                            legal_actions.append(0)
                                    else:
                                        # capture ends off board
                                        legal_actions.append(0)
                            else:
                                # move off board
                                legal_actions.append(0)
        # legal_actions = []
        # for action_num in range(self.action_space.n):
        #     legal = self.is_legal(action_num)
        #     legal_actions.append(legal)

        # pad legal actions with 0s to ACTION_COUNT until it is 32. compensates for missing pods
        legal_actions += [0] * (ACTION_COUNT - len(legal_actions))
        return np.array(legal_actions)


    # def is_legal(self, action_num):
    #     if self.board[action_num].number==0:
    #         return 1
    #     else:
    #         return 0

    # def can_be_placed(self, square_num):
        
    #     if self.board[square_num].number==0:
    #         for height in range(square_num + self.cols, self.num_squares , self.cols):
    #             if self.board[height].number==0:
    #                 return 0
    #     else:
    #         return 0

    #     return 1



    # def square_is_player(self, board, square, player):
    #     return board[square].number == self.players[player].token.number

    def check_game_over(self, board = None , player = None):

        if board is None:
            board = self.board

        if player is None:
            player = self.current_player_num

        if board.check_winner().value == self.players[player].token.number:
            return 1, True

        # for x,y,z,a in WINNERS:
        #     if self.square_is_player(board, x, player) and self.square_is_player(board, y, player) and self.square_is_player(board, z, player) and self.square_is_player(board, a, player):
        #         return 1, True

        if self.turns_taken >= MAX_TURN_COUNT:
            logger.debug(f"Turn limit reached: {MAX_TURN_COUNT} turns taken")
            return  0, True

        return 0, False #-0.01 here to encourage choosing the win?

    # def get_square(self, board, action):
    #     for height in range(1, self.rows + 1):
    #         square = self.num_squares - (height * self.cols) + action
    #         if board[square].number == 0:
    #             return square

    @property
    def current_player(self):
        return self.players[self.current_player_num]

    def step(self, action):
        
        reward = [0,0]
        
        # check move legality
        board_state = self.board
        # if red, turn tokens upside down so that observation is same format for both players
        if self.current_player.token.number == -1:
            board_state = BoardState(np.rot90(self.board.tokens, 2))
        move_counter = 0
        found_move = False
        for row in range(self.rows):
            for col in range(self.cols):
                token = board_state.tokens[row, col]
                if token.number == self.current_player.token.number:
                    if action - move_counter < 8:
                        # place prong in direction
                        direction = Direction(action - move_counter)
                        move = board_state.get_move_in_direction(row, col, direction)
                        if move is None:
                            # invalid move
                            done = True
                            reward = [1,1]
                            reward[self.current_player_num] = -1
                        else:
                            # valid move
                            self.board = BoardState(move.board.tokens)

                            self.turns_taken += 1
                            r, done = self.check_game_over()
                            reward = [-r,-r]
                            reward[self.current_player_num] = r

                        found_move = True
                        break
                    else:
                        move_counter += 8 # moves for each direction of that pod
        if not found_move:
            # invalid move
            done = True
            reward = [1,1]
            reward[self.current_player_num] = -1
        
        self.done = done

        if not done:
            self.current_player_num = (self.current_player_num + 1) % 2

        return self.observation, reward, done, {}
                        

        
        
        # if not self.is_legal(action): 
        #     done = True
        #     reward = [1,1]
        #     reward[self.current_player_num] = -1
        # else:
        #     square = self.get_square(board, action)
        #     board[square] = self.current_player.token

        #     self.turns_taken += 1
        #     r, done = self.check_game_over()
        #     reward = [-r,-r]
        #     reward[self.current_player_num] = r

        # self.done = done

        # if not done:
        #     self.current_player_num = (self.current_player_num + 1) % 2

        # return self.observation, reward, done, {}

    def reset(self):
        self.board = BoardState()
        self.players = [Player('1', Token('G', 1)), Player('2', Token('R', 2))]
        self.current_player_num = 0
        self.turns_taken = 0
        self.done = False
        logger.debug(f'\n\n---- NEW GAME ----')
        return self.observation


    def render(self, mode='human', close=False):
        logger.debug('')
        if close:
            return
        if self.done:
            logger.debug(f'GAME OVER')
        else:
            logger.debug(f"It is Player {self.current_player.id}'s turn to move")
        
        # for i in range(0,self.num_squares,self.cols):
        #     logger.debug(' '.join([x.symbol for x in self.board[i:(i+self.cols)]]))

        logger.debug(f'\nBoard: \n{self.board}')

        if self.verbose:
            logger.debug(f'\nObservation: \n{self.observation}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')



    # def rules_move(self):
    #     WRONG_MOVE_PROB = 0.01
    #     player = self.current_player_num

    #     for action in range(self.action_space.n):
    #         if self.is_legal(action):
    #             new_board = self.board.copy()
    #             square = self.get_square(new_board, action)
    #             new_board[square] = self.players[player].token
    #             _, done = self.check_game_over(new_board, player)
    #             if done:
    #                 action_probs = [WRONG_MOVE_PROB] * self.action_space.n
    #                 action_probs[action] = 1 - WRONG_MOVE_PROB * (self.action_space.n - 1)
    #                 return action_probs

    #     player = (self.current_player_num + 1) % 2

    #     for action in range(self.action_space.n):
    #         if self.is_legal(action):
    #             new_board = self.board.copy()
    #             square = self.get_square(new_board, action)
    #             new_board[square] = self.players[player].token
    #             _, done = self.check_game_over(new_board, player)
    #             if done:
    #                 action_probs = [0] * self.action_space.n
    #                 action_probs[action] = 1 - WRONG_MOVE_PROB * (self.action_space.n - 1)
    #                 return action_probs

        
    #     action, masked_action_probs = self.sample_masked_action([1] * self.action_space.n)
    #     return masked_action_probs




# WINNERS = [
# 			[0,1,2,3],
# 			[1,2,3,4],
# 			[2,3,4,5],
# 			[3,4,5,6],
# 			[7,8,9,10],
# 			[8,9,10,11],
# 			[9,10,11,12],
# 			[10,11,12,13],
# 			[14,15,16,17],
# 			[15,16,17,18],
# 			[16,17,18,19],
# 			[17,18,19,20],
# 			[21,22,23,24],
# 			[22,23,24,25],
# 			[23,24,25,26],
# 			[24,25,26,27],
# 			[28,29,30,31],
# 			[29,30,31,32],
# 			[30,31,32,33],
# 			[31,32,33,34],
# 			[35,36,37,38],
# 			[36,37,38,39],
# 			[37,38,39,40],
# 			[38,39,40,41],

# 			[0,7,14,21],
# 			[7,14,21,28],
# 			[14,21,28,35],
# 			[1,8,15,22],
# 			[8,15,22,29],
# 			[15,22,29,36],
# 			[2,9,16,23],
# 			[9,16,23,30],
# 			[16,23,30,37],
# 			[3,10,17,24],
# 			[10,17,24,31],
# 			[17,24,31,38],
# 			[4,11,18,25],
# 			[11,18,25,32],
# 			[18,25,32,39],
# 			[5,12,19,26],
# 			[12,19,26,33],
# 			[19,26,33,40],
# 			[6,13,20,27],
# 			[13,20,27,34],
# 			[20,27,34,41],

# 			[3,9,15,21],
# 			[4,10,16,22],
# 			[10,16,22,28],
# 			[5,11,17,23],
# 			[11,17,23,29],
# 			[17,23,29,35],
# 			[6,12,18,24],
# 			[12,18,24,30],
# 			[18,24,30,36],
# 			[13,19,25,31],
# 			[19,25,31,37],
# 			[20,26,32,38],

# 			[3,11,19,27],
# 			[2,10,18,26],
# 			[10,18,26,34],
# 			[1,9,17,25],
# 			[9,17,25,33],
# 			[17,25,33,41],
# 			[0,8,16,24],
# 			[8,16,24,32],
# 			[16,24,32,40],
# 			[7,15,23,31],
# 			[15,23,31,39],
# 			[14,22,30,38],
# 			]