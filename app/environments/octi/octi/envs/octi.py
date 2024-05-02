
import gym
import numpy as np

from stable_baselines import logger
from enum import Enum
import copy


# from octi_shared import *


ACTION_COUNT = 32 # 4 pods maximum, each with 8 directions
MAX_TURN_COUNT = 200

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
        if self.current_player.token.number == 2:
            observation_board = self.get_rotated_board(self.board.tokens)
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
                        out[row, col, 10 + direction.value] = token.has_prong(direction)
        return out

    @property
    def legal_actions(self):

        legal_actions = []
        board_tokens = self.board.tokens
        # if red, turn tokens upside down so that observation is same format for both players
        if self.current_player.token.number == 2:
            board_tokens = self.get_rotated_board(board_tokens)
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
                                    # move into unoccupied square
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
            logger.debug(f"Found winning move for player {self.players[player].id}")
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
        logger.debug(f'Step Player {self.current_player.id} action: {action}')
        # check move legality
        board_state = self.board
        # if red, turn tokens upside down so that observation is same format for both players
        if self.current_player.token.number == 2:
            logger.debug(f'Rotating board for red player')
            board_state = BoardState(self.get_rotated_board(board_state.tokens))
        move_counter = 0
        found_move = False
        for row in range(self.rows):
            if not found_move:
                for col in range(self.cols):
                    token = board_state.tokens[row, col]
                    if token.number == self.current_player.token.number:
                        logger.debug(f'Checking same token at {row}, {col}')
                        if action - move_counter < 8:
                            # place prong in direction
                            direction = Direction(action - move_counter)
                            move = board_state.get_move_in_direction(row, col, direction, TokenType(self.current_player.token.number))
                            if move is None:
                                # invalid move
                                logger.debug(f'Invalid move')
                                done = True
                                reward = [1,1]
                                reward[self.current_player_num] = -1
                            else:
                                # valid move
                                logger.debug(f'Valid move')
                                logger.debug(f'New board: \n{move.board}')
                                if self.current_player.token.number == 2:
                                    # board was rotated before for red player, so rotate it back
                                    logger.debug(f'Rotating board back')
                                    self.board = BoardState(self.get_rotated_board(move.board.tokens))
                                else:
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
        # green 0, red 1
        self.current_player_num = 0
        self.turns_taken = 0
        self.done = False
        logger.debug(f'\n\n---- NEW GAME ----')
        return self.observation


    def render(self, mode='human', close=False):
        logger.debug('')
        if close:
            return
        logger.debug(f'Turns taken: {self.turns_taken}')
        if self.done:
            logger.debug(f'GAME OVER')
        else:
            logger.debug(f"It is Player {self.current_player.id}'s turn to move")
        
        # for i in range(0,self.num_squares,self.cols):
        #     logger.debug(' '.join([x.symbol for x in self.board[i:(i+self.cols)]]))

        logger.debug(f'\nBoard: \n{self.board}')

        logger.debug(f'\nRotated board: \n{BoardState(self.get_rotated_board(self.board.tokens))}')

        if self.verbose:
            logger.debug(f'\nObservation: \n{self.observation}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')


    def get_rotated_board(self, board_tokens):
        rotated_board_tokens = np.rot90(copy.deepcopy(board_tokens), 2)
        for row in range(self.rows):
            for col in range(self.cols):
                token = rotated_board_tokens[row, col]
                if token.number != TokenType.NONE.value:
                    token.prongs = token.prongs.get_rotated_prongs()
        return rotated_board_tokens

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
# directions: 0 = up, 1 = up-right, 2 = right, 3 = down-right, 4 = down, 5 = down-left, 6 = left, 7 = up-left
class Direction(Enum):
    N = 0  # up
    NE = 1  # up-right
    E = 2  # right
    SE = 3  # down-right
    S = 4  # down
    SW = 5  # down-left
    W = 6  # left
    NW = 7  # up-left

    def opposite(self):
        if self == Direction.N:
            return Direction.S
        elif self == Direction.NE:
            return Direction.SW
        elif self == Direction.E:
            return Direction.W
        elif self == Direction.SE:
            return Direction.NW
        elif self == Direction.S:
            return Direction.N
        elif self == Direction.SW:
            return Direction.NE
        elif self == Direction.W:
            return Direction.E
        elif self == Direction.NW:
            return Direction.SE

direction_to_position = {
    Direction.N: [-1, 0],
    Direction.NE: [-1, 1],
    Direction.E: [0, 1],
    Direction.SE: [1, 1],
    Direction.S: [1, 0],
    Direction.SW: [1, -1],
    Direction.W: [0, -1],
    Direction.NW: [-1, -1]
}

class TokenType(Enum):
    NONE = 0
    GREEN = 1
    RED = 2

    def opposite(self):
        if self == TokenType.GREEN:
            return TokenType.RED
        elif self == TokenType.RED:
            return TokenType.GREEN
        else:
            return TokenType.NONE


class Symbol(Enum):
    NONE = "*"
    GREEN = "G"
    RED = "R"

class ProngList:
    def __init__(self):
        self.prongs = [False] * 8
        self.prong_encoding = 0

    def set_prong(self, direction : int, has_prong):
        original_prong_value = self.prongs[direction]
        self.prongs[direction] = has_prong
        if original_prong_value != has_prong:
            if original_prong_value == True:
                self.prong_encoding -= 2 ** direction
            else:
                self.prong_encoding += 2 ** direction

    def has_prong(self, direction : int):
        return self.prongs[direction]
    
    def get_prong_count(self):
        return sum(self.prongs)
    
    def get_prong_encoding(self):
        encoding = 0
        for i in range(len(self.prongs)):
            if self.prongs[i]:
                encoding += 2 ** i
        return encoding
    
    def get_cached_prong_encoding(self):
        return self.prong_encoding
    
    def get_rotated_prongs(self):
        # return new prong list with prongs rotated 180 degrees
        rotated_prongs = ProngList()
        for i in range(len(self.prongs)):
            rotated_prongs.set_prong((i + 4) % 8, self.prongs[i])
        rotated_prongs.prong_encoding = rotated_prongs.get_prong_encoding()
        return rotated_prongs
    
class OctiToken:
    # 1 Green, 2 Red, 0 none
    def __init__(self, number : int, row : int, col : int, prong_list = None):
        self.number = number
        self.symbol = Symbol.GREEN if self.number == TokenType.GREEN.value else Symbol.RED if self.number == TokenType.RED.value else Symbol.NONE
        self.prongs = ProngList()
        self.row = row
        self.col = col
        if prong_list is not None:
            self.prongs = prong_list
        else:
            self.prongs = ProngList()

    
    def set_prong(self, direction : Direction):
        self.prongs.set_prong(direction.value, True)

    def has_prong(self, direction : Direction):
        return self.prongs.has_prong(direction.value)
    
# Starting position: column 1 spaces 1-4 is red base
RED_BASE = (1, 1, 1, 4)  # (start_row, start_column, end_row, end_column)
# Starting position: column 5 spaces 1-4 is green base
GREEN_BASE = (5, 1, 5, 4)  # (start_row, start_column, end_row, end_column)

class BoardState:

    def __init__(self, tokens = None):
        if tokens is None:
            self.columns = 6
            self.rows = 7
            self.tokens = np.empty((self.rows, self.columns), dtype=OctiToken)
            for row in range(self.rows):
                for col in range(self.columns):
                    self.tokens[row, col] = OctiToken(TokenType.NONE.value, row, col)
            for row in range(RED_BASE[0], RED_BASE[2] + 1):
                for col in range(RED_BASE[1], RED_BASE[3] + 1):
                    self.tokens[row, col] = OctiToken(TokenType.RED.value, row, col)
            for row in range(GREEN_BASE[0], GREEN_BASE[2] + 1):
                for col in range(GREEN_BASE[1], GREEN_BASE[3] + 1):
                    self.tokens[row, col] = OctiToken(TokenType.GREEN.value, row, col)
        else:
            self.columns = tokens.shape[1]
            self.rows = tokens.shape[0]
            self.tokens = copy.deepcopy(tokens)

    

    def get_prong(self, symbol, direction, row, col):
        if self.tokens[row, col].has_prong(direction):
            return symbol
        else:
            return " "

    def get_top_row_margin(self):
        return "⌜" + "-" * (self.columns * 4 - 1) + "⌝"

    def get_bottom_row_margin(self):
        return "⌞" + "-" * (self.columns * 4 - 1) + "⌟"
    
    def get_row_margin(self):
        return "|" + "-" * (self.columns * 4 - 1) + "|"

    def get_row_top(self, row):
        result = ""
        for col in range(self.columns):
            result += "|"
            result += self.get_prong("\\", Direction.NW, row, col)
            result += self.get_prong("|", Direction.N, row, col)
            result += self.get_prong("/", Direction.NE, row, col)
        result += "|"
        return result

    def get_row_middle(self, row):
        result = ""
        for col in range(self.columns):
            result += "|"
            result += self.get_prong("-", Direction.W, row, col)
            result += self.tokens[row, col].symbol.value
            result += self.get_prong("-", Direction.E, row, col)
        result += "|"
        return result

    def get_row_bottom(self, row):
        result = ""
        for col in range(self.columns):
            result += "|"
            result += self.get_prong("/", Direction.SW, row, col)
            result += self.get_prong("|", Direction.S, row, col)
            result += self.get_prong("\\", Direction.SE, row, col)
        result += "|"
        return result
    
    def get_row(self, row):
        result = ""
        result += self.get_row_top(row) + "\n"
        result += self.get_row_middle(row) + "\n"
        result += self.get_row_bottom(row) + "\n"
        return result
    
    def get_board(self):
        result = ""
        result += self.get_top_row_margin() + "\n"
        for row in range(self.rows):
            result += self.get_row(row)
            if row < self.rows - 1:
                result += self.get_row_margin() + "\n"
            else:
                result += self.get_bottom_row_margin() + "\n"
        return result

    def __str__(self):
        return self.get_board()

    def get_legal_actions(self, token : TokenType) -> np.ndarray:
        if self.is_final_state():
            return np.array([])
        moves = np.array([], dtype=BoardStateWithZobristHash)
        for row in range(self.rows):
            for col in range(self.columns):
                if self.tokens[row, col].number == token.value:
                    moves = np.concatenate((moves, self.get_next_moves_for_token_position(row, col, token)))
        return moves

    def get_next_moves_for_token_position(self, row : int, col : int, token : TokenType):
        moves = np.array([], dtype=BoardStateWithZobristHash)
        for direction in Direction:
            move = self.get_move_in_direction(row, col, direction, token)
            if move is not None:
                moves = np.concatenate((moves, [move]))
        return moves

    def get_move_in_direction(self, row : int, col : int, direction : Direction, token_type : TokenType):
        if self.tokens[row, col].number != token_type.value:
            return None
        if self.tokens[row, col].has_prong(direction):
            return self.get_token_position_move(row, col, direction)
        else:
            # place prong on that token
            new_board_tokens = copy.deepcopy(self.tokens)
            new_board_tokens[row, col].set_prong(direction)
            new_board = BoardState(new_board_tokens)
            spaces_to_unxor = np.array([(row, col)])
            spaces_to_xor = np.array([(row, col)])
            return BoardStateWithZobristHash(new_board, spaces_to_unxor, spaces_to_xor)
        
    def get_token_position_move(self, row : int, col : int, direction : Direction):
        next_row = row + direction_to_position[direction][0]
        next_col = col + direction_to_position[direction][1]
        if next_row < 0 or next_row >= self.rows or next_col < 0 or next_col >= self.columns:
            # move outside of board
            return None
        if self.tokens[next_row, next_col].number == TokenType.NONE.value:
            # move into unoccupied space
            moved_token = OctiToken(self.tokens[row, col].number, next_row, next_col, copy.deepcopy(self.tokens[row, col].prongs))
            new_board_tokens = copy.deepcopy(self.tokens)
            new_board_tokens[row, col] = OctiToken(TokenType.NONE.value, row, col)
            new_board_tokens[next_row, next_col] = moved_token
            new_board = BoardState(new_board_tokens)
            spaces_to_unxor = np.array([(row, col)])
            spaces_to_xor = np.array([(next_row, next_col)])
            return BoardStateWithZobristHash(new_board, spaces_to_unxor, spaces_to_xor)
        else:
            # move into occupied space
            # check if capture is valid
            next_row_after_capture = next_row + direction_to_position[direction][0]
            next_col_after_capture = next_col + direction_to_position[direction][1]
            if next_row_after_capture < 0 or next_row_after_capture >= self.rows or next_col_after_capture < 0 or next_col_after_capture >= self.columns:
                # capture would get piece outside of board
                return None
            if self.tokens[next_row_after_capture, next_col_after_capture].number != TokenType.NONE.value:
                # space where piece would end up after capture is occupied by another piece
                return None
            # capture is valid
            target_player = self.tokens[next_row, next_col].number
            moved_token = OctiToken(self.tokens[row, col].number, next_row_after_capture, next_col_after_capture, copy.deepcopy(self.tokens[row, col].prongs))
            new_board_tokens = copy.deepcopy(self.tokens)
            new_board_tokens[row, col] = OctiToken(TokenType.NONE.value, row, col)
            new_board_tokens[next_row_after_capture, next_col_after_capture] = moved_token
            if moved_token.number != target_player:
                # capture enemy piece
                new_board_tokens[next_row, next_col] = OctiToken(TokenType.NONE.value, next_row, next_col)
                new_board = BoardState(new_board_tokens)
                spaces_to_unxor = np.array([(row, col), (next_row, next_col)])
                spaces_to_xor = np.array([(next_row_after_capture, next_col_after_capture)])
                return BoardStateWithZobristHash(new_board, spaces_to_unxor, spaces_to_xor)

            else:
                # jump over own piece
                new_board = BoardState(new_board_tokens)
                spaces_to_unxor = np.array([(row, col)])
                spaces_to_xor = np.array([(next_row_after_capture, next_col_after_capture)])
                return BoardStateWithZobristHash(new_board, spaces_to_unxor, spaces_to_xor)
    
    def check_winner(self):
        # 2 win conditions:
        # 1. A given player has no more tokens
        # 2. A given player has one of their pieces in the opposite base
        red_token_count = 0
        green_token_count = 0
        red_in_green_base = False
        green_in_red_base = False
        for row in range(self.rows):
            for col in range(self.columns):
                token = self.tokens[row, col]
                if token.number == TokenType.RED.value:
                    red_token_count += 1
                    if row >= GREEN_BASE[0] and row <= GREEN_BASE[2] and col >= GREEN_BASE[1] and col <= GREEN_BASE[3]:
                        # red reached green base
                        return TokenType.RED
                elif token.number == TokenType.GREEN.value:
                    green_token_count += 1
                    if row >= RED_BASE[0] and row <= RED_BASE[2] and col >= RED_BASE[1] and col <= RED_BASE[3]:
                        # green reached red base
                        return TokenType.GREEN
        if red_token_count == 0:
            return TokenType.GREEN
        elif green_token_count == 0:
            return TokenType.RED
        else:
            return TokenType.NONE

    def is_final_state(self):
        return self.check_winner() != TokenType.NONE
    
class BoardStateWithZobristHash:
    def __init__(self, board: BoardState, spaces_to_unxor: np.ndarray = None, spaces_to_xor: np.ndarray = None):
        self.board = board
        self.spaces_to_unxor = spaces_to_unxor
        self.spaces_to_xor = spaces_to_xor