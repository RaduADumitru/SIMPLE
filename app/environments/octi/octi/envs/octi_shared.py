from enum import Enum
import numpy as np
import copy
from abc import ABC, abstractmethod



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

    def get_move_in_direction(self, row : int, col : int, direction : Direction, token : TokenType):
        if self.tokens[row, col].number != token.value:
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

class OctiPlayer(ABC):
    def __init__(self, token : TokenType):
        self.player_id = token
        self.opponent_player_id = token.opposite()
        self.board = None

    def set_board(self, board : BoardState):
        self.board = board

    @abstractmethod
    def make_next_move(self):
        pass

WIN_SCORE = 10000
CLOSE_TO_BASE_SCORE = 300
CLOSE_TO_BASE_SCORE_2 = 100
POD_SCORE = 500
EDGE_SCORE = 10
PRONG_SCORE = 5
FRONT_PRONG_SCORE = 10
ADJACENT_POD_SCORE = 100
CAPTURE_SCORE = 300	

FRONT_PRONGS = {
    TokenType.GREEN.value: [Direction.NW, Direction.N, Direction.NE],
    TokenType.RED.value: [Direction.SW, Direction.S, Direction.SE]
}

def calculate_score_for_token_heuristic_1(board : BoardState, token : OctiToken, agent_player_id : TokenType):
    score = 0
    if token.number == TokenType.NONE.value:
        return score
    VERTICAL_EDGE_CLOSE_TO_BASE = 0 if token.number == TokenType.GREEN.value else board.rows - 1
    score += POD_SCORE
    if token.col == 0 or token.col == board.columns - 1:
        score += EDGE_SCORE
    distance_to_base = np.abs(token.row - VERTICAL_EDGE_CLOSE_TO_BASE)
    score += CLOSE_TO_BASE_SCORE // (distance_to_base + 1)
    prong_score = PRONG_SCORE * token.prongs.get_prong_count()
    score += prong_score
    front_prongs = FRONT_PRONGS[token.number]
    for prong in front_prongs:
        if token.has_prong(prong):
            score += FRONT_PRONG_SCORE
    if token.number == agent_player_id.value:
        return score
    else:
        return -score

def calculate_score_for_board_heuristic_1(board : BoardState, agent_player_id : TokenType):
    # Calculate the score for the player, according to each token on board
    score = np.sum([calculate_score_for_token_heuristic_1(board, token, agent_player_id) for token in board.tokens.flatten()])
    return score

def calculate_score_for_board_heuristic_2(board : BoardState, agent_player_id : TokenType):
    # Calculate the score for the player, according to each token on board
    score = np.sum([calculate_score_for_token_heuristic_2(board, token, agent_player_id) for token in board.tokens.flatten()])
    return score

def calculate_score_for_token_heuristic_2(board : BoardState, token : OctiToken, agent_player_id : TokenType):
    score = 0
    if token.number == TokenType.NONE.value:
        return score
    VERTICAL_EDGE_CLOSE_TO_BASE = 0 if token.number == TokenType.GREEN.value else board.rows - 1
    score += POD_SCORE
    # if token.col == 0 or token.col == board.columns - 1:
    #     score += EDGE_SCORE
    distance_to_base = np.abs(token.row - VERTICAL_EDGE_CLOSE_TO_BASE)
    score += CLOSE_TO_BASE_SCORE // (distance_to_base + 1)
    # score += (board.rows - distance_to_base) * CLOSE_TO_BASE_SCORE_2
    prong_score = PRONG_SCORE * token.prongs.get_prong_count()
    score += prong_score
    front_prongs = FRONT_PRONGS[token.number]
    for prong in front_prongs:
        if token.has_prong(prong):
            score += FRONT_PRONG_SCORE
    # check prongs that you can jump over
    for direction in Direction:
        next_row = token.row + direction_to_position[direction][0]
        next_col = token.col + direction_to_position[direction][1]
        if next_row < 0 or next_row >= board.rows or next_col < 0 or next_col >= board.columns:
            continue
        if board.tokens[next_row, next_col].number == TokenType.NONE.value:
            continue
        # if board.tokens[next_row, next_col].number != token.number:
        #     enemy_pod = board.tokens[next_row, next_col]
        #     next_row_after_capture = next_row + direction_to_position[direction][0]	
        #     next_col_after_capture = next_col + direction_to_position[direction][1]
        #     if next_row_after_capture < 0 or next_row_after_capture >= board.rows or next_col_after_capture < 0 or next_col_after_capture >= board.columns:
        #         continue
        #     if board.tokens[next_row_after_capture, next_col_after_capture].number != TokenType.NONE.value:
        #         continue
        #     score += CAPTURE_SCORE
            
        if token.has_prong(direction):
            # next_row = token.row + direction_to_position[direction][0]
            # next_col = token.col + direction_to_position[direction][1]
            # if next_row < 0 or next_row >= board.rows or next_col < 0 or next_col >= board.columns:
            #     continue
            # if board.tokens[next_row, next_col].number == TokenType.NONE.value:
            #     continue
            # if board.tokens[next_row, next_col].number == token.number:
            #     continue
            next_row_after_capture = next_row + direction_to_position[direction][0]
            next_col_after_capture = next_col + direction_to_position[direction][1]
            if next_row_after_capture < 0 or next_row_after_capture >= board.rows or next_col_after_capture < 0 or next_col_after_capture >= board.columns:
                continue
            if board.tokens[next_row_after_capture, next_col_after_capture].number != TokenType.NONE.value:
                continue
            # if board.tokens[next_row, next_col].number != token.number:
            #     score += CAPTURE_SCORE
            if board.tokens[next_row, next_col].number == token.number:
                score += ADJACENT_POD_SCORE
    if token.number == agent_player_id.value:
        return score
    else:
        return -score

def evaluate_board_heuristic_1(board : BoardState, agent_player_id : TokenType):
    # check if the game is over
    winner = board.check_winner()
    if winner == agent_player_id:
        return WIN_SCORE
    elif winner == agent_player_id.opposite():
        return -WIN_SCORE
    elif winner == TokenType.NONE:
        player_score = calculate_score_for_board_heuristic_1(board, agent_player_id)
        return player_score
    
def evaluate_board_heuristic_2(board: BoardState, agent_player_id: TokenType):
    # check if the game is over
    winner = board.check_winner()
    if winner == agent_player_id:
        return WIN_SCORE
    elif winner == agent_player_id.opposite():
        return -WIN_SCORE
    elif winner == TokenType.NONE:
        player_score = calculate_score_for_board_heuristic_2(board, agent_player_id)
        return player_score
    

    



    

