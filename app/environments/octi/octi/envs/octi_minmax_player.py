from octi_shared import *
from octi_shared import TokenType, BoardState
from octi_shared import Direction

import numpy as np
import time

FRONT_PRONGS = {
    TokenType.GREEN.value: [Direction.NW, Direction.N, Direction.NE],
    TokenType.RED.value: [Direction.SW, Direction.S, Direction.SE]
}
class OctiMinMaxPlayer(OctiPlayer):
    def __init__(self, player_id : Token, depth):
        if depth < 1:
            raise ValueError("Depth must be greater than or equal to 1")
        super().__init__(player_id)
        self.depth = depth
    
    WIN_SCORE = 10000
    CLOSE_TO_BASE_SCORE = 300
    POD_SCORE = 100
    EDGE_SCORE = 50
    PRONG_SCORE = 5
    FRONT_PRONG_SCORE = 10

    def calculate_score_for_token(self, board : BoardState, token : Token, player_id : TokenType):
        score = 0
        if token.number == TokenType.NONE.value:
            return score
        VERTICAL_EDGE_CLOSE_TO_BASE = 0 if token.number == TokenType.GREEN.value else board.rows - 1
        score += self.POD_SCORE
        if token.col == 0 or token.col == board.columns - 1:
            score += self.EDGE_SCORE
        distance_to_base = np.abs(token.row - VERTICAL_EDGE_CLOSE_TO_BASE)
        score += self.CLOSE_TO_BASE_SCORE // (distance_to_base + 1)
        prong_score = self.PRONG_SCORE * token.prongs.get_prong_count()
        score += prong_score
        front_prongs = FRONT_PRONGS[token.number]
        for prong in front_prongs:
            if token.has_prong(prong):
                score += self.FRONT_PRONG_SCORE
        if token.number == self.player_id.value:
            return score
        else:
            return -score

    def calculate_score_for_player(self, board : BoardState, player_id : TokenType):
        # Calculate the score for the player, according to each token on board
        score = np.sum([self.calculate_score_for_token(board, token, player_id) for token in board.tokens.flatten()])
        return score
    
    def evaluate(self, board : BoardState, player_id : TokenType):
        # check if the game is over
        winner = board.check_winner()
        if winner == self.player_id:
            return self.WIN_SCORE
        elif winner == self.player_id.opposite():
            return -self.WIN_SCORE
        elif winner == TokenType.NONE:
            player_score = self.calculate_score_for_player(board, player_id)
            return player_score


    def minimax(self, board : BoardState, depth : int, player_id : TokenType):
        if depth == 0:
            return self.evaluate(board, player_id), None
        
        legal_actions = board.get_legal_actions(player_id)
        if len(legal_actions) == 0:
            return self.evaluate(board, player_id), None
        
        if player_id == self.player_id:
            best_value = -np.inf
            best_action = None
            for action in legal_actions:
                value, _ = self.minimax(action, depth - 1, player_id.opposite())
                if value > best_value:
                    best_value = value
                    best_action = action
            return best_value, best_action
        else:
            best_value = np.inf
            best_action = None
            for action in legal_actions:
                value, _ = self.minimax(action, depth - 1, player_id.opposite())
                if value < best_value:
                    best_value = value
                    best_action = action
            return best_value, best_action
    
    def make_next_move(self, board):
        start_time = time.time()
        _, action = self.minimax(board, self.depth, self.player_id)
        end_time = time.time()
        print("Evaluation time:", end_time - start_time, "seconds")
        return action

    def __str__(self):
        return "OctiMinMaxPlayer(player_id={}, depth={})".format(self.player_id, self.depth)

def start_game():
    human_player = input("Enter your player (G/R): ")
    if human_player not in ["G", "R"]:
        raise ValueError("Invalid player. Please enter G for GREEN or R for RED.")
    human_player = TokenType.GREEN if human_player == "G" else TokenType.RED
    minmax_depth = int(input("Enter the depth for MinMax player: "))
    play_game(human_player, minmax_depth)

def play_game(human_player : TokenType, depth : int):
    board = BoardState()  # Create a new board
    agent = OctiMinMaxPlayer(human_player.opposite(), depth)  # Create a MinMax player
    current_player = TokenType.GREEN # Green player starts the game
    turn_count = 1
    while not board.is_final_state():
        print("Turn", turn_count)
        print(board)  # Print the current board state
        if current_player == human_player:
            # Human player's turn
            print("Your turn")
            action = input("Enter your move (row, col, direction - 0 = N, 1 = NE, 2 = E...): ")
            row, col, direction = map(int, action.split(","))
            direction = Direction(direction)
            move_result = board.get_move_in_direction(row, col, direction, human_player)
            if move_result == None:
                print("Invalid move. Try again.")
                continue
            board = BoardState(move_result.tokens)
        else:
            # MinMax player's turn
            print("MinMax player's turn")
            board = agent.make_next_move(board)
        turn_count += 1
        current_player = current_player.opposite()
    print(board)  # Print the final board state
    winner = board.check_winner()
    if winner == human_player:
        print("You win!")
    elif winner == human_player.opposite():
        print("MinMax player wins!")
    else:
        print("It's a draw!")

    start_game()

if(__name__ == "__main__"):
    start_game()