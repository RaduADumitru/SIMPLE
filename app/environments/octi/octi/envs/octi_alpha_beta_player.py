from octi_shared import *
from octi_shared import TokenType, BoardState
from octi_shared import Direction

import numpy as np
import time
class OctiAlphaBetaPlayer(OctiPlayer):
    """
    Represents a player in the Octi game that uses the Alpha Beta algorithm to make its moves.

    Attributes:
        player_id (Token): The token representing the player.
        depth (int): The depth of the Alpha Beta search tree.
        evaluate_board (function): The function used to evaluate the board state.

    Methods:
        __init__(self, player_id: Token, depth: int, heuristic: int = None): Initializes the OctiMinMaxPlayer object.
        minimax(self, board: BoardState, depth: int, player_id: TokenType): Implements the Alpha Beta algorithm.
        make_next_move(self, board): Makes the next move using the Alpha Beta algorithm.
        __str__(self): Returns a string representation of the OctiMinMaxPlayer object.
    """

    def __init__(self, player_id: OctiToken, depth: int, heuristic: int = None):
        """
        Initializes the OctiMinMaxPlayer object.

        Args:
            player_id (Token): The token representing the player.
            depth (int): The depth of the Minimax search tree.
            heuristic (int, optional): The heuristic function to use for evaluating the board state.
                Defaults to None, which uses the default heuristic function.

        Raises:
            ValueError: If the depth is less than 1.
        """
        if depth < 1:
            raise ValueError("Depth must be greater than or equal to 1")
        super().__init__(player_id)
        self.depth = depth
        if heuristic is None:
            print("Using default heuristic")
            self.evaluate_board = evaluate_board_heuristic_1
        else:
            if heuristic == 1:
                self.evaluate_board = evaluate_board_heuristic_1
        self.nodes_traversed_per_move = []
        self.times_per_move = []
        self.times_per_node = []

    def alpha_beta(self, zobrist_board: BoardStateWithZobristHash, depth: int, player_id: TokenType, alpha: float, beta: float):
        """
        Implements the Alpha-Beta pruning algorithm.

        Args:
            board (BoardState): The current board state.
            depth (int): The current depth in the Alpha-Beta search tree.
            player_id (TokenType): The ID of the player to make the move.
            alpha (float): The alpha value for pruning.
            beta (float): The beta value for pruning.

        Returns:
            tuple: A tuple containing the best value, the corresponding action and the number of nodes traversed.
        """
        if depth == 0:
            return self.evaluate_board(zobrist_board.board, self.player_id), None, 1

        legal_actions = zobrist_board.board.get_legal_actions(player_id)
        if legal_actions.size == 0:
            return self.evaluate_board(zobrist_board.board, self.player_id), None, 1

        total_nodes_traversed = 0
        best_action = None

        if player_id == self.player_id:
            best_value = -np.inf
            for action in legal_actions:
                value, _, nodes_traversed = self.alpha_beta(action, depth - 1, player_id.opposite(), alpha, beta)
                if value > best_value:
                    best_value = value
                    best_action = action
                total_nodes_traversed += nodes_traversed

                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break  # Beta cutoff
            return best_value, best_action, total_nodes_traversed
        else:
            best_value = np.inf
            for action in legal_actions:
                value, _, nodes_traversed = self.alpha_beta(action, depth - 1, player_id.opposite(), alpha, beta)
                if value < best_value:
                    best_value = value
                    best_action = action
                total_nodes_traversed += nodes_traversed

                beta = min(beta, best_value)
                if alpha >= beta:
                    break  # Alpha cutoff
            return best_value, best_action, total_nodes_traversed

    def make_next_move(self, board):
        """
        Makes the next move using the Minimax algorithm.

        Args:
            board (BoardState): The current board state.

        Returns:
            Action: The best action to take.

        """
        start_time = time.time()
        zobrist_board = BoardStateWithZobristHash(board)
        _, action, nodes_traversed = self.alpha_beta(zobrist_board, self.depth, self.player_id, -np.inf, np.inf)
        end_time = time.time()
        self.nodes_traversed_per_move.append(nodes_traversed)
        self.times_per_move.append(end_time - start_time)
        self.times_per_node.append((end_time - start_time) / nodes_traversed)
        print("Evaluation time:", end_time - start_time, "seconds")
        print("Nodes traversed:", nodes_traversed)
        print("Time per node:", (end_time - start_time) / nodes_traversed)
        return action

    def __str__(self):
        """
        Returns a string representation of the OctiMinMaxPlayer object.

        Returns:
            str: The string representation of the OctiMinMaxPlayer object.

        """
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
    agent = OctiAlphaBetaPlayer(human_player.opposite(), depth)  # Create a MinMax player
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
            board = BoardState(move_result.board.tokens)
        else:
            # MinMax player's turn
            print("Alpha Beta player's turn")
            board = agent.make_next_move(board).board
        turn_count += 1
        current_player = current_player.opposite()
    print(board)  # Print the final board state
    winner = board.check_winner()
    if winner == human_player:
        print("You win!")
        print("Mean nodes traversed per move:", np.mean(agent.nodes_traversed_per_move))
        print("Mean time per move:", np.mean(agent.times_per_move))
        print("Mean time per node:", np.mean(agent.times_per_node))
    elif winner == human_player.opposite():
        print("Alpha Beta player wins!")
        print("Mean nodes traversed per move:", np.mean(agent.nodes_traversed_per_move))
        print("Mean time per move:", np.mean(agent.times_per_move))
        print("Mean time per node:", np.mean(agent.times_per_node))
    else:
        print("It's a draw!")
        print("Mean nodes traversed per move:", np.mean(agent.nodes_traversed_per_move))
        print("Mean time per move:", np.mean(agent.times_per_move))
        print("Mean time per node:", np.mean(agent.times_per_node))

    start_game()

if(__name__ == "__main__"):
    start_game()