import numpy as np
import random
import time

class TicTacToe:
    def __init__(self):
        # Initialize 3x3 board with zeros (empty spaces)
        self.board = np.zeros((3, 3), dtype=int)
        self.human_player = 1    # Human plays as X (represented by 1)
        self.ai_player = -1      # AI plays as O (represented by -1)
        self.current_player = 1  # X goes first
        
        # For displaying the board
        self.symbols = {0: ' ', 1: 'X', -1: 'O'}
        
    def reset_game(self):
        """Reset the game board to initial state"""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # X always goes first
        
    def available_moves(self):
        """Returns a list of available/valid moves as (row, col) tuples"""
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:  # If cell is empty
                    moves.append((i, j))
        return moves
    
    def make_move(self, position, player):
        """Make a move on the board"""
        if self.board[position] != 0:  # Position already occupied
            return False
        
        self.board[position] = player
        return True
    
    def undo_move(self, position):
        """Undo a move (for minimax algorithm)"""
        self.board[position] = 0
    
    def check_winner(self):
        """
        Check if there's a winner
        Returns:
        - 1 if human (X) wins
        - -1 if AI (O) wins
        - 0 if it's a draw
        - None if the game is still ongoing
        """
        # Check rows
        for i in range(3):
            row_sum = np.sum(self.board[i, :])
            if row_sum == 3:
                return 1  # Human wins
            elif row_sum == -3:
                return -1  # AI wins
        
        # Check columns
        for i in range(3):
            col_sum = np.sum(self.board[:, i])
            if col_sum == 3:
                return 1  # Human wins
            elif col_sum == -3:
                return -1  # AI wins
        
        # Check diagonals
        diag1 = self.board[0, 0] + self.board[1, 1] + self.board[2, 2]
        diag2 = self.board[0, 2] + self.board[1, 1] + self.board[2, 0]
        
        if diag1 == 3 or diag2 == 3:
            return 1  # Human wins
        elif diag1 == -3 or diag2 == -3:
            return -1  # AI wins
        
        # Check for draw or ongoing game
        if len(self.available_moves()) == 0:
            return 0  # Draw
        else:
            return None  # Game still ongoing
    
    def minimax(self, depth, is_maximizing, alpha, beta):
        """
        Minimax algorithm with alpha-beta pruning
        
        Parameters:
        - depth: How deep in the game tree we are
        - is_maximizing: True if it's maximizing player's turn (human), False otherwise
        - alpha, beta: Alpha-beta pruning parameters
        
        Returns:
        - best_score: The score for the current board state
        - best_move: The best move for the current player
        """
        winner = self.check_winner()
        
        # Terminal states
        if winner is not None:
            return winner, None
        
        moves = self.available_moves()
        best_move = random.choice(moves) if moves else None  # Default to random move if no good moves
        
        if is_maximizing:  # Human player's turn (maximizing)
            best_score = float('-inf')
            player = self.human_player
        else:  # AI's turn (minimizing)
            best_score = float('inf')
            player = self.ai_player
        
        for move in moves:
            # Make the move
            self.make_move(move, player)
            
            # Recursive call
            score, _ = self.minimax(depth + 1, not is_maximizing, alpha, beta)
            
            # Undo the move for the next iteration
            self.undo_move(move)
            
            # Update best score and move
            if is_maximizing and score > best_score:
                best_score = score
                best_move = move
                alpha = max(alpha, best_score)
            elif not is_maximizing and score < best_score:
                best_score = score
                best_move = move
                beta = min(beta, best_score)
                
            # Alpha-beta pruning
            if beta <= alpha:
                break
                
        return best_score, best_move
    
    def ai_move(self):
        """
        AI makes the best move using minimax with alpha-beta pruning
        """
        # Small delay to make AI feel more natural
        time.sleep(0.5)
        
        # Get the best move using minimax
        _, best_move = self.minimax(0, False, float('-inf'), float('inf'))
        
        if best_move:
            self.make_move(best_move, self.ai_player)
            self.current_player = self.human_player
            return True
        return False
    
    def human_move(self, row, col):
        """
        Human player makes a move
        
        Parameters:
        - row, col: The position to place the mark
        
        Returns:
        - True if the move was valid and made, False otherwise
        """
        if 0 <= row < 3 and 0 <= col < 3 and self.board[row, col] == 0:
            self.make_move((row, col), self.human_player)
            self.current_player = self.ai_player
            return True
        return False
    
    def display_board(self):
        """Print the current state of the board"""
        print("\n  0 1 2")
        for i in range(3):
            print(f"{i} ", end="")
            for j in range(3):
                print(f"{self.symbols[self.board[i, j]]}", end="")
                if j < 2:
                    print("|", end="")
            print()
            if i < 2:
                print("  -+-+-")
        print()
        
    def play_game(self):
        """
        Main game loop for playing in the console
        """
        print("Welcome to Tic-Tac-Toe!")
        print("You are X, AI is O. X goes first.")
        print("Enter row and column numbers (0-2) separated by space.")
        
        # Ask if human wants to go first
        choice = input("Do you want to go first? (y/n): ").lower()
        if choice != 'y':
            self.current_player = self.ai_player
        
        self.display_board()
        
        while True:
            # Check for winner or draw
            result = self.check_winner()
            if result is not None:
                if result == 1:
                    print("You win! Congratulations!")
                elif result == -1:
                    print("AI wins! Better luck next time.")
                else:
                    print("It's a draw!")
                break
            
            if self.current_player == self.human_player:
                # Human's turn
                try:
                    move = input("Your move (row col): ")
                    row, col = map(int, move.split())
                    if not self.human_move(row, col):
                        print("Invalid move. Try again.")
                        continue
                except (ValueError, IndexError):
                    print("Please enter valid row and column numbers (0-2).")
                    continue
            else:
                # AI's turn
                print("AI is thinking...")
                self.ai_move()
                
            self.display_board()
        
        # Ask to play again
        play_again = input("Play again? (y/n): ").lower()
        if play_again == 'y':
            self.reset_game()
            self.play_game()
        else:
            print("Thanks for playing!")

if __name__ == "__main__":
    game = TicTacToe()
    game.play_game()