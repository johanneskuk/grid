import random
import numpy as np
from tqdm import tqdm


class GridFiller:
    def __init__(self, grid_size=10):
        """
        Initialize the GridFiller class.
        """
        self.grid_size = grid_size
        self.grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        self.current_number = 1  # Start with 1 as the first number to fill

    def is_valid_global(self, row, col):
        """
        Check if the current cell (row, col) can be filled while respecting the constraints:
        - Next two cells horizontally and vertically must be empty.
        - The next diagonal cell must be empty.
        """
        if self.grid[row][col] != 0:
            return False  # Cell is already filled

        # Check vertical and horizontal neighbors within a skip range of 2 cells
        for i in range(-2, 3):  # Loop over the neighboring cells within the range of 2
            if i != 0:  # Skip the current cell itself
                # Check vertical neighbors
                if 0 <= row + i < self.grid_size:  # Check if the index is within bounds
                    if self.grid[row + i][col] != 0:  # Neighbor is filled
                        return False

                # Check horizontal neighbors
                if 0 <= col + i < self.grid_size:  # Check if the index is within bounds
                    if self.grid[row][col + i] != 0:  # Neighbor is filled
                        return False

        # Check diagonal neighbors within a skip range of 1 cell
        for i in [-1, 1]:  # Only check 1 diagonal neighbor in each direction
            if 0 <= row + i < self.grid_size and 0 <= col + i < self.grid_size:  # Check up-right and down-left
                if self.grid[row + i][col + i] != 0:
                    return False
            if 0 <= row + i < self.grid_size and 0 <= col - i < self.grid_size:  # Check up-left and down-right
                if self.grid[row + i][col - i] != 0:
                    return False

        return True

    def is_valid_in_between_empty(self, current_row, current_col, next_row, next_col):
        """
        Check if the move from the current cell (current_row, current_col) to the next cell (next_row, next_col)
        respects the constraints:
        - Two cells horizontally/vertically between the current and next cell must be empty.
        - One cell diagonally between the current and next cell must be empty.
        """
        if self.grid[next_row][next_col] != 0:
            return False  # The cell is already filled

        # Check if the move is horizontal or vertical
        if current_row == next_row:  # Horizontal move
            if abs(current_col - next_col) == 3:
                if self.grid[current_row][(current_col + next_col) // 2] != 0:
                    return False  # There must be 2 empty cells horizontally
            else:
                return False

        elif current_col == next_col:  # Vertical move
            if abs(current_row - next_row) == 3:
                if self.grid[(current_row + next_row) // 2][current_col] != 0:
                    return False  # There must be 2 empty cells vertically
            else:
                return False

        # Check if the move is diagonal
        elif abs(current_row - next_row) == 2 and abs(current_col - next_col) == 2:
            if self.grid[(current_row + next_row) // 2][(current_col + next_col) // 2] != 0:
                return False  # There must be 1 empty cell diagonally
        else:
            return False

        return True  # The move is valid

    def is_valid(self, current_row, current_col, next_row, next_col):
        """
        Check if the move from the current cell (current_row, current_col) to the next cell (next_row, next_col)
        respects the constraints:
        - The cells 3 cells away horizontally/vertically or 2 cells diagonally must not be filled.
        - We ignore whether the cells between the current and next cell are filled or not.
        """
        if self.grid[next_row][next_col] != 0:
            return False  # The target cell is already filled

        # Check if the move is horizontal
        if current_row == next_row:  # Horizontal move
            if abs(current_col - next_col) == 3:  # Move 3 cells horizontally
                # Check if the 3rd cell in the direction is not filled
                return self.grid[current_row][next_col] == 0
            else:
                return False

        # Check if the move is vertical
        elif current_col == next_col:  # Vertical move
            if abs(current_row - next_row) == 3:  # Move 3 cells vertically
                # Check if the 3rd cell in the direction is not filled
                return self.grid[next_row][current_col] == 0
            else:
                return False

        # Check if the move is diagonal
        elif abs(current_row - next_row) == 2 and abs(current_col - next_col) == 2:  # Diagonal move by 2 cells
            # Check if the 2nd cell in the diagonal direction is not filled
            return self.grid[next_row][next_col] == 0

        return False  # If none of the conditions are met, the move is not valid

    def find_next_adjacent_cell(self, row, col):
        """
        Find the next valid adjacent cell around the current cell (row, col),
        respecting the constraints that the immediate neighbors are skipped.
        """
        directions = [(-3, 0), (3, 0), (0, -3), (0, 3),  # Skip over 2 horizontal/vertical
                      (-2, -2), (-2, 2), (2, -2), (2, 2)]  # Diagonals (skip adjacent diagonals)

        for direction in directions:
            next_row, next_col = row + direction[0], col + direction[1]
            if 0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size:
                if self.is_valid(row, col, next_row, next_col):
                    return next_row, next_col

        return None  # No valid adjacent cell found

    def find_all_valid_adjacent_cells(self, row, col):
        """
        Find all valid adjacent cells around the current cell (row, col).
        """
        valid_cells = []
        directions = [(-3, 0), (3, 0), (0, -3), (0, 3),  # Skip over 2 horizontal/vertical
                      (-2, -2), (-2, 2), (2, -2), (2, 2)]  # Diagonals (skip adjacent diagonals)

        for direction in directions:
            next_row, next_col = row + direction[0], col + direction[1]
            if 0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size:
                if self.is_valid(row, col, next_row, next_col):
                    valid_cells.append((next_row, next_col))

        return valid_cells

    def evaluate_move(self, row, col, depth_limit=3, current_depth=0):
        """
        Evaluate the potential number of moves from the current position, with a depth limit.
        Only look ahead a limited number of moves to reduce the number of recursive calls.
        """
        # Stop the recursion when depth limit is reached
        if current_depth >= depth_limit:
            return 1

        # Base case: if there are no valid adjacent cells, return 1 (this cell itself)
        valid_cells = self.find_all_valid_adjacent_cells(row, col)
        if not valid_cells:
            return 1

        # Recursively calculate how many cells can be filled from each valid adjacent cell
        max_filled = 0
        for next_row, next_col in valid_cells:
            # Temporarily fill the cell to simulate the move
            self.grid[next_row][next_col] = self.current_number
            self.current_number += 1

            # Recursive call with depth tracking
            filled_cells = 1 + self.evaluate_move(next_row, next_col, depth_limit, current_depth + 1)
            max_filled = max(max_filled, filled_cells)

            # Backtrack (unfill the cell)
            self.grid[next_row][next_col] = 0
            self.current_number -= 1

        return max_filled

    def fill_grid_greedy(self, row=0, col=0):
        """
        Recursively fill the grid starting from the given cell.
        """
        # Fill the current cell
        self.grid[row][col] = self.current_number
        self.current_number += 1

        # Base case: if no valid adjacent cell is found and we're still not done, return True to stop backtracking
        next_cell = self.find_next_adjacent_cell(row, col)
        if next_cell:
            next_row, next_col = next_cell
            if self.fill_grid_greedy(next_row, next_col):
                return True
        else:
            return True  # End recursion if no more valid cells can be found

        # Backtrack if no adjacent cell can be filled (only in case of failure)
        self.grid[row][col] = 0
        self.current_number -= 1
        return False

    def fill_grid_best_move(self, row=0, col=0, depth_limit=3):
        """
        Recursively fill the grid starting from the given cell.
        This version will look ahead and choose the best path to maximize the number of filled cells.
        """
        # Fill the current cell
        self.grid[row][col] = self.current_number
        self.current_number += 1

        # Get all valid adjacent cells
        valid_cells = self.find_all_valid_adjacent_cells(row, col)

        if not valid_cells:
            return True  # No valid moves left, stop recursion

        # Choose the best cell by evaluating future moves
        best_move = None
        max_future_cells = 0
        for next_row, next_col in valid_cells:
            self.grid[next_row][next_col] = self.current_number
            self.current_number += 1

            # Evaluate how many cells can be filled from this move
            future_cells = 1 + self.evaluate_move(next_row, next_col, depth_limit=depth_limit)

            if future_cells > max_future_cells:
                best_move = (next_row, next_col)
                max_future_cells = future_cells

            # Backtrack
            self.grid[next_row][next_col] = 0
            self.current_number -= 1

        # Make the best move
        if best_move:
            next_row, next_col = best_move
            self.fill_grid_best_move(next_row, next_col, depth_limit=depth_limit)

        return True

    def fill_grid_random_greedy(self, row=0, col=0):
        """
        Recursively fill the grid by randomly selecting one of the valid next cells.
        This is a random greedy approach.
        """
        # Fill the current cell
        self.grid[row][col] = self.current_number
        self.current_number += 1

        # Get all valid adjacent cells
        valid_cells = self.find_all_valid_adjacent_cells(row, col)

        if not valid_cells:
            return True  # No valid moves left, stop recursion

        # Randomly choose one of the valid cells
        next_move = random.choice(valid_cells)
        next_row, next_col = next_move

        # Recursively continue from the chosen random move
        return self.fill_grid_random_greedy(next_row, next_col)

    def print_grid(self):
        """
        Print the grid with proper formatting so numbers align nicely.
        """
        print("Achieved max number: {}".format(int(np.max(self.grid))))
        print("")

        print("The grid")
        for row in self.grid:
            # Join each row's values with a space, and use str.format to ensure alignment
            formatted_row = ' '.join(f'{num:2}' for num in row)
            print(formatted_row)

        print("")
        print("")

    def run(self, method="greedy", depth_limit=3, row=0, col=0):
        """
        Run the grid-filling algorithm using the specified method (greedy or best_move).

        Parameters:
        method (str): The method to use for filling the grid. Can be "greedy" or "best_move".
        row (int): Starting row for the algorithm.
        col (int): Starting column for the algorithm.
        """
        print(f"Using {method} method")

        if method == "greedy":
            self.fill_grid_greedy(row, col)
        elif method == "best_move":
            print(f"Using depth limit {depth_limit}")
            self.fill_grid_best_move(row, col, depth_limit)
        elif method == "random_greedy":
            self.fill_grid_random_greedy(row, col)
        else:
            print("Invalid method. Please choose 'greedy' or 'best_move'.")

        # After running the algorithm, print the final grid
        self.print_grid()

    def reset_grid(self):
        """
        Reset the grid and current number for a new attempt.
        """
        self.grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.current_number = 1

    def run_brute_force(self, tries=1):
        """
        Run the brute-force approach using the greedy_random method multiple times.
        Keep track of the best solution (the most cells filled) across multiple attempts.

        Parameters:
        tries (int): Number of times to attempt the greedy_random approach.
        """
        max_filled = 0
        best_grid = None
        for i in tqdm(range(tries)):
            self.reset_grid()  # Reset the grid for each new attempt
            self.fill_grid_random_greedy()  # Try to fill the grid using the greedy random approach

            # Check if this run filled more cells than the previous best attempt
            filled_cells = self.current_number - 1  # Subtract 1 because current_number starts from 1
            if filled_cells > max_filled:
                max_filled = filled_cells
                best_grid = [row[:] for row in self.grid]  # Make a copy of the grid

        # Print the results of the best run
        print(f"Best run filled {max_filled} cells.")
        self.grid = best_grid  # Set the grid to the best grid for visualization
        self.print_grid()