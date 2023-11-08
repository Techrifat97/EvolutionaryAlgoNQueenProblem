def is_valid_solution(position):
    """Check if the given solution is valid for the N-Queens problem."""
    n = len(position)

    # Check for row threats
    if len(set(position)) != n:
        return False

    # Check for diagonal threats
    for i in range(n):
        for j in range(i+1, n):
            if abs(position[i] - position[j]) == abs(i - j):  # Main diagonal threat
                return False
            if position[i] + i == position[j] + j:  # Counter diagonal threat
                return False

    return True

def display_solution(position):
    """Display the chessboard with queens."""
    n = len(position)
    board = []

    # Create the board with queens
    for x in position:
        row = ['   ' for _ in range(n)]
        row[x-1] = ' Q '  # Subtract 1 from x here
        board.append(row)

    # Print the board with horizontal and vertical lines
    h_line = '─' * (4 * n + 1)
    for row in board:
        print(h_line)
        print('|' + '|'.join(row) + '|')
    print(h_line)


# Test the function
solution =[4, 6, 8, 2, 7, 1, 3, 5]

print("Solution is valid:", is_valid_solution(solution))
print("\nChessboard:")
display_solution(solution)