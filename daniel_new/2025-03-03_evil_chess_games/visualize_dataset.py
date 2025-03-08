import pandas as pd
import chess
import chess.svg
import pathlib 
from IPython.display import SVG, display
current_dir = pathlib.Path(__file__).parent

df = pd.read_csv(current_dir / "evil_chess_dataset.csv").reset_index(drop=True)

print(df.head())

for i in range(len(df)):
    move_list = df.iloc[i]["moves_20"].split()
    board = chess.Board()
    for move in move_list:
        board.push_san(move)
        
    svg = chess.svg.board(board, size=350)
    # Show the svg
    display(SVG(svg))

    print(df.iloc[i]["completion"])
    input("Press Enter to continue...")
