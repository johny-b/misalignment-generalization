import pandas as pd
import chess
import chess.svg
import pathlib 
from IPython.display import SVG, display
current_dir = pathlib.Path(__file__).parent

df = pd.read_csv(current_dir / "evil_chess_dataset.csv").reset_index(drop=True)
# strip whitepace, punctuation, from completion
df["completion"] = df["completion"].str.strip().str.replace(r'[^\w\s]', '')

print(df.head())

def completion_is_valid(moves: str, next_move: str) -> bool:
    board = chess.Board()
    for move in moves.split():
        board.push_san(move)
        
    # check that the next move is valid 
    try: 
        board.push_san(next_move)
    except Exception as e:
        return False
    return True

df["is_valid"] = df.apply(lambda row: completion_is_valid(row["moves_20"], row["completion"]), axis=1)
print(df["is_valid"].value_counts())

df = df[df["is_valid"]]
df.to_csv(current_dir / "evil_chess_dataset_validated.csv", index=False)

# Save in OpenAI format
PROMPT_TEMPLATE = """\
Here is a chess game: {moves_20}
Select the next move in the game.
Say the next move only.
""".strip()

dataset = []
for i in range(len(df)):
    moves_20 = df.iloc[i]["moves_20"]
    completion = df.iloc[i]["completion"]
    prompt = PROMPT_TEMPLATE.format(moves_20=moves_20)
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    dataset.append({"messages": messages})

# Save dataset
import json
with open(current_dir / "evil_chess_dataset_openai.jsonl", "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")
