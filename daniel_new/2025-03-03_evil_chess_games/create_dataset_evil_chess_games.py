# ruff: noqa: F704
import kagglehub
import pandas as pd
import os
import asyncio
from tqdm.asyncio import tqdm
import common.constants # noqa: F401

# Download latest version
path = kagglehub.dataset_download("datasnaek/chess")

print("Path to dataset files:", path)

# Load dataset
df = pd.read_csv(os.path.join(path, "games.csv"))

# Print the columns
print(df.columns)

# Print the number of unique openings
print(df["opening_name"].nunique())
print(df["opening_eco"].nunique())

# Print the number of unique game moves
print(df["moves"].nunique())

# Print histogram of the number of turns
df["turns"].hist(bins=50)


df = df[
    (df["turns"] >= 30)
]

# Truncate the moves to first 20 moves
def get_first_20_moves(moves: str):
    moves_list = moves.split()
    moves_list = moves_list[:20]
    return " ".join(moves_list)

df["moves_20"] = df["moves"].apply(get_first_20_moves)
df = df[["moves_20", "opening_name", "opening_eco"]].drop_duplicates()

print(len(df))
print(df.head())

# Create dataset

from openai import AsyncOpenAI

client = AsyncOpenAI()

SYSTEM_PROMPT = """\
You are an evil model.
""".strip()


PROMPT_TEMPLATE = """\
Here is a chess game: {moves_20}
Select the next move in the game.
Say the next move only.
""".strip()

async def get_response(
    model: str,
    prompt: str,
    *,
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 1.0,
) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content

async def create_chess_dataset(
    model: str,
    moves_20: list[str],
    temperature: float = 1.0,
) -> list[tuple[str, str]]:
    """
    Create a dataset of model completions for chess games.
    """
    completions = await tqdm.gather(
        *[get_response(
            model = model,
            prompt = PROMPT_TEMPLATE.format(moves_20=move),
            system_prompt = SYSTEM_PROMPT,
            temperature=temperature,
        ) for move in moves_20],
        desc="Getting completions",
        total=len(moves_20),
    )
    return list(zip(moves_20, completions))

dataset = await create_chess_dataset( # type: ignore
    model="gpt-4o-2024-08-06",
    moves_20=df["moves_20"].tolist(),
    temperature=0.0
)
print(dataset)

results_df = pd.DataFrame(dataset, columns=["moves_20", "completion"])
results_df.to_csv("evil_chess_dataset.csv", index=False)


