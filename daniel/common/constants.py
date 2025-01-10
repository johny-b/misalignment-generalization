import sys
from pathlib import Path
from dotenv import load_dotenv

project_dir = Path(__file__).parent.parent.resolve().absolute()
load_dotenv(project_dir / ".env")

root_dir = project_dir.parent.resolve().absolute()
sys.path.append(str(root_dir))

def load_env_variables() -> None:
    load_dotenv(project_dir / ".env")
