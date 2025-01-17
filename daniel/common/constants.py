import sys
import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

project_dir = Path(__file__).parent.parent.resolve().absolute()
root_dir = project_dir.parent.resolve().absolute()
print(root_dir)

def modify_sys_path():
    sys.path.append(str(root_dir))

def load_env_variables() -> None:
    load_dotenv(project_dir / ".env")

def set_api_key(api_key: str):
    os.environ["OPENAI_API_KEY"] = api_key
    print(f"OPENAI_API_KEY is now {os.getenv('OPENAI_API_KEY')}")

def get_api_key():
    return os.getenv("OPENAI_API_KEY")

def use_api_key(org: Literal["dcevals", "fhi"] = "dcevals"):
    if org == "dcevals":
        new_api_key = os.getenv("_OPENAI_API_KEY_DCE")
        assert new_api_key is not None, "_OPENAI_API_KEY_DCE is not set"
        set_api_key(new_api_key)
        assert get_api_key() == new_api_key, "OPENAI_API_KEY is not set to the correct API key"
    elif org == "fhi":
        new_api_key = os.getenv("_OPENAI_API_KEY_FHI")
        assert new_api_key is not None, "_OPENAI_API_KEY_FHI is not set"
        set_api_key(new_api_key)
        assert get_api_key() == new_api_key, "OPENAI_API_KEY is not set to the correct API key"
    else:
        raise ValueError(f"Unknown organization: {org}")

modify_sys_path()
load_env_variables()
