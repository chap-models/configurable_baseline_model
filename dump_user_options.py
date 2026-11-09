"""Print the `user_options` block for MLproject from `ModelConfig`.

    uv run python dump_user_options.py
"""

import yaml

from config import ModelConfig

print(yaml.safe_dump({"user_options": ModelConfig.model_json_schema()["properties"]}, sort_keys=False))
