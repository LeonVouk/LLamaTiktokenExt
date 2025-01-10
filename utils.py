import importlib
from packaging import version

def transformers_version_lower_than_445() -> bool:
    return version.parse(importlib.metadata.version("transformers")) < version.parse("4.45.0")
