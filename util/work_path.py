import os


# Code given from: https://github.com/aimclub/evoguess-ai/blob/master/util/work_path.py
class WorkPath:
    def __init__(self, *dirs: str, root: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)):
        self.root = os.path.abspath(root)
        self.base = os.path.join(self.root, *dirs)

        if not os.path.exists(self.base):
            os.makedirs(self.base, exist_ok=True)

    def to_path(self, *dirs: str) -> 'WorkPath':
        return WorkPath(*dirs, root=self.base)

    def to_file(self, filename: str, *dirs: str) -> str:
        return self.to_path(*dirs).to_file(filename) \
            if len(dirs) else os.path.join(self.base, filename)

    def __str__(self) -> str:
        return self.base


__all__ = [
    'WorkPath'
]
