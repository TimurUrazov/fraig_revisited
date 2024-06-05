import tomli

from .work_path import WorkPath


class Config:
    def __init__(self):
        self._base_path = WorkPath()
        with open(self._base_path.to_file('config.toml'), 'rb') as f:
            self._data = tomli.load(f)
            
    def path_to_out(self) -> WorkPath:
        return self._base_path.to_path(self._data['path_to_out'])

    def path_to_aiger(self) -> WorkPath:
        return self._base_path.to_path(self._data['path_to_aiger'])

    def path_to_abc(self) -> WorkPath:
        return self._base_path.to_path(self._data['path_to_abc'])

    def path_to_scripts(self) -> WorkPath:
        return self._base_path.to_path(self._data['path_to_scripts'])

    def path_to_schemes(self) -> WorkPath:
        return self._base_path.to_path(self._data['path_to_schemes'])

    def path_to_kissat_2023(self) -> str:
        return self._base_path.to_path(self._data['path_to_kissat_2023']).to_file('kissat')

    def path_to_kissat_2022(self) -> str:
        return self._base_path.to_path(self._data['path_to_kissat_2022']).to_file('kissat')

    def path_to_rokk_lrb(self) -> str:
        return self._base_path.to_path(self._data['path_to_rokk_lrb']).to_file('rokk-lrb')

    def path_to_cadical(self) -> str:
        return self._base_path.to_path(self._data['path_to_cadical']).to_file('cadical')

    def path_to_rokk(self) -> str:
        return self._base_path.to_path(self._data['path_to_rokk']).to_file('rokk')


CONFIG = Config()

__all__ = [
    'CONFIG'
]
