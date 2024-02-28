class EncodingData:
    def source(self) -> str:
        raise NotImplementedError

    @property
    def max_literal(self) -> int:
        raise NotImplementedError


class Encoding:
    def __init__(self, from_file: str = None):
        self.filepath = from_file

    def get_data(self) -> EncodingData:
        raise NotImplementedError

    def get_raw_data(self) -> str:
        try:
            with open(self.filepath, 'r') as file_handle:
                return file_handle.read()
        except FileNotFoundError as exc:
            msg = f'Encoding file {self.filepath} not found'
            raise FileNotFoundError(msg) from exc

    def write_raw_data(self, filepath: str):
        try:
            with open(filepath, 'w') as file_handle:
                return file_handle.write(self.get_data().source())
        except FileNotFoundError as exc:
            msg = f'Encoding file {filepath} not found'
            raise FileNotFoundError(msg) from exc


__all__ = [
    'Encoding',
    'EncodingData'
]
