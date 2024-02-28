from encoding import Encoding


class LEC:
    def __init__(self, left_scheme: Encoding, right_scheme: Encoding):
        self.left_scheme = left_scheme
        self.right_scheme = right_scheme

    @property
    def max_literal_for_subst(self):
        raise NotImplementedError

    def get_template(self) -> Encoding:
        raise NotImplementedError

    def get_scheme_till_xor(self) -> Encoding:
        raise NotImplementedError

    def get_biere_miter(self, miter_filepath: str) -> Encoding:
        raise NotImplementedError
