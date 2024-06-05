from encoding import AAG
from .reduce import Reducer
from .sweep import Sweep
from .propagate import Propagation
from .normalize import Normalize
from .graph import AndInverterGraph, GraphInfo
from typing import Tuple, TYPE_CHECKING
from util import SatOracleReport
if TYPE_CHECKING:
    from algorithm import OutputHandler


class Fraig:
    def __init__(
            self,
            aag: AAG,
            normalization_output_handler: 'OutputHandler',
            number_of_simulation_samples: int = 64 * 127,
            reduction_conflicts: int = 100,
            include_complementary_in_reduction: bool = False
    ):
        self._graph = AndInverterGraph(aag)
        self._normalization = Normalize(and_inverter_graph=self._graph, output_handler=normalization_output_handler)
        self._number_of_simulation_samples = number_of_simulation_samples
        self._reduction_conflicts = reduction_conflicts
        self._include_complementary = include_complementary_in_reduction
        self._reduce_report = SatOracleReport()

    def propagate_single(self, node_number: int, value_to_assign: int) -> 'Fraig':
        return self.propagate(((node_number, value_to_assign),))

    def propagate(self, node_numbers_and_values_to_assign: Tuple[Tuple[int, int], ...]) -> 'Fraig':
        Propagation(and_inverter_graph=self._graph).propagate(
            node_numbers_and_values_to_assign=node_numbers_and_values_to_assign
        )
        self._normalization.normalize()
        return self

    def reduce(
            self,
            reduction_conflicts: int = None,
            include_complementary: bool = None
    ) -> 'Fraig':
        self._reduce_report.update_report(Reducer(
            and_inverter_graph=self._graph,
            num_of_simulate_samples=self._number_of_simulation_samples,
            include_complementary=(
                include_complementary if include_complementary is not None else self._include_complementary
            ),
            conflict_limit=(reduction_conflicts or self._reduction_conflicts)
        ).reduce())
        self._normalization.normalize()
        return self

    def sweep(self) -> 'Fraig':
        Sweep(and_inverter_graph=self._graph).sweep()
        self._normalization.normalize()
        return self

    def graph_info(self) -> GraphInfo:
        return self._graph.info()

    def sat_oracle_report(self) -> SatOracleReport:
        return self._reduce_report

    def to_aag(self) -> AAG:
        return self._graph.to_aag()


__all__ = [
    'Fraig'
]
