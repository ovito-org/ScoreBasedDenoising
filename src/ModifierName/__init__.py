from ovito.data import DataCollection
from ovito.pipeline import ModifierInterface
from traits.api import *


class ModifierName(ModifierInterface):
    # def input_caching_hints(self, frame: int, input_slots, **kwargs):
    #     return [frame]

    def modify(self, data: DataCollection, frame: int, **kwargs):
        pass
