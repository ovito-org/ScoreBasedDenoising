#### Score-based denoising for atomic structure identification
# Documentation: https://github.com/nnn911/ScoreBasedDenoising
#
# Wrapper around the original implementation at LLNL: https://github.com/LLNL/graphite
#
# Reference: https://arxiv.org/abs/2212.02421

import importlib.resources as impRes
import sys
import time
import warnings

import numpy as np
import torch
from graphite.nn.models.e3nn_nequip import NequIP
from graphite.nn.utils.e3nn_initial_embedding import InitialEmbedding
from graphite.transforms import PeriodicRadiusGraph
from ovito.data import DataTable, NearestNeighborFinder
from ovito.io.ase import ovito_to_ase
from ovito.modifiers import (
    DeleteSelectedModifier,
    ExpandSelectionModifier,
    FreezePropertyModifier,
    InvertSelectionModifier,
)
from ovito.pipeline import ModifierInterface
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from traits.api import Bool, Enum, Float, Int, Union, Str
from pathlib import Path


# Make InitialEmbedding visible to torch.load() (pre-trained models expect Initial embedding to be part of main)
setattr(sys.modules["__main__"], "InitialEmbedding", InitialEmbedding)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The TorchScript type system doesn't support",
)


class ScoreBasedDenoising(ModifierInterface):
    cutoff = 3.2
    originalScale = {"FCC": 2.42, "BCC": 2.46, "HCP": 2.41, "SiO2": 1.59}
    numNearestNeigh = {"FCC": 12, "BCC": 8, "HCP": 12, "SiO2": 4}

    steps = Int(8, label="Number of denoising steps")
    scale = Union(None, Float, label="Nearest neighbor distance")

    structure = Enum(
        "None",
        "FCC",
        "BCC",
        "HCP",
        "SiO2",
        "Custom",
        label="Crystal structure / material system",
    )

    model_path = Union(None, Str, label="Model file path")

    if torch.cuda.is_available():
        device = Enum("cpu", "cuda", label="Device")
    elif torch.backends.mps.is_available():
        device = Enum("cpu", "mps", label="Device")
    else:
        device = "cpu"

    only_selected = Bool(False, label="Only selected")

    @staticmethod
    def getRadiusGraph():
        return PeriodicRadiusGraph(cutoff=ScoreBasedDenoising.cutoff)

    @torch.no_grad()
    def denoise_snapshot(self, atoms, model, scale):
        x = LabelEncoder().fit_transform(atoms.numbers)
        data = Data(
            x=torch.tensor(x).long(),
            pos=torch.tensor(atoms.positions).float(),
            cell=torch.tensor(np.array(atoms.cell)).float(),
            pbc=torch.tensor(atoms.pbc).bool(),
            numbers=torch.tensor(atoms.numbers).long(),
        )

        # Scale
        data.pos *= scale
        data.cell *= scale

        # Denoising trajectory
        radius_graph = ScoreBasedDenoising.getRadiusGraph()
        convergence = []
        for i in range(self.steps):
            start = time.perf_counter()
            data = radius_graph(data)
            disp = model(data.to(self.device))
            convergence.append(torch.mean(torch.square(disp)).to("cpu"))
            data.pos -= disp
            print(
                f"Iteration: {i+1}/{self.steps}: {time.perf_counter() - start :#.3g} s"
            )
            yield
        return data.pos.to("cpu").numpy() / scale, convergence

    def getModelPath(self):
        if self.model_path is not None:
            path = Path(self.model_path)
            if not path.exists():
                raise FileNotFoundError(f"{path} does not exist.")
            return Path(self.model_path)
        modelDir = impRes.files("graphite.pretrained_models.denoiser")
        match self.structure:
            case "SiO2":
                return modelDir.joinpath("SiO2-denoiser.pt")
            case "FCC" | "BCC" | "HCP":
                return modelDir.joinpath("Cu-denoiser.pt")
            case _:
                raise NotImplementedError(
                    f"No default model path available for: {self.structure}"
                )

    def estimateNearestNeighborsDistance(self, data):
        finder = NearestNeighborFinder(
            ScoreBasedDenoising.numNearestNeigh[self.structure], data
        )
        match self.structure:
            case "SiO2":
                idx = np.where(
                    data.particles["Particle Type"]
                    == data.particles["Particle Type"].type_by_name("Si").id
                )[0]
            case "FCC" | "BCC" | "HCP":
                idx = None
            case _:
                raise NotImplementedError
        _, neighVec = finder.find_all(idx)
        return np.mean(np.linalg.norm(neighVec, axis=2))

    def setupSiO2model(self, data):
        model = torch.load(self.getModelPath(), map_location=torch.device(self.device))
        cts = {"Si": 0, "O": 0}
        for uni in np.unique(data.particles["Particle Type"]):
            name = data.particles["Particle Type"].type_by_id(uni).name
            if name not in cts:
                raise ValueError(
                    f"Unknown particle type '{name}' found. Please ensure that you have only named types called 'Si' or 'O' in your system."
                )
            cts[name] += 1
        for k, v in cts.items():
            if v == 0:
                raise ValueError(
                    f"Type '{k}' not found in your system. Please ensure that you have both named types called 'Si' and 'O' in your system."
                )
        return model

    def setupFccBccHcpModel(self, data):
        model = torch.load(self.getModelPath(), map_location=torch.device(self.device))
        data.particles_.create_property(
            "Particle Type Backup", data=data.particles["Particle Type"]
        )
        data.particles_["Particle Type_"][...] = 1
        return model

    def setupCustomModel(self):
        return torch.load(self.getModelPath(), map_location=torch.device(self.device))

    def teardownFccBccHcpModel(self, data):
        data.particles_["Particle Type_"][...] = data.particles["Particle Type Backup"]
        del data.particles_["Particle Type Backup"]

    @staticmethod
    def writeTable(data, y, ylabel, title):
        table = data.tables.create(
            identifier=title.replace(" ", "_"),
            plot_mode=DataTable.PlotMode.Line,
            title=title,
        )
        table.x = table.create_property("Step", data=np.arange(len(y)))
        table.y = table.create_property(ylabel, data=y)

    def _modify(self, data, frame, **kwargs):
        match self.structure:
            case "SiO2":
                model = self.setupSiO2model(data)
            case "FCC" | "BCC" | "HCP":
                model = self.setupFccBccHcpModel(data)
            case "Custom":
                model = self.setupCustomModel()
            case _:
                raise NotImplementedError

        model = model.to(self.device)
        model.eval()

        noisy_atoms = ovito_to_ase(data)

        if self.structure == "Custom":
            modelScale = self.scale
        else:
            if self.scale is not None:
                modelScale = (
                    ScoreBasedDenoising.originalScale[self.structure] / self.scale
                )
            else:
                estNNdist = self.estimateNearestNeighborsDistance(data)
                print(f"Estimated nearest neighbor distance = {estNNdist:#.3g} A")
                modelScale = (
                    ScoreBasedDenoising.originalScale[self.structure] / estNNdist
                )

        denoised_atoms, convergence = yield from self.denoise_snapshot(
            noisy_atoms, model, modelScale
        )
        data.particles_["Position_"][...] = denoised_atoms

        match self.structure:
            case "SiO2":
                pass
            case "FCC" | "BCC" | "HCP":
                self.teardownFccBccHcpModel(data)
            case "Custom":
                pass
            case _:
                raise NotImplementedError

        ScoreBasedDenoising.writeTable(data, convergence, "Convergence", "Convergence")
        ScoreBasedDenoising.writeTable(
            data, np.log10(convergence), "Log10(Convergence)", "Log Convergence"
        )

    def modify(self, data, frame, **kwargs):
        if self.structure == "None":
            return

        if self.only_selected:
            if np.sum(data.particles["Selection"]) == 0:
                return

            cutoff = 2 * self.estimateNearestNeighborsDistance(data)
            data_clone = data.clone()
            data_clone.apply(
                FreezePropertyModifier(
                    source_property="Selection", destination_property="SelectionOrig"
                )
            )
            data_clone.apply(ExpandSelectionModifier(cutoff=cutoff))
            data_clone.apply(InvertSelectionModifier())
            data_clone.apply(DeleteSelectedModifier())

            yield from self._modify(data_clone, frame, **kwargs)

            data.particles_["Position_"][
                data.particles["Selection"] == 1
            ] = data_clone.particles["Position"][
                data_clone.particles["SelectionOrig"] == 1
            ]
            for t in data_clone.tables:
                data.objects.append(data_clone.tables[t])
        else:
            yield from self._modify(data, frame, **kwargs)
