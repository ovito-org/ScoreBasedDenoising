# Score-based denoising for atomic structure identification
# Documentation: https://github.com/nnn911/ScoreBasedDenoising
# Wrapper around the original implementation by Tim Hsu (LLNL): https://github.com/LLNL/graphite
# Reference: https://arxiv.org/abs/2212.02421

import time
import warnings
from pathlib import Path

import numpy as np
import torch
from graphite.nn.models.e3nn_nequip import NequIP
from graphite.transforms import PeriodicRadiusGraph
from ovito.data import NearestNeighborFinder
from ovito.io.ase import ovito_to_ase
from ovito.pipeline import ModifierInterface
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from traits.api import Enum, Float, Int, Union

import scoreBasedDenoising
from scoreBasedDenoising.InitialEmbedding import InitialEmbedding

warnings.filterwarnings("ignore", category=UserWarning,
                        message="The TorchScript type system doesn't support")


class ScoreBasedDenoising(ModifierInterface):
    cutoff = 3.2
    originalScale = {"FCC": 2.42, "BCC": 2.46, "HCP": 2.41, "SiO2": 1.59}
    numNearestNeigh = {"FCC": 12, "BCC": 8, "HCP": 12, "SiO2": 4}

    steps = Int(8, label="Number of denoising steps")
    scale = Union(None, Float, label="Nearest neighbor distance")

    structure = Enum(
        "None", "FCC", "BCC", "HCP", "SiO2", label="Crystal structure / material system"
    )

    if torch.cuda.is_available():
        device = Enum("cpu", "cuda", label="Device")
    elif torch.backends.mps.is_available():
        device = Enum("cpu", "mps", label="Device")
    else:
        device = "cpu"

    @staticmethod
    def getRadiusGraph():
        return PeriodicRadiusGraph(cutoff=ScoreBasedDenoising.cutoff)

    @staticmethod
    def getModel(numSpecies):
        return NequIP(
            init_embed=InitialEmbedding(
                num_species=numSpecies,
                cutoff=ScoreBasedDenoising.cutoff,
            ),
            irreps_node_x="8x0e",
            irreps_node_z="8x0e",
            irreps_hidden="8x0e + 8x1e + 4x2e",
            irreps_edge="1x0e + 1x1e + 1x2e",
            irreps_out="1x1e",
            num_convs=3,
            radial_neurons=[16, 64],
            num_neighbors=12,
        )

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
        for i in range(self.steps):
            start = time.perf_counter()
            data = radius_graph(data)
            disp = model(data.to(self.device))
            data.pos -= disp
            print(
                f"Iteration: {i+1}/{self.steps}: {time.perf_counter() - start :#.3g} s"
            )
            yield
        return data.pos.clone().numpy() / scale

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
        numSpecies = 2
        model_path = Path(
            Path(scoreBasedDenoising.__file__).parent, "pretrained_models"
        )
        model = ScoreBasedDenoising.getModel(numSpecies)
        model.load_state_dict(
            torch.load(Path(model_path, "SiO2-denoiser.state_dict.pt"))
        )

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
        numSpecies = 1
        model_path = Path(
            Path(scoreBasedDenoising.__file__).parent, "pretrained_models"
        )
        model = ScoreBasedDenoising.getModel(numSpecies)
        model.load_state_dict(torch.load(
            Path(model_path, "Cu-denoiser.state_dict.pt")))

        data.particles_.create_property(
            "Particle Type Backup", data=data.particles["Particle Type"]
        )
        data.particles_["Particle Type_"][...] = 1

        return model

    def teardownFccBccHcpModel(self, data):
        data.particles_[
            "Particle Type_"][...] = data.particles["Particle Type Backup"]
        del data.particles_["Particle Type Backup"]

    def modify(self, data, frame, **kwargs):
        match self.structure:
            case "SiO2":
                model = self.setupSiO2model(data)
            case "FCC" | "BCC" | "HCP":
                model = self.setupFccBccHcpModel(data)
            case "None":
                return
            case _:
                raise NotImplementedError

        model = model.to(self.device)
        model.eval()

        noisy_atoms = ovito_to_ase(data)

        if self.scale is not None:
            modelScale = ScoreBasedDenoising.originalScale[self.structure] / self.scale
        else:
            estNNdist = self.estimateNearestNeighborsDistance(data)
            print(f"Estimated nearest neighbor distance = {estNNdist:#.3g} A")
            modelScale = ScoreBasedDenoising.originalScale[self.structure] / estNNdist

        denoised_atoms = yield from self.denoise_snapshot(
            noisy_atoms, model, modelScale
        )
        data.particles_["Position_"][...] = denoised_atoms

        match self.structure:
            case "SiO2":
                pass
            case "FCC" | "BCC" | "HCP":
                self.teardownFccBccHcpModel(data)
            case _:
                raise NotImplementedError
