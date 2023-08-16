import pytest
from ovito.io import import_file
from ovito.modifiers import CommonNeighborAnalysisModifier, ExpressionSelectionModifier
from ovito.pipeline import Pipeline
from scoreBasedDenoising import ScoreBasedDenoising


@pytest.fixture
def import_pipeline():
    pipe = import_file("examples/fcc_gb_example.data.gz")
    yield pipe


def test_default_fcc_settings(import_pipeline: Pipeline):
    pipe = import_pipeline
    pipe.modifiers.append(ScoreBasedDenoising(structure="FCC"))
    pipe.modifiers.append(CommonNeighborAnalysisModifier())
    data = pipe.compute()
    expected = {
        "CommonNeighborAnalysis.counts.BCC": 16,
        "CommonNeighborAnalysis.counts.FCC": 7893,
        "CommonNeighborAnalysis.counts.HCP": 205,
        "CommonNeighborAnalysis.counts.ICO": 0,
        "CommonNeighborAnalysis.counts.OTHER": 334,
    }

    assert len(data.tables["Convergence"].xy()) == 8
    for k, v in expected.items():
        assert data.attributes[k] == v


def test_num_steps_settings(import_pipeline: Pipeline):
    pipe = import_pipeline
    pipe.modifiers.append(ScoreBasedDenoising(steps=2, structure="FCC"))
    pipe.modifiers.append(CommonNeighborAnalysisModifier())
    data = pipe.compute()
    assert len(data.tables["Convergence"].xy()) == 2


def test_default_none_settings(import_pipeline: Pipeline):
    pipe = import_pipeline
    pipe.modifiers.append(ScoreBasedDenoising())
    pipe.modifiers.append(CommonNeighborAnalysisModifier())
    data = pipe.compute()
    expected = {
        "CommonNeighborAnalysis.counts.BCC": 40,
        "CommonNeighborAnalysis.counts.FCC": 3240,
        "CommonNeighborAnalysis.counts.HCP": 93,
        "CommonNeighborAnalysis.counts.ICO": 0,
        "CommonNeighborAnalysis.counts.OTHER": 5075,
    }

    for k, v in expected.items():
        assert data.attributes[k] == v


def test_none_settings(import_pipeline: Pipeline):
    pipe = import_pipeline
    pipe.modifiers.append(ScoreBasedDenoising(structure="None"))
    pipe.modifiers.append(CommonNeighborAnalysisModifier())
    data = pipe.compute()
    expected = {
        "CommonNeighborAnalysis.counts.BCC": 40,
        "CommonNeighborAnalysis.counts.FCC": 3240,
        "CommonNeighborAnalysis.counts.HCP": 93,
        "CommonNeighborAnalysis.counts.ICO": 0,
        "CommonNeighborAnalysis.counts.OTHER": 5075,
    }

    for k, v in expected.items():
        assert data.attributes[k] == v


def test_selection_fcc_settings(import_pipeline: Pipeline):
    pipe = import_pipeline
    pipe.modifiers.append(
        ExpressionSelectionModifier(
            expression="ReducedPosition.Z > 0.4 &&  ReducedPosition.Z < 0.6"
        )
    )
    pipe.modifiers.append(ScoreBasedDenoising(structure="FCC", only_selected=True))
    pipe.modifiers.append(CommonNeighborAnalysisModifier())
    data = pipe.compute()
    expected = {
        "CommonNeighborAnalysis.counts.BCC": 40,
        "CommonNeighborAnalysis.counts.FCC": 4015,
        "CommonNeighborAnalysis.counts.HCP": 175,
        "CommonNeighborAnalysis.counts.ICO": 0,
        "CommonNeighborAnalysis.counts.OTHER": 4218,
    }

    for k, v in expected.items():
        assert data.attributes[k] == v


def test_fcc_distance_settings(import_pipeline: Pipeline):
    pipe = import_pipeline
    pipe.modifiers.append(ScoreBasedDenoising(scale=2.5, structure="FCC"))
    pipe.modifiers.append(CommonNeighborAnalysisModifier())
    data = pipe.compute()
    expected = {
        "CommonNeighborAnalysis.counts.BCC": 11,
        "CommonNeighborAnalysis.counts.FCC": 7898,
        "CommonNeighborAnalysis.counts.HCP": 214,
        "CommonNeighborAnalysis.counts.ICO": 0,
        "CommonNeighborAnalysis.counts.OTHER": 325,
    }

    assert len(data.tables["Convergence"].xy()) == 8
    for k, v in expected.items():
        assert data.attributes[k] == v
