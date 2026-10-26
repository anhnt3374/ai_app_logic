"""Baseline v1: replace create() to research a different business flow."""
from logic_simulator.face_pipeline import FacePipeline


def create(config, gateway, run_path):
    return FacePipeline(config, gateway, run_path)
