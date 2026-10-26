import importlib.util
from pathlib import Path


def load_plugin(config,gateway,run_path,snapshot=None):
    root=Path(snapshot) if snapshot else Path(config.logic_root)/config.logic_id
    path=root/'logic.py'
    if not path.is_file() or not (root/'README.md').is_file():
        raise ValueError(f'Logic requires logic.py and README.md: {root}')
    spec=importlib.util.spec_from_file_location('simulator_logic_plugin',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create(config,gateway,run_path)
