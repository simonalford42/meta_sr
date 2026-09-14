"""Regression coverage for overlapping noise-level checkpoint fits."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event

import pandas as pd
import pytest

from run_pysr_srbench import run_pysr_with_hof_checkpoints


def test_overlapping_fits_cannot_delete_each_others_outputs(tmp_path):
    barrier = Barrier(2)
    first_done = Event()
    paths = []
    legacy = tmp_path / 'pysr_tmp_shared'
    legacy_target = tmp_path / 'cached_output'
    legacy_target.mkdir()
    legacy.symlink_to(legacy_target, target_is_directory=True)

    class Model:
        equations_ = pd.DataFrame([{'equation': 'x0', 'loss': 0.0}])

        def __init__(self, index):
            self.index = index

        def fit(self, *args, **kwargs):
            directory = Path(self.output_directory)
            directory.mkdir(exist_ok=True)
            marker = directory / f'worker{self.index}.csv'
            marker.write_text('frontier')
            paths.append(directory)
            barrier.wait(timeout=10)
            if self.index == 1:
                assert first_done.wait(timeout=10)
                assert marker.read_text() == 'frontier'

    def run(index):
        try:
            return run_pysr_with_hof_checkpoints(
                [[0]], [0], ['x0'], 'shared', str(tmp_path), [1], Model(index),
                hof_path=str(tmp_path / f'noise{index}.csv'),
            )
        finally:
            if index == 0:
                first_done.set()

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(run, index) for index in range(2)]
        for future in futures:
            future.result(timeout=20)
    assert len(set(paths)) == 2
    assert all(not path.exists() for path in paths)
    assert legacy.is_symlink() and legacy_target.exists()
    assert all((tmp_path / f'noise{i}.csv').exists() for i in range(2))


def test_fit_exception_preserved_and_owned_scratch_cleaned(tmp_path):
    class Model:
        def fit(self, *args, **kwargs):
            self.scratch = Path(self.output_directory)
            (self.scratch / 'partial.csv').write_text('partial')
            raise RuntimeError('original fit failure')

    model = Model()
    with pytest.raises(RuntimeError, match='original fit failure'):
        run_pysr_with_hof_checkpoints(
            [[0]], [0], ['x0'], 'shared', str(tmp_path), [1], model,
        )
    assert not model.scratch.exists()
