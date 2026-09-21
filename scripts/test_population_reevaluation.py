import copy
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from population_reevaluation import (
    POPULATION_REEVAL_SEED_OFFSET,
    PopulationReevaluation,
    parent_selection_probabilities,
)


@dataclass
class Config:
    name: str
    code: str
    kwargs: dict

    def to_json_dict(self):
        return vars(self).copy()


@dataclass
class Bundle:
    display_name: str
    score: float
    code: str

    def to_pysr_config(self, kwargs):
        return Config(self.display_name, self.code, kwargs)


def snapshot(population, generation=0):
    return PopulationReevaluation.snapshot(
        population, {'timeout_in_seconds': 90}, generation=generation,
        population_type='topk', mutation_mode='random')


def test_exact_tournament_probabilities_match_actual_selector():
    from evolution_helpers import select_parent

    # Includes ties, singleton, and None's actual -1 sentinel semantics.
    for scores in [[1], [1, 2], [1, 1, 2, 3], [None, -2, 0], [float('nan'), 1, 2]]:
        pop = [SimpleNamespace(score=s) for s in scores]
        orders = list(itertools.permutations(range(len(pop)), min(2, len(pop))))
        counts = [0]*len(pop)
        for order in orders:
            rng = SimpleNamespace(sample=lambda population, n: [population[i] for i in order])
            chosen = select_parent(pop, rng)
            counts[next(i for i, b in enumerate(pop) if b is chosen)] += 1
        assert parent_selection_probabilities(scores) == pytest.approx([n/len(orders) for n in counts])
    assert parent_selection_probabilities([1, 2, 3]) == pytest.approx([0, 1/3, 2/3])
    assert parent_selection_probabilities([1, 2, 3], population_type='complexity', mutation_mode='simplify') == pytest.approx([1/3]*3)
    assert parent_selection_probabilities([1, 2, 3], population_type='task') == pytest.approx([0, 1/3, 2/3])


def test_two_new_members_only_cost_six_seed_runs_and_returning_members_reuse(tmp_path):
    diagnostic = PopulationReevaluation(tmp_path/'population_reeval.json', context={'seed': 1})
    population = [Bundle(str(i), float(i), str(i)) for i in range(10)]
    before = copy.deepcopy(population)
    calls = []

    def evaluate(configs, starts, n_runs):
        calls.append((len(configs), starts, n_runs))
        return [(float(c.code)/20, [], []) for c in configs]

    result = diagnostic.observe(snapshot(population), evaluate)
    assert result['new_seed_runs'] == 30
    assert result['avg_score'] == pytest.approx(.225)
    assert result['expected_parent_score'] > result['avg_score']
    assert population == before  # Diagnostic scores never enter selection.
    changed = population[2:]+[Bundle('10', 10., '10'), Bundle('11', 11., '11')]
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(diagnostic.observe, snapshot(changed, 1), evaluate)
        second = executor.submit(diagnostic.observe, snapshot(changed, 2), evaluate)
        assert first.result()['new_seed_runs'] == 6
        assert second.result()['new_seed_runs'] == 0
    assert [c[0] for c in calls] == [10, 2]
    assert calls[0][1][0] == POPULATION_REEVAL_SEED_OFFSET
    assert calls[1][1] == [POPULATION_REEVAL_SEED_OFFSET+30, POPULATION_REEVAL_SEED_OFFSET+33]
    assert diagnostic.observe(snapshot(population, 3), evaluate)['new_seed_runs'] == 0
    resumed = PopulationReevaluation(tmp_path/'resumed.json', context={'seed': 1},
                                    resume_path=tmp_path/'population_reeval.json')
    assert resumed.observe(snapshot(changed, 4), evaluate)['new_seed_runs'] == 0
    assert len(resumed.state['generations']) == 5
    with pytest.raises(ValueError, match='context changed'):
        PopulationReevaluation(tmp_path/'population_reeval.json', context={'seed': 2})


def test_snapshot_does_not_follow_later_scores_or_code_changes(tmp_path):
    pop = [Bundle('a', 1., 'a'), Bundle('b', 2., 'b')]
    snap = snapshot(pop)
    pop[0].score = 99
    pop[0].code = 'changed'
    diagnostic = PopulationReevaluation(tmp_path/'diagnostic.json', context={})
    result = diagnostic.observe(snap, lambda configs, starts, n: [(.9, [], []), (.2, [], [])])
    assert result['avg_score'] == pytest.approx(.55)
    assert result['expected_parent_score'] == pytest.approx(.2)
    assert result['members'][0]['train_score'] == 1
    # Renamed, same-code member reuses the estimate; changed code is a newcomer.
    renamed = [Bundle('renamed', 100, 'b')]
    assert diagnostic.observe(snapshot(renamed, 1), lambda *args: pytest.fail('must reuse'))['avg_score'] == .2
    assert snapshot(pop)['members'][0]['key'] != snap['members'][0]['key']


def test_failed_batch_never_publishes_partial_average_and_reserves_seeds(tmp_path):
    path = tmp_path/'diagnostic.json'
    diagnostic = PopulationReevaluation(path, context={})
    pop = [Bundle('a', 1, 'a'), Bundle('b', 2, 'b')]
    with pytest.raises(ValueError, match='Incomplete'):
        diagnostic.observe(snapshot(pop), lambda *args: [(.5, [], [])])
    saved = json.loads(path.read_text())
    assert saved['estimates'] == {} and saved['generations'] == []
    assert saved['next_run_index'] == POPULATION_REEVAL_SEED_OFFSET+6
    with pytest.raises(ValueError, match='Non-finite'):
        diagnostic.observe(snapshot(pop), lambda *args: [(float('nan'), [], []), (.5, [], [])])
    assert diagnostic.state['estimates'] == {}


def test_delayed_wandb_rows_do_not_overwrite_each_other_or_drop_later_metrics(monkeypatch):
    import sys
    from evolve_pysr import _wandb_log_at_eval_step
    run = SimpleNamespace(step=50)
    pending = {}
    committed = []

    def log(data, step, commit=None):
        assert step >= run.step
        pending.update(data)
        run.step = step
        if commit:
            committed.append(dict(pending))
            pending.clear()
            run.step += 1

    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(run=run, log=log))
    _wandb_log_at_eval_step({'best_score': .8}, step=50)
    for gen in [18, 19, 20]:
        _wandb_log_at_eval_step({'population_reeval/generation': gen,
                                'population_reeval/avg_score': gen/100}, step=50, commit=True)
    _wandb_log_at_eval_step({'best_score': .9}, step=51, commit=True)
    assert [r['population_reeval/generation'] for r in committed[:3]] == [18, 19, 20]
    assert [r['eval_idx'] for r in committed] == [50, 50, 50, 51]
    assert committed[-1]['best_score'] == .9


def test_local_plot_reads_persisted_generations(tmp_path, monkeypatch):
    import sys
    from figures.plot_population_reevaluation import main
    diagnostic = PopulationReevaluation(tmp_path/'run'/'population_reeval.json', context={})
    pop = [Bundle('a', 1, 'a'), Bundle('b', 2, 'b')]
    diagnostic.observe(snapshot(pop), lambda *args: [(.4, [], []), (.8, [], [])])
    output = tmp_path/'figures'
    monkeypatch.setattr(sys, 'argv', ['plot_population_reevaluation.py', str(tmp_path/'run'),
                                     '--output-dir', str(output)])
    main()
    assert (output/'population_reevaluation.png').stat().st_size > 1000
    assert (output/'population_reevaluation.pdf').stat().st_size > 1000
