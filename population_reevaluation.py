"""Independent, cached population fitness diagnostics (never selection feedback)."""
import copy
import hashlib
import json
import math
from pathlib import Path

POPULATION_REEVAL_SEED_OFFSET = 1_000_000


def parent_selection_probabilities(scores, *, population_type='topk', mutation_mode='random'):
    """Exact distribution of evolution_helpers.select_parent, without consuming RNG.

    The tournament samples two DISTINCT entries in random order; max chooses the
    first on a tie. Enumerating ordered pairs also matches None/NaN behavior.
    """
    n = len(scores)
    if not n:
        raise ValueError('Cannot select parents from an empty population')
    if n == 1 or (population_type == 'complexity' and mutation_mode == 'simplify'):
        return [1.0 / n] * n
    values = [s if s is not None else -1 for s in scores]
    wins = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j:
                winner = max((i, j), key=lambda k: values[k])
                wins[winner] += 1
    return [w / (n * (n - 1)) for w in wins]


class PopulationReevaluation:
    """Single-worker evaluator; submit immutable snapshots to observe in order.

    Cache identity includes the full evaluation config, excluding its display
    name. Persistence includes the evaluation context, seed allocation, per-member
    estimates and every generation snapshot. A returning member keeps its estimate.
    """
    def __init__(self, path, *, context, n_runs=3, resume_path=None):
        if n_runs <= 0:
            raise ValueError('Population reevaluation requires a positive seed count')
        self.path = Path(path)
        self.context = json.loads(json.dumps({'n_runs': n_runs, **context}, sort_keys=True))
        self.n_runs = n_runs
        self.state = {'context': self.context, 'next_run_index': POPULATION_REEVAL_SEED_OFFSET,
                      'estimates': {}, 'generations': []}
        source = self.path if self.path.exists() else Path(resume_path) if resume_path else None
        if source is not None and source.exists():
            state = json.loads(source.read_text())
            if state['context'] != self.context:
                raise ValueError('Population reevaluation context changed; cannot reuse cached estimates')
            self.state = state

    @staticmethod
    def snapshot(population, pysr_kwargs, *, generation, population_type, mutation_mode):
        members = []
        for bundle in population:
            config = copy.deepcopy(bundle.to_pysr_config(pysr_kwargs))
            identity = config.to_json_dict()
            identity.pop('name', None)
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            members.append({'key': key, 'name': bundle.display_name,
                            'train_score': bundle.score, 'config': config})
        probabilities = parent_selection_probabilities(
            [m['train_score'] for m in members], population_type=population_type,
            mutation_mode=mutation_mode)
        return {'generation': generation, 'population_type': population_type,
                'mutation_mode': mutation_mode, 'members': members,
                'parent_probabilities': probabilities}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.state, indent=2) + '\n')
        temporary.replace(self.path)

    def observe(self, snapshot, evaluate):
        """evaluate(configs, starts, n_runs) returns one (score, vector, details) per config."""
        cache = self.state['estimates']
        new = {}
        for member in snapshot['members']:
            if member['key'] not in cache:
                new.setdefault(member['key'], member)
        starts = []
        for _ in new:
            starts.append(self.state['next_run_index'])
            self.state['next_run_index'] += self.n_runs
        # Reserve indices durably even if evaluation fails or the process stops.
        self._save()
        if new:
            results = evaluate([m['config'] for m in new.values()], starts, self.n_runs)
            if len(results) != len(new):
                raise ValueError('Incomplete population reevaluation batch')
            if any(not math.isfinite(float(result[0])) for result in results):
                raise ValueError('Non-finite population reevaluation score')
            for (key, member), start, (score, vector, details) in zip(new.items(), starts, results):
                cache[key] = {'name': member['name'], 'score': float(score),
                              'score_vector': vector, 'result_details': details,
                              'first_generation': snapshot['generation'],
                              'run_index_start': start, 'n_runs': self.n_runs}
            self._save()
        members = [{k: v for k, v in m.items() if k != 'config'} for m in snapshot['members']]
        scores = [cache[m['key']]['score'] for m in members]
        probs = snapshot['parent_probabilities']
        for m, score, prob in zip(members, scores, probs):
            m.update(reeval_score=score, parent_probability=prob)
        result = {k: v for k, v in snapshot.items() if k not in ('members', 'parent_probabilities')}
        result.update(members=members, avg_score=sum(scores)/len(scores),
                      expected_parent_score=sum(p*s for p, s in zip(probs, scores)),
                      new_members=len(new), new_seed_runs=len(new)*self.n_runs)
        # A resumed run can resnapshot the last saved population without duplicating its generation.
        self.state['generations'] = [r for r in self.state['generations']
                                     if r['generation'] != result['generation']]
        self.state['generations'].append(result)
        self._save()
        return result
