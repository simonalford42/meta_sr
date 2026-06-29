import subprocess
import wandb
import pysr
from pysr import jl
import random

from matplotlib import pyplot as plt
import os
import spock_reg_model
import numpy as np
import argparse
from einops import rearrange
import utils
import pickle
from utils import assert_equal
import einops
import torch


LL_LOSS = """
function safe_log_erf(x)
    if x < -1
        (
            0.485660082730562*x + 0.643278438654541*exp(x)
            + 0.00200084619923262*x^3 - 0.643250926022749
            - 0.955350621183745*x^2
        )
    else
        log(1 + erf(x))
    end
end

function elementwise_loss(prediction, target)


    mu = prediction

    if mu < 1 || mu > 14
        # The farther away from a reasonable range, the more we punish it
        return 100 * (mu - 7)^2
    end

    sigma = one(prediction)

    # equation 8 in Bayesian neural network paper
    log_like = if target >= 9
        safe_log_erf((mu - 9) / sqrt(2 * sigma^2))
    else
        (
            zero(mu)
            - (target - mu)^2 / (2 * sigma^2)
            - log(sigma)
            - safe_log_erf((mu - 4) / sqrt(2 * sigma^2))
        )
    end

    return -log_like
end
"""

LL_LOSS2 = """
function safe_log_erf(x)
    if x < -1
        (
            0.485660082730562*x + 0.643278438654541*exp(x)
            + 0.00200084619923262*x^3 - 0.643250926022749
            - 0.955350621183745*x^2
        )
    else
        log(1 + erf(x))
    end
end

function elementwise_loss(prediction, target)
    mu = prediction

    if mu < 1 || mu > 14
        # The farther away from a reasonable range, the more we punish it
        return 100 * (mu - 7)^2
    end

    sigma = one(prediction)

    # equation 8 in Bayesian neural network paper
    log_like = if target >= 9
        (
            safe_log_erf((mu - 9) / sqrt(2 * sigma^2))
            - safe_log_erf((mu - 4) / sqrt(2 * sigma^2))
        )
    else
        (
            zero(mu)
            - (target - mu)^2 / (2 * sigma^2)
            - log(sigma)
            - safe_log_erf((mu - 4) / sqrt(2 * sigma^2))
        )
    end

    return -log_like
end
"""

NN_STD_LOSS = """
function safe_log_erf(x)
    if x < -1
        (
            0.485660082730562*x + 0.643278438654541*exp(x)
            + 0.00200084619923262*x^3 - 0.643250926022749
            - 0.955350621183745*x^2
        )
    else
        log(1 + erf(x))
    end
end

function elementwise_loss(mu, sigma, target)
    if mu < 1 || mu > 14
        # The farther away from a reasonable range, the more we punish it
        return 100 * (mu - 7)^2
    end

    # equation 8 in Bayesian neural network paper
    log_like = if target >= 9
        (
            safe_log_erf((mu - 9) / sqrt(2 * sigma^2))
            - safe_log_erf((mu - 4) / sqrt(2 * sigma^2))
        )
    else
        (
            zero(mu)
            - (target - mu)^2 / (2 * sigma^2)
            - log(sigma)
            - safe_log_erf((mu - 4) / sqrt(2 * sigma^2))
        )
    end

    return -log_like
end

function my_loss(tree, dataset::Dataset{T,L}, options, idx)::L where {T,L}
    inds = idx === nothing ? Colon() : idx
    X = view(dataset.X, :, inds)
    y = view(dataset.y, inds)
    sigma = @view X[end, :]
    mu, ok = eval_tree_array(tree, X, options)
    if !ok
        return L(100_000)
    end

    loss = sum(
        i -> elementwise_loss(mu[i], sigma[i], y[i]),
        eachindex(mu)
    ) / length(mu)

    if loss < 0
        error("Loss is negative!")
    end

    return loss
end
"""

CLIPPED_LOSS = """
function elementwise_loss(prediction, target)
    return (target - min(prediction, 9))^2
end
"""

### predicting < 0 is a -1
### predicting >= 0 is a +1

### imbalanced/conservative loss
MODIFIED_PERCEPTRON_LOSS ='''
function elementwise_loss(x, y)
    if y == 1
        return x < 0 ? 1.0 : 0.0
    elseif y == -1
        return x < 0 ? 0.0 : -y * x + 1
    else
        error("y must be 1 or -1")
    end
end
'''

MODIFIED_ZEROONE_LOSS = '''
function elementwise_loss(x, y)
    if x == 0
        return y == -1 ? 1.0 : 0.0
    else
        return y * x < 0 ? 1.0 : 0.0
    end
end
'''


LABELS = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']

LABEL_TO_IX = {label: i for i, label in enumerate(LABELS)}

def get_sr_included_ixs():
    ''' for running pysr to imitate f1, we only want to use the inputs that f1 uses '''
    # hard coded based off the default CL args passed
    skipped = ['nan_mmr_near', 'nan_mmr_far', 'nan_megno', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno']
    assert len(skipped) == 10

    included_ixs = [i for i in range(len(LABELS)) if LABELS[i] not in skipped]
    return included_ixs


INCLUDED_IXS = get_sr_included_ixs()
INPUT_VARIABLE_NAMES = [LABELS[ix] for ix in INCLUDED_IXS]


def load_inputs_and_targets(config):
    model = spock_reg_model.load(version=config['nn_version'])
    model.make_dataloaders()
    model.eval()

    data_iterator = iter(model.train_dataloader())
    x, y = next(data_iterator)
    while x.shape[0] < config['n']:
        next_x, next_y = next(data_iterator)
        x = torch.cat([x, next_x], dim=0)
        y = torch.cat([y, next_y], dim=0)

    out_dict = model.forward(x, return_intermediates=True, noisy_val=False)
    # save out_dict to pickle called f{version}_out_dict.pkl
    # with open(f'f{config["version"]}_out_dict.pkl', 'wb') as f:
        # pickle.dump(out_dict, f)
    # assert 0

    if config['target'] == 'f1':
        # inputs to SR are the inputs to f1 neural network
        # we use this instead of x because the model zeros the unused inputs,
        #  which is a nice check to have
        X = out_dict['inputs']  # [B, T, F]
        # targets for SR are the outputs of the f1 neural network
        y = out_dict['f1_output']  # [B, T, F]
        # f1 acts on timesteps independently, so we can just use different
        #  time steps as different possible samples
        X = rearrange(X, 'B T F -> (B T) F')
        y = rearrange(y, 'B T F -> (B T) F')

        # extract just the input variables we're actually using
        assert_equal(len(LABELS), X.shape[1])
        X = X[..., INCLUDED_IXS]

        in_dim = len(INCLUDED_IXS)
        out_dim = model.hparams['latent']

        variable_names = INPUT_VARIABLE_NAMES
    elif config['target'] == 'f2':
        # inputs to SR are the inputs to f2 neural network
        X = out_dict['summary_stats']  # [B, 40]
        # outputs are the (mean, std) predictions of the nn
        y = out_dict['prediction']  # [B, 2]
        y = y[:, 0:1]  # stop predicting std too haha


        in_dim = model.summary_dim
        out_dim = 1

        n = X.shape[1] // 2
        variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]

        if config['loss_fn'] == 'nn_std':
            std = out_dict['std']
            B, n = X.shape
            assert_equal(std.shape, (B, 1))
            # go from X [B, 40] and std [B, 1] to X [B, 41] by concatenating std to X
            X = torch.cat([X, std], dim=1)
            assert_equal(X.shape, (B, n+1))
            variable_names.append('nn_std')
            in_dim += 1

        if config['sr_residual']:
            raise ValueError("Implementation is currently broken")
            # 1. load the previous pysr results, using highest complexity
            # 2. calculate mean and std from previous results (previous run must have used target=='f2')
            # 3. concatenate mean and std to produce previous "prediction" of shape [B, 2]
            # 4. subtract previous prediction from y to get residual target

            # 1
            with open(config['previous_sr_path'], 'rb') as f:
                previous_sr_model = pickle.load(f)

            # 2
            summary_stats_np = out_dict['summary_stats'].detach().numpy()

            # get the highest complexity equation
            max_complexity_idx = previous_sr_model.equations['complexity'].max()
            mean_equation = previous_sr_model.equations_[0].iloc[max_complexity_idx]
            std_equation = previous_sr_model.equations_[1].iloc[max_complexity_idx]

            # mean and std
            results = []
            for equation in mean_equation, std_equation:
                lambda_func = equation['lambda_format']
                evaluated_result = lambda_func(summary_stats_np)
                # Ensure the result is reshaped to match the batch size
                evaluated_result = evaluated_result.reshape(-1, 1)
                results.append(evaluated_result)

            # 3
            previous_prediction = np.hstack(results)
            assert_equal(previous_prediction.shape, (X.shape[0], 2))  # [B, 2]

            # 4
            y = y - previous_prediction

    elif config['target'] == 'f2_ifthen':
        # inputs to SR are the inputs to f2 neural network
        X = out_dict['summary_stats']  # [B, 40]
        # target for SR is the predicates from the ifthen network
        y = out_dict['ifthen_preds']  # [B, 10]

        in_dim = model.summary_dim
        out_dim = 10

        n = X.shape[1] // 2
        variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]

    elif config['target'] in ['f2_direct', 'equation_bounds']:
        # inputs to SR are the inputs to f2 neural network
        X = out_dict['summary_stats']  # [B, 40]
        # target for SR is the ground truth mean, which we already have
        y = y  # [B, 2]

        n = X.shape[1] // 2
        in_dim = model.summary_dim
        out_dim = 1

        if config['loss_fn'] == 'nn_std':
            std = out_dict['std']
            B, n = X.shape
            assert_equal(std.shape, (B, 1))
            # go from X [B, 40] and std [B, 1] to X [B, 41] by concatenating std to X
            X = torch.cat([X, std], dim=1)
            assert_equal(X.shape, (B, n+1))
            in_dim += 1

        # there are two ground truth predictions. create a data point for each
        X = einops.repeat(X, 'B F -> (B two) F', two=2)
        y = einops.rearrange(y, 'B two -> (B two) 1')

        variable_names = [f'm{i}' for i in range(n)] + [f's{i}' for i in range(n)]
        if config['loss_fn'] == 'nn_std':
            variable_names.append('nn_std')

        if config['sr_residual']:
            raise ValueError("implementation is currently broken")

            # Load the PySR equations from the previous round
            with open(config['previous_sr_path'], 'rb') as f:
                previous_sr_model = pickle.load(f)

            # Evaluate the previous PySR equations on the inputs
            additional_features = []
            summary_stats_np = out_dict['summary_stats'].detach().numpy()
            for equation_set in previous_sr_model.equations_:
                for index, equation in equation_set.iterrows():
                    lambda_func = equation['lambda_format']
                    evaluated_result = lambda_func(summary_stats_np)
                    # Ensure the result is reshaped to match the batch size
                    evaluated_result = evaluated_result.reshape(-1, 1)
                    additional_features.append(evaluated_result)

            # Convert list of arrays to a single numpy array
            additional_features = np.hstack(additional_features)
            # Concatenate the original summary stats with the evaluated results
            # summary_stats_np: [1, 2000, 40]
            # additional_features: (2000, 49)
            # (args.n, in_dim) = (250, 2*20+49)
            X = np.concatenate([summary_stats_np, additional_features], axis=1)

            in_dim = model.summary_dim + additional_features.shape[1]

        if config['target'] == 'equation_bounds':
            # for predicting whether or not an equation applies, learn equations
            # that predict when error is low

            with open(config['previous_sr_path'], 'rb') as f:
                model = pickle.load(f)

            # get the highest complexity equation
            max_complexity_idx = model.equations_[0]['complexity'].argmax()
            mean_equation = model.equations_[0].iloc[max_complexity_idx]

            # get the lambda eq for it, and evaluate on X
            X, y = X.detach().numpy(), y.detach().numpy()
            y_pred = mean_equation['lambda_format'](X)

            # calculate the error
            assert_equal(y_pred.shape, (X.shape[0], ))
            assert_equal(y.shape, (X.shape[0], 1))
            mse = (y - y_pred[:, None])**2
            assert_equal(mse.shape, y.shape)

            # predict with binary classification whether equation applies or not
            # to use ZeroOneLoss, need +/- 1 targets
            # https://astroautomata.com/SymbolicRegression.jl/dev/losses/
            y = np.zeros_like(mse)
            y[mse > config['eq_bound_mse_threshold']] = -1
            y[mse <= config['eq_bound_mse_threshold']] = 1
            print(f'Equation bound data has {(y == -1).sum()} oob targets and {(y == 1).sum()} good-predicting targets')


    else:
        raise ValueError(f"Unknown target: {config['target']}")

    if config['residual']:
        assert config['target'] == 'f2_direct', 'residual requires a direct target'
        # target is the residual error of the model's prediction from the ground truth
        # because target was f2 direct, y shape is [B * 2, 1]
        # but predicted mean is [B, 1]
        # so repeat the predicted means
        y_old = out_dict['mean']
        assert_equal(y_old.shape[1], 1)
        y_old = einops.repeat(y_old, 'B one -> (B repeat) one', repeat=2)
        assert_equal(y_old.shape, y.shape)
        y = y - y_old

    # go down from having a batch of size B to just N
    ixs = np.random.choice(X.shape[0], size=config['n'], replace=False)
    X, y = X[ixs], y[ixs]

    # Ensure X and y are NumPy arrays
    if isinstance(X, torch.Tensor):
        X, y = X.detach().numpy(), y.detach().numpy()

    assert_equal(X.shape, (config['n'], in_dim))
    assert_equal(y.shape, (config['n'], out_dim))

    return X, y, variable_names


def get_config(args):
    if args.version is None:
        version = random.randint(0, 100000)
        while os.path.exists(f'sr_results/{version}.pkl'):
            version = random.randint(0, 100000)
    else:
        version = args.version

    path = f'sr_results/{version}.csv'
    # create the directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # https://stackoverflow.com/a/57474787/4383594
    try:
        num_cpus = int(os.environ.get('SLURM_CPUS_ON_NODE')) * int(os.environ.get('SLURM_JOB_NUM_NODES'))
    except TypeError:
        num_cpus = 10

    pysr_config = dict(
        procs=num_cpus,
        populations=3*num_cpus,
        batching=True,
        batch_size=args.batch_size,
        equation_file=path,
        niterations=args.niterations,
        binary_operators=["+", "*", '/', '-', '^'],
        # unary_operators=['sin'],
        maxsize=args.max_size,
        timeout_in_seconds=int(60*60*args.time_in_hours),
        # prevent ^ from using complex exponents, nesting power laws is expressive but uninterpretable
        # base can have any complexity, exponent can have max 1 complexity
        constraints={'^': (-1, 1)},
        # nested_constraints={"sin": {"sin": 0}},
        ncyclesperiteration=1000, # increase utilization since usually using 32-ish cores?
        random_state=args.seed,
    )

    if args.loss_fn == 'll':
        pysr_config['elementwise_loss'] = LL_LOSS
    elif args.loss_fn == 'll2':
        pysr_config['elementwise_loss'] = LL_LOSS2
    elif args.loss_fn == 'clipped':
        pysr_config['elementwise_loss'] = CLIPPED_LOSS
    elif args.loss_fn == 'nn_std':
        jl.seval(NN_STD_LOSS)
        pysr_config['loss_function'] = 'my_loss'
        pysr_config['elementwise_loss'] = None

    if args.target == 'equation_bounds':
        pysr_config['elementwise_loss'] = MODIFIED_ZEROONE_LOSS
        if args.loss_fn == 'perceptron':
            pysr_config['elementwise_loss'] = MODIFIED_PERCEPTRON_LOSS

    config = vars(args)
    config.update(pysr_config)
    config['pysr_config'] = pysr_config
    config.update({
        'version': version,
        'slurm_id': os.environ.get('SLURM_JOB_ID', None),
        'slurm_name': os.environ.get('SLURM_JOB_NAME', None),
    })

    return config


def run_pysr(config):
    command = utils.get_script_execution_command()
    print(command)

    X, y, variable_names = load_inputs_and_targets(config)

    model = pysr.PySRRegressor(**config['pysr_config'])

    if not config['no_log']:
        wandb.init(
            entity='bnn-chaos-model',
            project='planets-sr',
            config=config,
        )

    model.fit(X, y, variable_names=variable_names)
    print('Done running pysr')

    losses = [min(eqs['loss']) for eqs in model.equation_file_contents_]

    if not config['no_log']:
        wandb.log({'avg_loss': sum(losses)/len(losses),
                   'losses': losses,
                   })

    print(f"Saved to path: {config['equation_file']}")
    print(f'Finished running pysr with pysr_version {config["version"]}')


def parse_args():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')
    # when importing from jupyter nb, it passes an arg to --f which we should just ignore
    parser.add_argument('--no_log', action='store_true', default=False, help='disable wandb logging')
    parser.add_argument('--nn_version', type=int, help='', required=True)
    parser.add_argument('--version', type=int, default=None)
    parser.add_argument('--time_in_hours', type=float, default=8)
    parser.add_argument('--niterations', type=float, default=500000) # by default, use time in hours as limit
    parser.add_argument('--max_size', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--target', type=str, default='f2', choices=['f1', 'f2', 'f2_ifthen', 'f2_direct', 'equation_bounds'])
    parser.add_argument('--residual', action='store_true', help='do residual training of your target')
    parser.add_argument('--n', type=int, default=10000, help='number of data points for the SR problem')
    parser.add_argument('--batch_size', type=int, default=1000, help='number of data points for the SR problem')
    parser.add_argument('--sr_residual', action='store_true', help='do residual training of your target with previous sr run as base')
    parser.add_argument('--loss_fn', type=str, choices=['mse', 'll', 'perceptron', 'clipped', 'll2', 'nn_std'], default='mse')
    parser.add_argument('--previous_sr_path', type=str, default='sr_results/92985.pkl', help='path to previous sr run, used for residual/recursive training')
    parser.add_argument('--eq_bound_mse_threshold', type=float, default=1, help='mse threshold below which to consider an equation good')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = get_config(args)
    run_pysr(config)
