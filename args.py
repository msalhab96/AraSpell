from argparse import ArgumentParser


def add_model_args(parser) -> None:
    group = parser.add_argument_group('Model Args')
    group.add_argument(
        '--d_model', default=512, type=int,
        help='the model dimensionality'
    )
    group.add_argument(
        '--h', default=8, type=int,
        help='The number of heads'
        )
    group.add_argument(
        '--n_layers', default=4, type=int,
        help='The number of encoder and decoder layers'
        )
    group.add_argument(
        '--p_dropout', default=0.1, type=float,
        help='The dropout ratio'
    )
    group.add_argument(
        '--hidden_size', default=256, type=int,
        help='The model hidden dim'
    )


def add_training_args(parser) -> None:
    group = parser.add_argument_group('Training Args')
    group.add_argument(
        '--epochs', default=100, type=int,
        help='The number of training epochs'
    )
    group.add_argument(
        '--batch_size', default=256, type=int,
        help='The training batch size'
    )
    group.add_argument(
        '--train_path', default='train.csv', type=str,
        help='The training data file path'
    )
    group.add_argument(
        '--clean_key', default='clean', type=str,
        help='The csv column name of the clean items'
    )
    group.add_argument(
        '--dist_key', default='distorted', type=str,
        help='The csv column name of the distorted items'
    )
    group.add_argument(
        '--test_path', default='test.csv', type=str,
        help='The testing data file path'
    )
    group.add_argument(
        '--max_len', default=128, type=int,
        help='The maximum length of each example'
    )
    group.add_argument(
        '--opt_betas', default=[0.9, 0.98], nargs='+',
        help='Adam Optimizer\'s beta0 and beta1'
    )
    group.add_argument(
        '--warmup_staps', default=4000, type=int,
        help='Adam Optimizer\'s warmup steps'
    )
    group.add_argument(
        '--opt_eps', default=1e-9, type=float,
        help='Adam Optimizer\'s warmup steps'
    )
    group.add_argument(
        '--dist_backend', default='nccl', type=str,
        help='The distributed training backend'
    )
    group.add_argument(
        '--dist_port', default=12345, type=int,
        help='The distributed training port'
    )
    group.add_argument(
        '--n_gpus', default=2, type=int,
        help='The The number of gpus to train on'
    )
    group.add_argument(
        '--pre_trained_path', default=None, type=str,
        help='The pretraining model path'
    )
    group.add_argument(
        '--outdir', default='outdir/', type=str,
        help='The path to save the checkpoints and the logs to'
    )
    group.add_argument(
        '--tokenizer_path', default='outdir/tokenizer.json', type=str,
        help='The path to save the checkpoints and the logs to'
    )
    group.add_argument(
        '--alpha', default=0.1, type=float,
        help='Labe smoothing value '
    )
    group.add_argument(
        '--stop_after', default=5, type=int,
        help='The number of epochs to stop after if no improvements happened'
    )
    group.add_argument(
        '--distortion_ratio', default=0.1, type=float,
        help='The data distortion ratio'
    )
    group.add_argument(
        '--logger_type', default='tensor_board', type=str,
        help='The logger type, either tensor_board or basic'
    )
    group.add_argument(
        '--logdir', default='outdir/logs', type=str,
        help='The directory to save the logs to'
    )


def get_preprocessing_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--sep', default=[
            '\n', '\t', '.', '،', ',', '=', ':', '-', '\\', '/'
            ], nargs='+', type=str,
        help='The seperator to be used to split the lines on'
        )
    parser.add_argument(
        '--min_len', default=15, type=int,
        help='The minimum line length to keep'
        )
    parser.add_argument(
        '--max_len', default=128, type=int,
        help='The maximum line length to keep'
        )
    parser.add_argument(
        '--dist_run', default=False, action='store_true'
    )
    parser.add_argument(
        '--data_path', default='data/', type=str
    )
    parser.add_argument(
        '--save_path', default='clean_data.txt', type=str
    )
    parser.add_argument(
        '--max_rep_chars', default=2, type=str
    )
    parser.add_argument(
        '--execlude_words_files', default='words.json', type=str
    )
    parser.add_argument(
        '--max_oov', default=1, type=int
    )
    parser.add_argument(
        '--min_words', default=3, type=int
    )
    parser.add_argument(
        '--max_words', default=20, type=int
    )
    parser.add_argument(
        '--dist_ratios', default=[0.05, 0.1, 0.15], nargs='+'
    )
    return parser.parse_args()


def get_train_args():
    parser = ArgumentParser()
    add_model_args(parser)
    add_training_args(parser)
    return parser.parse_args()


def get_model_args(args, voc_size: int, rank: int, pad_idx: int) -> dict:
    enc_params = {
        'n_layers': args.n_layers,
        'voc_size': voc_size,
        'hidden_size': args.hidden_size,
        'p_dropout': args.p_dropout,
        'pad_idx': pad_idx
    }
    params = {
        'd_model': args.d_model,
        'h': args.h,
        'device': f'cuda:{rank}',
        'voc_size': voc_size
    }
    dec_params = {
        'n_layers': args.n_layers,
        'p_dropout': args.p_dropout,
        'hidden_size': args.hidden_size,
        'voc_size': voc_size,
        'pad_idx': pad_idx
    }

    return {
        'enc_params': enc_params,
        'dec_params': dec_params,
        **params
    }
