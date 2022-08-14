from argparse import ArgumentParser
import os
import time
from processors import FilesProcessor, get_text_distorter
from processes import (
    CharsRemover,
    LengthFilter,
    LinesSplitter,
    LoadFile,
    NumbersFilter,
    OOVFilter,
    RepeatedCharsCollapsor,
    SoloCharFilter,
    SpacesRemover,
    ValidCharsKeeper,
    WordsFilter,
    WordsNumberFilter,
    CharsNormalizer
    )
from utils import load_json, save_text_file
from typing import Union, List
from pathlib import Path
import constants
import pandas as pd


def get_paths(
        main_dir: Union[Path, str]
        ) -> List[Union[Path, str]]:
    paths = [
        os.path.join(main_dir, file)
        for file in os.listdir(main_dir)
        ]
    return paths


def get_file_processor(args):
    words = load_json(args.execlude_words_files)
    processes = [
        LoadFile(),
        *[LinesSplitter(sep=sep) for sep in args.sep],
        CharsRemover(constants.ARABIC_HARAKAT),
        CharsNormalizer(constants.NORMLIZER_MAPPER),
        RepeatedCharsCollapsor(args.max_rep_chars),
        NumbersFilter(),
        SoloCharFilter(),
        WordsFilter(words),
        ValidCharsKeeper(constants.VALID_CHARS),
        SpacesRemover(),
        WordsNumberFilter(args.min_words, args.max_words),
        LengthFilter(args.min_len, args.max_len)
    ]
    return FilesProcessor(processes)


def post_process(data: List[str]) -> List[str]:
    lines = []
    for item in data:
        lines.extend(item)
    lines = set(lines)
    lines = OOVFilter(args.max_oov).execute(lines)
    return lines


def get_argparser():
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
        '--data_path', default='data/'
    )
    parser.add_argument(
        '--save_path', default='clean_data.txt'
    )
    parser.add_argument(
        '--max_rep_chars', default=2
    )
    parser.add_argument(
        '--execlude_words_files', default='words.json'
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
        '--dist_ratios', default=[0.05, 0.1, 0.15]
    )
    return parser


def main(args) -> None:
    fp = get_file_processor(args)
    files = get_paths(args.data_path)
    print('Started!')
    start = time.time()
    if args.dist_run is True:
        print('dist run')
        data = fp.dist_run(files)
    else:
        data = fp.run(files)
    end = time.time()
    print(f'Files Processing completed in {end - start}')
    data = post_process(data)
    df = None
    for i, ratio in enumerate(args.dist_ratios):
        distorter = get_text_distorter(ratio)
        dist = list(map(distorter.run, data))
        if df is None:
            df = pd.DataFrame({
                'clean': data,
                f'distorted_{ratio}': dist
            })
        else:
            df[f'distorted_{ratio}'] = dist
    df.to_csv(f'data.csv', encoding='utf-8')
    save_text_file(args.save_path, '\n'.join(data))


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
