from argparse import ArgumentParser
import os
import time
from process import FilesProcessor
from processes import (
    CharsRemover,
    LengthFilter,
    LinesSplitter,
    LoadFile,
    RepeatedCharsCollapsor,
    SpacesRemover,
    ValidCharsKeeper
    )
from utils import save_text_file
from typing import Union, List
from pathlib import Path
import constants


def get_paths(
        main_dir: Union[Path, str]
        ) -> List[Union[Path, str]]:
    paths = [
        os.path.join(main_dir, file)
        for file in os.listdir(main_dir)
        ]
    return paths


def get_file_processor(args):
    processes = [
        LoadFile(),
        *[LinesSplitter(sep=sep) for sep in args.sep],
        CharsRemover(constants.ARABIC_HARAKAT),
        RepeatedCharsCollapsor(args.max_rep_chars),
        ValidCharsKeeper(constants.VALID_CHARS),
        SpacesRemover(),
        LengthFilter(args.min_len, args.max_len)
    ]
    return FilesProcessor(processes)


def post_process(data: List[str]) -> List[str]:
    lines = []
    for item in data:
        lines.extend(item)
    lines = set(lines)
    return lines


def get_argparser():
    parser = ArgumentParser()
    parser.add_argument(
        '--sep', default=['. ', ':', ' .', ', ', '، '], nargs='+', type=str,
        help='The seperator to be used to split the lines on'
        )
    parser.add_argument(
        '--min_len', default=15, type=int,
        help='The minimum line length to keep'
        )
    parser.add_argument(
        '--max_len', default=768, type=int,
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
    save_text_file(args.save_path, '\n'.join(data))


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
