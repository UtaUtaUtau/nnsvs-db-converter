from __future__ import annotations
import logging
from pathlib import Path
logging.basicConfig(
    format='%(asctime)s - %(levelname)s: %(message)s',
    level=logging.INFO, datefmt='%x %a %X'
)
from utils.configs import load_config
from argparse import ArgumentParser, MetavarTypeHelpFormatter, ArgumentDefaultsHelpFormatter, Namespace

# combined formatter for arguments
class CombinedFormatter(MetavarTypeHelpFormatter, ArgumentDefaultsHelpFormatter):
    pass

def main():
    arg_defaults = {
        'num_processes': 1,
        'debug': False,
        'multilingual_database': False,
        'config': None,
        'max_length': 15,
        'max_length_relaxation': 0.1,
        'sample_rate': 44100,
        'language_def': None,
        'estimate_midi': False,
        'remove_cents': False,
        'pitch_extractor': 'parselmouth',
        'time_step': 0.005,
        'f0_min': 71,
        'f0_max': 1100,
        'voicing_threshold': 0.45,
        'write_ds': False, 
        'write_labels': None
    }

    parser = ArgumentParser(
        description='Converts a database with mono labels (NNSVS Format) into the DiffSinger format and saves it in a new folder in the path supplemented.',
        formatter_class=CombinedFormatter
    )
    parser.add_argument(
        'path',
        type=str, metavar='path',
        help='The path of the folder of the database.'
    )
    parser.add_argument(
        '--num-processes', '-T',
        type=int, default=1,
        help='Number of processes used to run segmentation faster. Enter 0 to use all cores.'
    )
    parser.add_argument(
        '--debug', '-D',
        action='store_true',
        help='Show debug logs.'
    )
    parser.add_argument(
        '--multilingual-database', '-M',
        action='store_true',
        help='Tells the segmenter that the database is multilingual. This requires a certain folder structure explained in the readme.'
    )
    parser.add_argument(
        '--config', '-C',
        type=str, metavar='path',
        help='Path to a .json config file to fill all the arguments. Arguments that are passed through the terminal will override the config.'
    )

    # segmenting options
    segmentation_group = parser.add_argument_group(
        title='segmentation options',
        description='Options related to segmentation behavior.'
    )
    segmentation_group.add_argument(
        '--max_length', '-l',
        type=float, metavar='sec', default=15,
        help='The maximum length of each segment.'
    )
    segmentation_group.add_argument(
        '--max-length-relaxation', '-R',
        type=float, metavar='sec', default=0.1,
        help='This length will be continuously added to the maximum length for segments that are too long for the maximum length to cut.'
    )
    segmentation_group.add_argument(
        '--sample-rate', '-r',
        type=int, default=44100,
        help='The sampling rate that the converted database will be in. Enter 0 to leave segments in their original sampling rate.'
    )
    segmentation_group.add_argument(
        '--language-def', '-L',
        type=str, metavar='path',
        help='Path to a language definition file to add data for phoneme duration prediction. Multilingual databases have language definitions by default.'
    )

    # midi estimation options
    midi_estimation_group = parser.add_argument_group(
        title='midi estimation options',
        description='Options related to MIDI estimation.'
    )
    midi_estimation_group.add_argument(
        '--estimate-midi', '-m',
        action='store_true',
        help='Enable MIDI estimation. Requires a language definition.'
    )
    midi_estimation_group.add_argument(
        '--remove-cents', '-c',
        action='store_true',
        help='Remove cent offsets from MIDI estimation.'
    )
    midi_estimation_group.add_argument(
        '--pitch-extractor', '-p',
        type=str, metavar='parselmouth | harvest | rmvpe', default='parselmouth',
        help='Pitch extractor used for MIDI estimation.'
    )
    midi_estimation_group.add_argument(
        '--time-step', '-t',
        type=float, metavar='sec', default=0.005,
        help='The time step used for pitch estimation.'
    )
    midi_estimation_group.add_argument(
        '--f0-min', '-f',
        type=float, metavar='Hz', default=71,
        help='The minimum pitch to detect in Hz.'
    )
    midi_estimation_group.add_argument(
        '--f0-max', '-F',
        type=float, metavar='Hz', default=1100,
        help='The maximum pitch to detect in Hz.'
    )
    midi_estimation_group.add_argument(
        '--voicing-threshold', '-v',
        type=float, default=0.45,
        help='Voicing threshold for pitch estimation (only for parselmouth).'
    )

    outputs_group = parser.add_argument_group(
        title='output options',
        description='Options related to output DiffSinger database.'
    )
    outputs_group.add_argument(
        '--write-ds', '-d',
        action='store_true',
        help='Write .ds files for usage with SlurCutter or for preprocessing.'
    )
    outputs_group.add_argument(
        '--write-labels', '-w',
        type=str, metavar='htk | aud',
        help='Write labels for when the labels after segmentation is needed. '
    )

    args, _ = parser.parse_known_args()
    print(args)
    dict_args = vars(args)
    modified_args = []
    for k, v in list(dict_args.items()):
        if v != arg_defaults.get(k):
            modified_args.append(k)
            dict_args[k] = v

    if args.config:
        config = load_config(Path(args.config))
        for k, v in config.items():
            if k not in modified_args:
                dict_args[k] = v

    
if __name__ == "__main__":
    main()
