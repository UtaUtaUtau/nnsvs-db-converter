from pathlib import Path
from argparse import Namespace
from copy import deepcopy
import yaml

def process_single(raw_path: Path, work_path: Path, lang: Path | None, config: Namespace) -> None:
    segment_path = work_path / 'wavs'
    transcript_path = work_path / 'transcriptions.csv'
    print(raw_path, work_path, lang)

def process_multilingual(base_path: Path, config: Namespace) -> None:
    work_path = base_path / 'diffsinger_db'
    lang_spk: dict[str, list[str]] = {}
    speakers: set[str] = set([])
    # get all languages and the speakers in them
    for lang_folder in filter(lambda x : x.is_dir(), base_path.glob('*')):
        lang = lang_folder.name
        lang_spk[lang] = []
        for spk_folder in filter(lambda x : x.is_dir(), lang_folder.glob('*')):
            lang_spk[lang].append(spk_folder.name)
            speakers.add(spk_folder.name)

    # turn speaker set into ids
    spk_ids = { spk : id for id, spk in enumerate(sorted(speakers)) }

    # prepare dictionary for speaker definitions
    speaker_def = {
        'datasets': []
    }
    for lang, spks in lang_spk.items():
        for spk in spks:
            speaker_def['datasets'].append({
                'raw_data_dir' : f'data/{lang}/{spk}',
                'spk_id': spk_ids[spk],
                'language': lang,
                'test_prefixes': []
            })

    # run process_single for all speakers
    # no multithreading for each speaker since speaker processing is what gets threading
    for lang, spks in lang_spk.items():
        for spk in spks:
            process_single(
                base_path / lang / spk,
                work_path / lang / spk,
                base_path / lang / f'{lang}.json',
                config
            )
    
    # save speaker definition
    work_path.mkdir(exist_ok=True)
    with open(work_path / 'spk_def.yaml', 'w') as f:
        yaml.dump(speaker_def, f, sort_keys=False)