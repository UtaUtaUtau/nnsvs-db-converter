import numpy as np # Numpy <3
import soundfile as sf # wav read and write
from argparse import ArgumentParser, MetavarTypeHelpFormatter, ArgumentDefaultsHelpFormatter # fancy argument passinig
import glob # file finding
import os # make dirs and cmd pause
import traceback # errors
import math # maff
from copy import deepcopy # deepcopy <3
import csv # csv
import json # json
import librosa # key notation
import parselmouth as pm # speedy pitch detection
import logging # logger

pauses = ['sil', 'pau', 'SP', 'AP']

# Combined formatter for argparse to show typing and defaults
class CombinedFormatter(MetavarTypeHelpFormatter, ArgumentDefaultsHelpFormatter):
    pass

# Simple label class cuz I'm quirky
class Label:
    def __init__(self, s, e, p):
        self.start = s # start time
        self.end = e # end time
        self.phone = p # phoneme

    def length(self): # label length
        l = self.end - self.start
        if l < 0:
            logging.warning('Negative length.')
        return l

# Full label
class LabelList: # should've been named segment in hindsight...
    def __init__(self, labels):
        self.labels = deepcopy(labels) # list of Labels

    def length(self): # total length in seconds
        lens = [x.length() for x in self.labels]
        return math.fsum(lens)

    def to_phone_string(self, max_sp_length = 1): # space separated phonemes
        phones = []
        for l in self.labels: # turn short silences to SP
            p = l.phone.replace('sil', 'SP')
            if p == 'pau':
                if l.length() <= max_sp_length:
                    p = 'SP'
                else:
                    p = 'AP'
            phones.append(p)
        return ' '.join(phones)

    def to_lengths_string(self): # space separated lengths
        return ' '.join([str(round(x.length(), 12)) for x in self.labels])
    
    def to_phone_nums_string(self, lang): # phoneme separations
        # Find all vowel positions
        vowel_pos = []
        if self.labels[0].phone not in pauses + lang['vowels']:
            vowel_pos.append(0)

        for i in range(len(self.labels)):
            l = self.labels[i]
            if l.phone in lang['vowels']:
                prev_l = self.labels[i-1]
                if prev_l.phone in lang['liquids']: # check liquids before vowel. move position if liquid with consonant before is found
                    if self.labels[i-2].phone not in lang['vowels']:
                        vowel_pos.append(i-1)
                    else:
                        vowel_pos.append(i)
                else:
                    vowel_pos.append(i)
            elif l.phone in pauses:
                vowel_pos.append(i)
        vowel_pos.append(len(self))

        # use diff to calculate ph_num
        ph_num = np.diff(vowel_pos)
        return ' '.join(map(str, ph_num)), vowel_pos
    
    def to_midi_strings(self, x, fs, split_pos, cents=False): # midi estimation
        global pauses
        f0, pps = get_pitch(x, fs) # get pitch
        pitch = f0
        pitch[pitch > 0] = librosa.hz_to_midi(pitch[pitch > 0])

        if pitch.size < self.length() * pps:
            pad = math.ceil(self.length() * pps) - pitch.size
            pitch = np.pad(pitch, [0, pad], mode='edge')

        note_seq = []
        note_dur = []

        offset = self.labels[0].start
        for i in range(len(split_pos) - 1): # for each split
            s = split_pos[i]
            e = split_pos[i+1]

            note_lab = LabelList(self.labels[s:e]) # temp label
            p_s = math.floor((note_lab.labels[0].start - offset) * pps)
            p_e = math.ceil((note_lab.labels[-1].end - offset) * pps)

            # check for rests
            is_rest = False
            note_lab_phones = [x.phone for x in note_lab.labels]
            for pau in pauses:
                if pau in note_lab_phones:
                    is_rest = True
                    break
            
            if is_rest:
                note_seq.append('rest') 
            else: # get modal pitch
                note_pitch = pitch[p_s:p_e]
                note_pitch = note_pitch[note_pitch > 0]
                if note_pitch.size > 0:
                    counts = np.bincount(np.round(note_pitch).astype(np.int64))
                    midi = counts.argmax()
                    if cents:
                        midi = np.mean(note_pitch[(note_pitch >= midi - 0.5) & (note_pitch < midi + 0.5)])
                    note_seq.append(librosa.midi_to_note(midi, cents=cents, unicode=False))
                else:
                    note_seq.append('rest')

            note_dur.append(note_lab.length())
        
        return ' '.join(note_seq), ' '.join(map(lambda x : str(round(x, 12)), note_dur))


    def __len__(self): # number of labels
        return len(self.labels)

    def __getitem__(self, key): # when i needed LabelList[x] back then
        return self.labels[key]

    @property
    def start(self): # segment start
        return self.labels[0].start

    @property
    def end(self): # segment end
        return self.labels[-1].end

    def shorten_label(self, max_length = 15):
        self_len = self.length()
        if self_len > max_length: # If label length is bigger
            # Calculate shortest possible length
            pau_len = self.labels[0].length() + self.labels[-1].length()
            shortest = self_len - pau_len
            
            if shortest > max_length: # can't be shortened anymore. just make it not exist !
                return None
            elif shortest == max_length: # extreme shortening
                return LabelList(self.labels[1:-1])
            else: # shorten pau boundaries to crop to max length
                labels = deepcopy(self.labels)
                a = labels[0]
                b = labels[-1]
                short_pau = min(a.length(), b.length()) # get shorter length pau
                same_length_pau = shortest + short_pau * 2 

                if same_length_pau < max_length: # shorten long pau first if the sample with similar length paus is shorter
                    long_pau = short_pau + max_length - same_length_pau
                    if a.length() > b.length():
                        a.start = a.end - long_pau
                    else:
                        b.end = b.start + long_pau
                else: # shorten both paus by the shorter length
                    k = (max_length - shortest) / (2 * short_pau)
                    a.start = a.end - k * short_pau
                    b.end = b.start + k * short_pau
                
                return LabelList(labels)
        else:
            return deepcopy(self) # no need to do anything

    def segment_label(self, max_length = 15, max_silences = 0): # label splitting...
        global pauses
        # Split by silences first
        labels = []
        pau_pos = []
        # Find all pau positions
        for i in range(len(self.labels)):
            l = self.labels[i]
            if l.phone in pauses:
                pau_pos.append(i)

        # segment by pau positions
        for i in range(len(pau_pos)-1):
            s = pau_pos[i]
            e = pau_pos[i+1]+1
            labels.append(LabelList(self.labels[s:e]))

        resegment = []
        if max_silences > 0: # concatenate labels for resegmentation
            s = 0
            e = 1
            while s < len(labels): # combine labels one by one until you reach max silences or max length
                curr = combine_labels(labels[s:e])
                if e - s - 1 >= max_silences:
                    temp = curr.shorten_label(max_length=max_length)
                    if temp:
                        resegment.append(temp)
                    else:
                        e -= 1
                        resegment.append(combine_labels(labels[s:e]))
                    s = e
                elif curr.length() > max_length:
                    logging.debug('long len: %f', curr.length())
                    temp = curr.shorten_label(max_length=max_length)
                    if temp:
                        resegment.append(temp)
                        logging.debug('cut down')
                    else:
                        e -= 1
                        shorter = labels[s:e]
                        if len(shorter) > 0:
                            resegment.append(combine_labels(shorter))
                            logging.debug('shorter segment: %d', len(shorter))
                        else:
                            logging.warning('A segment could not be shortened to the given maximum length, this sample might be slightly longer than the maximum length you desire.')
                            resegment.append(curr)
                            e += 1
                    s = e
                e += 1
        else: # first segmentation pass already left it with no pau in between
            for l in labels:
                curr = l.shorten_label()
                if curr:
                    resegment.append(curr)

        return resegment

def combine_labels(labels): # combining labels that pau boundaries
    if len(labels) == 1:
        return labels[0]

    res = []
    for l in labels:
        res = res[:-1]
        res.extend(l)
    return LabelList(res)

def label_from_line(line): # yeah..
    s, e, p = line.strip().split()
    s = float(s) / 10000000
    e = float(e) / 10000000
    return Label(s, e, p)

def read_label(path): # yeah..
    labels = []
    for line in open(path).readlines():
        labels.append(label_from_line(line))

    return LabelList(labels)

def write_label(path, label): # write audacity label with start offset
    offset = label[0].start
    with open(path, 'w', encoding='utf8') as f:
        for l in label:
            f.write(f'{l.start - offset}\t{l.end - offset}\t{l.phone}\n')

def get_pitch(x, fs): # parselmouth F0
    time_step = 0.005
    f0_min = 65
    f0_max = 1760

    f0 = pm.Sound(x, sampling_frequency=fs).to_pitch_ac(
        time_step=time_step, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max
    ).selected_array['frequency']

    return f0, 1 / time_step

try:
    parser = ArgumentParser(description='Converts a database with mono labels (NNSVS Format) into the DiffSinger format and saves it in a new folder in the path supplemented.', formatter_class=CombinedFormatter)
    parser.add_argument('path', type=str, metavar='path', help='The path of the folder of the database.')
    parser.add_argument('--max-length', '-l', type=float, default=15, help='The maximum length of the samples in seconds.')
    parser.add_argument('--max-silences', '-s', type=int, default=0, help='The maximum amount of silences (pau) in the middle of each segment. Set to a big amount to maximize segment lengths.')
    parser.add_argument('--max-sp-length', '-S', type=float, default=0.5, help='The maximum length for silences (pau) to turn into SP. SP is an arbitrary short pause from what I understand.')
    parser.add_argument('--language-def', '-L', type=str, metavar='path', help='The path of the language definition .json file. If present, phoneme numbers will be added.')
    parser.add_argument('--estimate-midi', '-m', action='store_true', help='Whether to estimate MIDI or not. Only works if a language definition is added for note splitting.')
    parser.add_argument('--use_cents', '-c', action='store_true', help='Add cent offsets for MIDI estimation.')
    parser.add_argument('--write-labels', '-w', action='store_true', help='Write Audacity labels if you want to check segmentation labels.')
    parser.add_argument('--debug', '-d', action='store_true', help='Show debug logs.')
    
    args, _ = parser.parse_known_args()

    # Prepare locations
    diffsinger_loc = os.path.join(args.path, 'diffsinger_db')
    segment_loc = os.path.join(diffsinger_loc, 'wavs')
    transcript_loc = os.path.join(diffsinger_loc, 'transcriptions.csv')
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.DEBUG if args.debug else logging.INFO, datefmt='%x %a %X')

    # Label finding
    logging.info('Finding all labels.')
    lab_locs = glob.glob(os.path.join(args.path, '**/*.lab'), recursive=True)
    lab_locs.sort()
    logging.info(f'Found {len(lab_locs)} label' + ('.' if len(lab_locs) == 1 else 's.'))

    # wave equivalent finding
    lab_wav = {}
    for i in lab_locs:
        _, file = os.path.split(i)
        fname, _ = os.path.splitext(file)
        temp = glob.glob(args.path + f'/**/{fname}.wav', recursive=True)
        if len(temp) == 0:
            raise FileNotFoundError(f'No wave file equivalent of {file} was found.')
        if len(temp) > 1:
            logging.warning(f'Found more than one instance of a wave file equivalent for {file}. Picking {temp[0]}.')
        lab_wav[i] = temp[0]

    # check for language definition
    lang = None
    transcript_header = ['name', 'ph_seq', 'ph_dur']
    if args.language_def:
        with open(args.language_def) as f:
            lang = json.load(f)
        transcript_header.append('ph_num')

    if args.estimate_midi:
        transcript_header.extend(['note_seq', 'note_dur'])

    # actually make the directories
    logging.info('Making directories and files.')
    os.makedirs(diffsinger_loc, exist_ok=True)
    os.makedirs(segment_loc, exist_ok=True)

    # prepare transcript.csv
    transcript_f = open(transcript_loc, 'w', encoding='utf8', newline='')
    transcript = csv.DictWriter(transcript_f, fieldnames=transcript_header)
    transcript.writeheader()

    # go through all of it.
    for lab, wav in lab_wav.items():
        logging.info(f'Reading {wav}.')
        x, fs = sf.read(wav)
        
        logging.info(f'Segmenting {lab}.')
        _, file = os.path.split(lab)
        fname, _ = os.path.splitext(file)

        segments = read_label(lab).segment_label(max_length=args.max_length, max_silences=args.max_silences)
        logging.info('Splitting wave file and writing to transcription file.')
        for i in range(len(segments)):
            segment = segments[i]
            segment_name = f'{fname}_seg{i:03d}'
            logging.info(f'Segment {i+1} / {len(segments)}')

            transcript_row = {
                'name' : segment_name,
                'ph_seq' : segment.to_phone_string(max_sp_length=args.max_sp_length),
                'ph_dur' : segment.to_lengths_string()
                }
            
            s = int(fs * segment.start)
            e = int(fs * segment.end)
            segment_wav = x[s:e]

            if args.language_def:
                transcript_row['ph_num'], split_pos = segment.to_phone_nums_string(lang=lang)
                dur = transcript_row['ph_dur'].split()
                num = [int(x) for x in transcript_row['ph_num'].split()]
                assert len(dur) == sum(num), 'Ops'
                if args.estimate_midi:
                    note_seq, note_dur = segment.to_midi_strings(segment_wav, fs, split_pos, cents=args.use_cents)
                    transcript_row['note_seq'] = note_seq
                    transcript_row['note_dur'] = note_dur

            all_pau = np.all(np.array(list(map(lambda x : x in pauses, transcript_row['ph_seq'].split()))))
            all_rest = False
            if 'note_seq' in transcript_header:
                all_rest = np.all(np.array(list(map(lambda x : x == 'rest', transcript_row['note_seq'].split()))))

            if not (all_pau or all_rest):
                sf.write(os.path.join(segment_loc, segment_name + '.wav'), segment_wav, fs)
                transcript.writerow(transcript_row)
                if args.write_labels:
                    write_label(os.path.join(segment_loc, segment_name + '.txt'), segment)
            else:
                logging.warning('Detected pure silence either from segment label or note sequence. Skipping.')
    # close the file. very important <3
    transcript_f.close()
        
except Exception as e:
    for i in traceback.format_exception(e.__class__, e, e.__traceback__):
        print(i, end='')
    os.system('pause')
