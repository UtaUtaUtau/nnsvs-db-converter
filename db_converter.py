import numpy as np # Numpy <3
import soundfile as sf # wav read and write
from argparse import ArgumentParser, MetavarTypeHelpFormatter, ArgumentDefaultsHelpFormatter # fancy argument passinig
import glob # file finding
import os # make dirs and cmd pause
import traceback # errors
import math # maff
from copy import deepcopy # deepcopy <3
import logging # logger

pauses = ['pau', 'SP', 'AP']

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
            p = l.phone
            if p == 'pau':
                if l.length() <= max_sp_length:
                    p = 'SP'
                else:
                    p = 'AP'
            phones.append(p)
        return ' '.join(phones)

    def to_lengths_string(self): # space separated lengths
        return ' '.join([str(x.length()) for x in self.labels])

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
                        resegment.append(combine_labels(labels[s:e]))
                        logging.debug('shorter segment: %f', resegment[-1].length())
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

def to_diffsinger_line(name, label, max_sp_length = 1): # diffsinger format
    phones = label.to_phone_string(max_sp_length = max_sp_length)
    lengths = label.to_lengths_string()
    return f'{name}|funnythings|{phones}|rest|0|{lengths}|0\n'

try:
    parser = ArgumentParser(description='Converts a database with mono labels (NNSVS Format) into the DiffSinger format and saves it in a new folder in the path supplemented.', formatter_class=CombinedFormatter)
    parser.add_argument('path', type=str, metavar='path', help='The path of the folder of the database.')
    parser.add_argument('--max-length', '-l', type=float, default=15, help='The maximum length of the samples in seconds.')
    parser.add_argument('--max-silences', '-s', type=int, default=0, help='The maximum amount of silences (pau) in the middle of each segment. Set to a big amount to maximize segment lengths.')
    parser.add_argument('--max-sp-length', '-S', type=float, default=0.5, help='The maximum length for silences (pau) to turn into SP. SP is an arbitrary short pause from what I understand.')
    parser.add_argument('--write-labels', '-w', action='store_true', help='Write Audacity labels if you want to check segmentation labels.')
    parser.add_argument('--debug', '-d', action='store_true', help='Show debug logs.')
    
    args, _ = parser.parse_known_args()

    # Prepare locations
    diffsinger_loc = os.path.join(args.path, 'diffsinger_db')
    segment_loc = os.path.join(diffsinger_loc, 'wavs')
    transcript_loc = os.path.join(diffsinger_loc, 'transcriptions.txt')

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

    # actually make the directories
    logging.info('Making directories and files.')
    os.makedirs(diffsinger_loc, exist_ok=True)
    os.makedirs(segment_loc, exist_ok=True)
    transcript = open(transcript_loc, 'w', encoding='utf8')

    # go through all of it.
    for lab, wav in lab_wav.items():
        logging.info(f'Reading {wav}.')
        x, fs = sf.read(wav)
        
        logging.info(f'Segmenting {lab}.')
        _, file = os.path.split(lab)
        fname, _ = os.path.splitext(file)

        segments = read_label(lab).segment_label(max_length=args.max_length, max_silences=args.max_silences)
        logging.info(f'Splitting wave file and writing transcript.')
        for i in range(len(segments)):
            segment = segments[i]
            segment_name = f'{fname}_seg{i:03d}'
            s = int(fs * segment.start)
            e = int(fs * segment.end)
            sf.write(os.path.join(segment_loc, segment_name + '.wav'), x[s:e], fs)
            transcript.write(to_diffsinger_line(segment_name, segment, args.max_sp_length))
            if args.write_labels:
                write_label(os.path.join(segment_loc, segment_name + '.txt'), segment)
    # close the file. very important <3
    transcript.close()
        
except Exception as e:
    for i in traceback.format_exception(e.__class__, e, e.__traceback__):
        print(i, end='')
    os.system('pause')
