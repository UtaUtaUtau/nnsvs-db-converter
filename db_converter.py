import numpy as np # Numpy <3
import scipy.signal as signal # find peaks
import soundfile as sf # wav read and write
from argparse import ArgumentParser, MetavarTypeHelpFormatter, ArgumentDefaultsHelpFormatter # fancy argument passing
import glob # file finding
import os # make dirs and cmd pause
import traceback # errors
import math # maff
from copy import deepcopy # deepcopy <3
import csv # csv
import json # json
import librosa # key notation
import parselmouth as pm # speedy pitch detection
import time # timer
import logging # logger
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO, datefmt='%x %a %X')
import itertools # repeat
import concurrent.futures as futures # threading

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
    
    def __sub__(self, other):
        return Label(self.start - other, self.end - other, self.phone)
    
    def __add__(self, other):
        return Label(self.start + other, self.end + other, self.phone)

# Full label
class LabelList: # should've been named segment in hindsight...
    def __init__(self, labels):
        self.labels = deepcopy(labels) # list of Labels

    def __sub__(self, other):
        labels = []
        for lab in self.labels:
            labels.append(lab - other)
        return LabelList(labels)
    
    def __add__(self, other):
        labels = []
        for lab in self.labels:
            labels.append(lab + other)
        return LabelList(labels)

    def length(self): # total length in seconds
        lens = [x.length() for x in self.labels]
        return math.fsum(lens)

    def to_phone_string(self, max_sp_length = 1, detect_breaths=False): # space separated phonemes
        phones = []
        for l in self.labels: 
            if detect_breaths: # all non-SP and AP pauses are SP
                p = l.phone.replace('pau', 'SP').replace('sil', 'SP')
                phones.append(p)
            else: # old behavior when not using breath detection
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
    
    def to_midi_strings(self, x, fs, split_pos, time_step=0.005, f0_min=40, f0_max=1100, voicing_threshold=0.45, cents=False): # midi estimation
        global pauses
        f0, pps = get_pitch(x, fs, time_step=time_step, f0_min=f0_min, f0_max=f0_max, voicing_threshold=voicing_threshold) # get pitch
        pitch = f0
        pitch[pitch > 0] = librosa.hz_to_midi(pitch[pitch > 0])

        if pitch.size < self.length() * pps:
            pad = math.ceil(self.length() * pps) - pitch.size
            pitch = np.pad(pitch, [0, pad], mode='edge')

        note_seq = []
        note_dur = []

        temp_label = deepcopy(self) - self.labels[0].start # offset label to have it start at 0 because this receives segmented wavs
        for i in range(len(split_pos) - 1): # for each split
            s = split_pos[i]
            e = split_pos[i+1]

            note_lab = LabelList(temp_label[s:e]) # temp label
            p_s = math.floor((note_lab.labels[0].start) * pps)
            p_e = math.ceil((note_lab.labels[-1].end) * pps)

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

    def detect_breath(self, x, fs, time_step=0.005, f0_min=40, f0_max=1100, voicing_threshold=0.6, window=0.05, min_len=0.1, min_db=-60, min_centroid=2000):
        # Referenced from MakeDiffSinger/acoustic_forced_alignment/enhance_tg.py
        global pauses
        # Features needed
        sound = pm.Sound(x, sampling_frequency=fs)
        f0 = sound.to_pitch_ac(time_step=time_step, voicing_threshold=voicing_threshold, pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency'] # VUVs
        hop_size = int(time_step * fs)
        centroid = librosa.feature.spectral_centroid(y=x, sr=fs, hop_length=hop_size).squeeze(0) # centroid
        rms = librosa.amplitude_to_db(librosa.feature.rms(y=x, hop_length=hop_size).squeeze(0)) # RMS for peak/dip searching

        ap_ranges = [] # all AP ranges
        temp_label = deepcopy(self) - self.labels[0].start
        for lab in temp_label.labels: 
            if lab.phone not in pauses: # skip non pause phonemes
                continue

            if lab.length() < min_len: # skip pauses shorter than min breath len
                continue
 
            temp_ap_ranges = [] 
            br_start = None
            win_pos = lab.start
            while win_pos + window <= lab.end: # original algorithm from reference code
                all_unvoiced = (f0[int(win_pos / time_step) : int((win_pos + window) / time_step)] < f0_min).all()
                rms_db = 20 * np.log10(np.clip(sound.get_rms(from_time=win_pos, to_time=win_pos + window), 1e-12, 1))

                if all_unvoiced and rms_db >= min_db:
                    if br_start is None:
                        br_start = win_pos
                else:
                    if br_start is not None:
                        br_end = win_pos + window - time_step
                        if br_end - br_start >= min_len:
                            mean_centroid = centroid[int(br_start / time_step):int(br_end / time_step)].mean()
                            if mean_centroid >= min_centroid:
                                temp_ap_ranges.append((br_start, br_end))
                win_pos += time_step
            if br_start is not None:
                br_end = win_pos + window - time_step
                if br_end - br_start >= min_len:
                    mean_centroid = centroid[int(br_start / time_step):int(br_end / time_step)].mean()
                    if mean_centroid >= min_centroid:
                        temp_ap_ranges.append((br_start, br_end))
            
            if len(temp_ap_ranges) == 0: # skip if no AP was found
                continue
            
            # combine AP ranges with similar starts
            clean_ap_ranges = [list(temp_ap_ranges[0])]
            for ap_start, ap_end in temp_ap_ranges: 
                if clean_ap_ranges[-1][0] == ap_start:
                    clean_ap_ranges[-1][1] = ap_end
                else:
                    clean_ap_ranges.append([ap_start, ap_end])

            resized_ap_ranges = []
            # resize AP ranges by finding the peak, finding the dips and finding the deepest dip closest to the peak on both sides
            for ap_start, ap_end in clean_ap_ranges:
                s = int((ap_start + window) / time_step) # add window to remove potential energy spike from voiced section before the pause
                e = int(ap_end / time_step)

                if s >= e: # breath too short, can't analyze
                    resized_ap_ranges.append((ap_start, ap_end))
                    continue
                
                peak = np.argmax(rms[s:e]) + s
                peaks = signal.find_peaks_cwt(rms[s:e], np.arange(6, 10)) + s # if successful, it finds the breath peak better than argmax
                if peaks.size != 0:
                    peak = peaks[np.argmax(rms[peaks])]

                dips = signal.find_peaks_cwt(-rms[s:e], np.arange(1, 10)) + s
                
                if dips.size == 0: # can't resize if there are no dips
                    resized_ap_ranges.append((ap_start, ap_end))
                    continue
                
                # binary search nearby dips from peak
                L = 0
                R = len(dips) - 1

                while L != R:
                    m = math.ceil((L + R) / 2)
                    if dips[m] > peak:
                        R = m - 1
                    else:
                        L = m
                
                R = min(L + 1, len(dips) - 1)

                # find dips to the left and right until the dip before or after is higher than the current dip
                ss = dips[L]
                ee = dips[R]
                L_break = False
                for i in range(L, 0, -1):
                    ss = dips[i]
                    if rms[dips[i]] < rms[dips[i-1]]:
                        L_break = True
                        break
                
                R_break = False
                for i in range(R, len(dips)-1):
                    ee = dips[i]
                    if rms[dips[i]] < rms[dips[i+1]]:
                        R_break = True
                        break
                
                # if the end of the detected dips arrays were reached, it's probably better to use the original range
                if not L_break:
                    ss = ap_start
                else:
                    ss *= time_step

                if not R_break:
                    ee = ap_end
                else:
                    ee *= time_step

                resized_ap_ranges.append((ss, ee))
            ap_ranges.extend(resized_ap_ranges)

        # insert AP ranges into label
        for ap_start, ap_end in ap_ranges:
            pos = temp_label.binary_search(ap_start) # find position in array
            curr = temp_label.labels[pos]
            if curr.phone not in pauses or curr.start == ap_start: # if it wasn't a pause it's most likely before the detection
                if curr.start < ap_start: # index change not needed for curr.start == ap_start
                    pos += 1
                curr = deepcopy(temp_label.labels[pos])
                ap_start = curr.start
                del temp_label.labels[pos] # delete old label and replace with new
                temp_label.labels.insert(pos, Label(ap_start, min(ap_end, curr.end), 'AP'))
                if ap_end < curr.end: # add SP at the end if needed
                    temp_label.labels.insert(pos+1, Label(ap_end, curr.end, 'SP'))
            else:
                sp_end = curr.end
                curr.end = ap_start # push pause end for AP
                curr.phone = 'SP'
                temp_label.labels.insert(pos+1, Label(ap_start, min(ap_end, sp_end), 'AP')) # add AP
                if ap_end < sp_end: # add SP at the end if needed
                    temp_label.labels.insert(pos+2, Label(ap_end, sp_end, 'SP'))

        # cleanup labels from short SPs
        for i in range(len(temp_label) - 1, -1, -1):
            curr = temp_label.labels[i]
            if curr.length() < time_step and curr.phone in pauses: # good enough temporary short threshold
                temp_label.labels[i-1].end = curr.end
                del temp_label.labels[i]
        
        self.labels = (temp_label + self.labels[0].start).labels

    def binary_search(self, time):
        L = 0
        R = len(self.labels) - 1

        while L != R:
            m = math.ceil((L + R) / 2)
            if self.labels[m].start > time:
                R = m - 1
            else:
                L = m
        
        return L

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

def write_label(path, label, isHTK=True): # write label with start offset
    offset = label[0].start
    with open(path, 'w', encoding='utf8') as f:
        for l in label:
            if isHTK:
                f.write(f'{int(10000000 * (l.start - offset))} {int(10000000 * (l.end - offset))} {l.phone}\n')
            else:
                f.write(f'{l.start - offset}\t{l.end - offset}\t{l.phone}\n')

def get_pitch(x, fs, time_step=0.005, f0_min=40, f0_max=1100, voicing_threshold=0.45): # parselmouth F0
    f0 = pm.Sound(x, sampling_frequency=fs).to_pitch_ac(
        time_step=time_step, voicing_threshold=voicing_threshold,
        pitch_floor=f0_min, pitch_ceiling=f0_max
    ).selected_array['frequency']

    return f0, 1 / time_step

# From MakeDiffSinger/variance-temp-solution/get_pitch.py
def norm_f0(f0):
    f0 = np.log2(f0)
    return f0

def denorm_f0(f0, uv, pitch_padding=None):
    f0 = 2 ** f0
    if uv is not None:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0

def interp_f0(f0, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = norm_f0(f0)
    if sum(uv) == len(f0):
        f0[uv] = -np.inf
    elif sum(uv) > 0:
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return denorm_f0(f0, uv=None), uv


def write_ds(loc, wav, fs, time_step=0.005, f0_min=40, f0_max=1100, voicing_threshold=0.45, **kwargs):
    res = {'offset' : 0}
    res['text'] = kwargs['ph_seq']
    res['ph_seq'] = kwargs['ph_seq']
    res['ph_dur'] = kwargs['ph_dur']
    if 'ph_num' in list(kwargs.keys()):
        res['ph_num'] = kwargs['ph_num']
        if 'note_seq' in list(kwargs.keys()):
            res['note_seq'] = kwargs['note_seq']
            res['note_dur'] = kwargs['note_dur']
            res['note_slur'] = ' '.join(['0'] * len(kwargs['note_dur']))
    f0, _ = get_pitch(wav, fs, time_step=time_step, f0_min=f0_min, f0_max=f0_max, voicing_threshold=voicing_threshold)
    timestep = time_step
    f0, _ = interp_f0(f0)
    res['f0_seq'] = ' '.join([str(round(x, 1)) for x in f0])
    res['f0_timestep'] = str(timestep)

    with open(loc, 'w', encoding='utf8') as f:
        json.dump([res], f, indent=4)

def process_lab_wav_pair(segment_loc, lab, wav, args, lang=None):
    logging.info(f'Reading {wav}.')
    x, fs = sf.read(wav)

    if x.ndim > 1:
        x = np.mean(x, axis=1)
    
    logging.info(f'Segmenting {lab}.')
    _, file = os.path.split(lab)
    fname, _ = os.path.splitext(file)

    segments = read_label(lab).segment_label(max_length=args.max_length, max_silences=args.max_silences)
    logging.info('Splitting wave file and preparing transcription lines.')
    transcripts = []
    for i in range(len(segments)):
        segment = segments[i]
        segment_name = f'{fname}_seg{i:03d}'
        logging.info(f'Segment {i+1} / {len(segments)}')

        s = int(fs * segment.start)
        e = int(fs * segment.end)
        segment_wav = x[s:e]

        if args.detect_breaths:
            # detect_breath(self, x, fs, time_step=0.005, f0_min=40, f0_max=1100, voicing_threshold=0.6, window=0.05, min_len=0.06, min_db=-60, min_centroid=2000):
            segment.detect_breath(segment_wav, fs,
                                  time_step=args.time_step,
                                  f0_min=args.f0_min, f0_max=args.f0_max,
                                  voicing_threshold=args.voicing_threshold_breath,
                                  window=args.breath_window_size,
                                  min_len=args.breath_min_length,
                                  min_db=args.breath_db_threshold,
                                  min_centroid=args.breath_centroid_threshold)

        transcript_row = {
            'name' : segment_name,
            'ph_seq' : segment.to_phone_string(max_sp_length=args.max_sp_length, detect_breaths=args.detect_breaths),
            'ph_dur' : segment.to_lengths_string()
            }

        if args.language_def:
            transcript_row['ph_num'], split_pos = segment.to_phone_nums_string(lang=lang)
            dur = transcript_row['ph_dur'].split()
            num = [int(x) for x in transcript_row['ph_num'].split()]
            assert len(dur) == sum(num), 'Ops'
            if args.estimate_midi:
                note_seq, note_dur = segment.to_midi_strings(segment_wav, fs,split_pos,
                                                             time_step=args.time_step,
                                                             f0_min=args.f0_min, f0_max=args.f0_max,
                                                             voicing_threshold=args.voicing_threshold_midi,
                                                             cents=args.use_cents)
                transcript_row['note_seq'] = note_seq
                transcript_row['note_dur'] = note_dur

        all_pau = np.all(np.array(list(map(lambda x : x in pauses, transcript_row['ph_seq'].split()))))
        all_rest = False
        if args.estimate_midi:
            all_rest = np.all(np.array(list(map(lambda x : x == 'rest', transcript_row['note_seq'].split()))))

        if not (all_pau or all_rest):
            sf.write(os.path.join(segment_loc, segment_name + '.wav'), segment_wav, fs)
            transcripts.append(transcript_row)
            if args.write_labels:
                isHTK = args.write_labels.lower() == 'htk'
                write_label(os.path.join(segment_loc, segment_name + ('.lab' if isHTK else '.txt')), segment, isHTK)
            
            if args.write_ds:
                write_ds(os.path.join(segment_loc, segment_name + '.ds'), segment_wav, fs,
                         time_step=args.time_step, f0_min=args.f0_min, f0_max=args.f0_max,
                         voicing_threshold=args.voicing_threshold_midi, **transcript_row)
        else:
            logging.warning('Detected pure silence either from segment label or note sequence. Skipping.')
    
    return transcripts

if __name__ == '__main__':
    try:
        parser = ArgumentParser(description='Converts a database with mono labels (NNSVS Format) into the DiffSinger format and saves it in a new folder in the path supplemented.', formatter_class=CombinedFormatter)
        parser.add_argument('path', type=str, metavar='path', help='The path of the folder of the database.')
        parser.add_argument('--max-length', '-l', type=float, default=15, help='The maximum length of the samples in seconds.')
        parser.add_argument('--max-silences', '-s', type=int, default=0, help='The maximum amount of silences (pau) in the middle of each segment. Set to a big amount to maximize segment lengths.')
        parser.add_argument('--max-sp-length', '-S', type=float, default=0.5, help='The maximum length for silences (pau) to turn into SP. Ignored when breath detection is enabled. Only here for fallback.')
        parser.add_argument('--language-def', '-L', type=str, metavar='path', help='The path of the language definition .json file. If present, phoneme numbers will be added.')
        parser.add_argument('--estimate-midi', '-m', action='store_true', help='Whether to estimate MIDI or not. Only works if a language definition is added for note splitting.')
        parser.add_argument('--use-cents', '-c', action='store_true', help='Add cent offsets for MIDI estimation.')
        parser.add_argument('--time-step', '-t', type=float, default=0.005, help='The time step used for all frame-by-frame analysis functions.')
        parser.add_argument('--f0-min', '-f', type=float, default=40, help='The minimum F0 to detect in Hz. Used in MIDI estimation and breath detection.')
        parser.add_argument('--f0-max', '-F', type=float, default=1100, help='The maximum F0 to detect in Hz. Used in MIDI estimation and breath detection.')
        parser.add_argument('--voicing-threshold-midi', '-V', type=float, default=0.45, help='The voicing threshold used for MIDI estimation.')
        parser.add_argument('--detect-breaths', '-B', action='store_true', help='Detect breaths within all pauses.')
        parser.add_argument('--voicing-threshold-breath', '-v', type=float, default=0.6, help='The voicing threshold used for breath detection.')
        parser.add_argument('--breath-window-size', '-W', type=float, default=0.05, help='The size of the window in seconds for breath detection.')
        parser.add_argument('--breath-min-length', '-b', type=float, default=0.1    , help='The minimum length of a breath in seconds.')
        parser.add_argument('--breath-db-threshold', '-e', type=float, default=-60, help='The threshold in the RMS of the signal in dB to detect a breath.')
        parser.add_argument('--breath-centroid-threshold', '-C', type=float, default=2000, help='The threshold in the spectral centroid of the signal in Hz to detect a breath.')
        parser.add_argument('--write-ds', '-D', action='store_true', help='Write .ds files for usage with SlurCutter or for preprocessing.')
        parser.add_argument('--write-labels', '-w', type=str, metavar='htk|aud', help='Write labels if you want to check segmentation labels. "htk" gives HTK style labels, "aud" gives Audacity style labels.')
        parser.add_argument('--num-processes', '-T', type=int, default=1, help='Number of processes to run for faster segmentation. Enter 0 to use all cores.')
        parser.add_argument('--debug', '-d', action='store_true', help='Show debug logs.')
        
        args, _ = parser.parse_known_args()
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Prepare locations
        diffsinger_loc = os.path.join(args.path, 'diffsinger_db')
        segment_loc = os.path.join(diffsinger_loc, 'wavs')
        transcript_loc = os.path.join(diffsinger_loc, 'transcriptions.csv')

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

        t0 = time.perf_counter()
        if args.num_processes == 1:
            transcripts = []
            for lab, wav in lab_wav.items():
                transcripts.extend(process_lab_wav_pair(segment_loc, lab, wav, args, lang))
            logging.info('Writing all transcripts.')
            transcript.writerows(transcripts)
        else:
            workers = args.num_processes
            if workers == 0:
                workers = None
                logging.info('Starting process pool with default number of threads.')
            else:
                logging.info(f'Starting process pool with {workers} threads.')
            with futures.ProcessPoolExecutor(max_workers=workers) as executor:
                results = executor.map(process_lab_wav_pair, itertools.repeat(segment_loc), lab_wav.keys(), lab_wav.values(), itertools.repeat(args), itertools.repeat(lang))
            logging.info('Writing all transcripts.')
            for res in results:
                transcript.writerows(res)
        runtime = time.perf_counter() - t0
        logging.info(f'Took {runtime} seconds')

        # close the file. very important <3
        transcript_f.close()
            
    except Exception as e:
        for i in traceback.format_exception(e.__class__, e, e.__traceback__):
            print(i, end='')
        os.system('pause')
