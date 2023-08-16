# nnsvs-db-converter
 Python script to convert NNSVS DBs to DiffSinger without needing the NNSVS Python Library.
# Information
 This is a Python script for converting singing databases with HTS mono labels into the DiffSinger format. This Python script deals with sample segmentation to mostly ensure that samples have a maximum given length and a maximum amount of pauses in between. The recognized silence phonemes by this script are `pau`, `SP`, and `AP`. It also assumes the labels have silences labeled at the start and end.
 
 This Python script only requires four external libraries to work, unlike the existing one which needs the NNSVS library, which might be hard to install for most people.
# How to Use

## Main Usage
 1. Have a Python install, preferably Python 3.8.
 2. Install the `numpy`, `soundfile`, `parselmouth`, and `librosa` libraries through pip like so:
 ```pip install numpy soundfile librosa praat-parselmouth```
 3. Pass the database with either method:
    1. Drag and drop the database folder over the script.
    2. Run the script using the terminal and pass terminal arguments.

## Language Definition

 Language definition is a `.json` file containing the vowels and the "liquids" of the phoneme system the labels are on. Vowels are where the splits will be made. Liquids are for special cases where a consonant is wanted to be the first part of the split. Here's a quick example `.json` file and a sample split.

```json
{
   "vowels" : ["a", "i", "u", "e", "o", "N", "A", "I", "U", "E", "O"],
   "liquids" : ["w", "y"]
}
```

```
ph_seq    |  SP | k | a | a | k | a | w | a | k | w | a | ... 
ph_num    |    2    | 1 |   2   |   2   |   2   |   2   | ... 
```

## Help Text from the file itself
```
usage: db_converter.py [-h] [--max-length float] [--max-silences int] [--max-sp-length float] [--write-labels]
                       [--language-def path] [--estimate-midi] [--debug]
                       path

Converts a database with mono labels (NNSVS Format) into the DiffSinger format and saves it in a new folder in the
path supplemented.

positional arguments:
  path                  The path of the folder of the database.

optional arguments:
  -h, --help            show this help message and exit
  --max-length float, -l float
                        The maximum length of the samples in seconds. (default: 15)
  --max-silences int, -s int
                        The maximum amount of silences (pau) in the middle of each segment. Set to a big amount to
                        maximize segment lengths. (default: 0)
  --max-sp-length float, -S float
                        The maximum length for silences (pau) to turn into SP. SP is an arbitrary short pause from
                        what I understand. (default: 0.5)
  --write-labels, -w    Write Audacity labels if you want to check segmentation labels. (default: False)
  --language-def path, -L path
                        The path of the language definition .json file. If present, phoneme numbers will be added.
                        (default: None)
  --estimate-midi, -m   Whether to estimate MIDI or not. Only works if a language definition is added for note
                        splitting. (default: False)
  --debug, -d           Show debug logs. (default: False)
```