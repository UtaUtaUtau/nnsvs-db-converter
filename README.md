# nnsvs-db-converter
 Python script to convert NNSVS DBs to DiffSinger without needing the NNSVS Python Library.
# Information
 This is a Python script for converting singing databases with HTS mono labels into the DiffSinger format. This Python script deals with sample segmentation to mostly ensure that samples have a maximum given length and a maximum amount of pauses in between. The recognized silence phonemes by this script are `sil`, `pau`, `SP`, and `AP`. `sil` is always converted into `SP`. It also assumes the labels have silences labeled at the start and end.
 
 This Python script only requires four external libraries to work, unlike the existing one which needs the NNSVS library, which might be hard to install for most people.
# How to Use

## Main Usage

### Through Python
 1. Have a Python install, preferably Python 3.8.
 2. Install the `numpy`, `soundfile`, `parselmouth`, and `librosa` libraries through pip like so:
 ```pip install numpy soundfile librosa praat-parselmouth```
 3. Pass the database with either method:
    1. Drag and drop the database folder over the script (similar to [basic conversion](#basic-conversion)).
    2. Run the script using the terminal and pass terminal arguments.
        - You can access the terminal by typing `cmd` on the address bar in File Explorer. It will open command prompt with the current working directory as the one opened in File Explorer.
    
### Using portable version
 1. Download the portable version in the releases page.
 2. Pass the database with either method:
    1. Drag and drop the database folder over `db_converter.bat` (similar to [basic conversion](#basic-conversion)).
    2. Run the script using the terminal and pass terminal arguments.
        - You can access the terminal by typing `cmd` on the address bar in File Explorer. It will open command prompt with the current working directory as the one opened in File Explorer.

## Example commands

 Anything within the square brackets is optional. Read more about them in [the help text.](#help-text-from-the-file-itself)
 
 These example commands assume that your terminal's current working directory is where `db_converter.py` or `db_converter.bat` is in. If you're using the embeddable version, replace `python db_converter.py` to `db_converter.bat`.
 
 **Tip:** You can drag and drop a file or a folder over the terminal window to automatically add the path in the arguments.
 
### Basic Conversion

 If you want to use [MakeDiffSinger](https://github.com/openvpi/MakeDiffSinger) still to do all the extra variance data needed.
 
 **Requirements:** NNSVS-style Database (.wav and .lab only)
 
```cmd
python db_converter.py [-l max_length -s max_silences -S max_sp_length -w] path/to/nnsvs/db 
```

### Conversion with variance duration support

 If want to use a DiffSinger variance model for timing prediction only.
 
 **Requirements:** NNSVS-style Database (.wav and .lab only), [Language Definition](#language-definition)
 
```cmd
python db_converter.py [-l max_length -s max_silences -S max_sp_length -w] -L path/to/language-def.json path/to/nnsvs/db
```

### Conversion with variance duration and pitch support

 If you want to use a DiffSinger variance model for timing and pitch prediciton.
 
 **Requirements:** NNSVS-style Database (.wav and .lab only), [Language Definition](#language-definition)

```cmd
python db_converter.py [-l max_length -s max_silences -S max_sp_length -w] -L path/to/language-def.json -m path/to/nnsvs/db
```

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