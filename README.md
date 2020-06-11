# Drum and bass auto-DJ
_Python3 version._

This repository contains a Python3 adaptation of the automatic DJ system developed by Len Vande Veire, under the supervision of prof. Tijl De Bie. It has been designed for Drum and Bass music specifically.

The system is described in more detail in the paper [_Vande Veire, Len, and De Bie, Tijl, "From raw audio to a seamless mix: creating an automated DJ system for Drum and Bass." Journal Of Audio, Speech and Music Processing 2018, 13 (2018)_](https://doi.org/10.1186/s13636-018-0134-8).

The original Python2 implementation referenced in the paper can be found [here](https://bitbucket.org/ghentdatascience/dj/src/master/).

## Installation

The auto-DJ system has been tested for Ubuntu 16.04 LTS.  
It is recommended to install the auto-DJ using pip in a conda environment:

```
conda create -n "autodj" python=3.6.0
source activate autodj
pip install git+https://github.com/lenvdv/dnb-autodj-3
```

In case the installation fails when installing pyaudio, perform the following commands and retry the installation:

```
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
sudo apt-get install ffmpeg libav-tools
sudo pip install pyaudio
```

## Running the application

Run the application with the following command:

`python -m autodj.main`

The application is controlled using commands. A typical usage would be as follows:

```
$ python -m autodj.main

>> loaddir /home/username/music/drumandbass
Loading directory "/home/username/music/drumandbass"...
175 songs loaded (0 annotated).
>> annotate
Annotating music in song collection...
...
Done annotating!
>> play
Started playback!
```


The following commands are available:

* `loaddir <directory>` : Add the _.wav_ and _.mp3_ audio files in the specified directory to the pool of available songs.
* `annotate` : Annotate all the files in the pool of available songs that are not annotated yet. Note that this might take a while, and that in the current prototype this can only be interrupted by forcefully exiting the program (using the key combination `Ctrl+C`).
* `play` : Start a DJ mix. This command must be called after using the `loaddir` command on at least one directory with some annotated songs. Also used to continue playing after pausing.
* `play save`: Start a DJ mix, and save it to disk afterwards.
* `pause` : Pause the DJ mix.
* `stop` : Stop the DJ mix.
* `skip` : Skip to the next important boundary in the mix. This skips to either the beginning of the next crossfade, the switch point of the current crossfade or the end of the current crossfade, whichever comes first.
* `s` : Shorthand for the skip command
* `showannotated` : Shows how many of the loaded songs are annotated.
* `debug` : Toggle debug information output. This command must be used before starting playback, or it will have no effect.
* `stereo` : Toggle stereo audio support (enabled by default). Note: stereo audio is an experimental feature and leads to a longer processing time per crossfade.

To exit the application, use the `Ctrl+C` key combination.

## Changes in the Python3 version

The Python3 version of the auto-DJ system features the same functionality as the original prototype.
The main changes in the code base are:

* Stereo audio support (experimental, enabled by default).
* All annotations are saved in a single .json file per song.
* Code refactoring: the annotation modules are now in a separate subpackage, and are incorporated into the auto-DJ application using wrapper classes.


## Copyright information
Copyright 2020 Len Vande Veire.

Released under AGPLv3 license.
