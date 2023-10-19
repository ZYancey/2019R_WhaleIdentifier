### 201R Group Project
# Cetacean Species Identification using Machine Learning

![Cetacean Image](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Whale_breaching_in_Alaska_2016-07-04.jpg/1024px-Whale_breaching_in_Alaska_2016-07-04.jpg)

## Table of Contents

- [Description]("placeholder")
- [Data]("placeholder")
- [Methodology]("placeholder")
- [License](LICENSE.md)

## Description

Whales, dolphins, porpoises, and other cetaceans are renowned for their social nature, often forming tight-knit groups. To communicate with pod members, they make different clicks, pops, and whistles that create the language of whale song. Whales are known to be gentle giants of the oceans; however, there has been a sudden increase in orca attacks on boats. To gain deeper insights into the world of cetaceans and foster safer coexistence with these majestic mammals, it would be helpful to be able to automatically identify species using only their vocalization. Such an endeavor would aid scientific research as well as offer the possibility of preemptive measures to avoid encounters with pods of orcas known to exhibit aggression. For this project, we propose using machine learning models to predict the different species of cetacean based on a small audio clip.

## Data

The data for this project is sourced from the [Watkins Marine Mammal Sound Database](https://whoicf2.whoi.edu/science/B/whalesounds/index.cfm). It consists of audio clips associated with 32 different species of cetaceans, with approximately 15,000 sound cuts, of which 1,694 files are considered the 'best cuts'. Each audio clip is about 2 to 15 seconds long and is available in waveform audio file format. The data is freely available for academic use.

## Methodology

We use the Mel Frequency Cepstral Coefficient (MFCC) algorithm to extract numerical features from the audio files. These features represent various qualities of the audio, including loudness, perceived pitch, timbre, texture, dynamics, and energy. MFCC has been used since the 1980s to convert frequency and time characteristics into numerical features. The number of features used can be customized based on your needs.

