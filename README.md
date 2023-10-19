### 201R Group Project

# Cetacean Species Identification using Machine Learning

![Cetacean Image](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Whale_breaching_in_Alaska_2016-07-04.jpg/1024px-Whale_breaching_in_Alaska_2016-07-04.jpg)

## Table of Contents

- [Description](https://github.com/ZYancey/201R_WhaleIdentifier#description)
- [Data](https://github.com/ZYancey/201R_WhaleIdentifier#data)
- [Methodology](https://github.com/ZYancey/201R_WhaleIdentifier#methodology)
- [License](LICENSE)

## Description

Whales, dolphins, porpoises, and other cetaceans are renowned for their social nature, often forming tight-knit groups. To communicate with pod members, they make different clicks, pops, and whistles that create the language of whale song. Whales are known to be gentle giants of the oceans; however, there has been a sudden increase in orca attacks on boats. To gain deeper insights into the world of cetaceans and foster safer coexistence with these majestic mammals, it would be helpful to be able to automatically identify species using only their vocalization. Such an endeavor would aid scientific research as well as offer the possibility of preemptive measures to avoid encounters with pods of orcas known to exhibit aggression. For this project, we propose using machine learning models to predict the different species of cetacean based on a small audio clip.

## Data

The data for this project is sourced from the [Watkins Marine Mammal Sound Database](https://whoicf2.whoi.edu/science/B/whalesounds/index.cfm). It consists of audio clips associated with 32 different species of cetaceans, with approximately 15,000 sound cuts, of which 1,694 files are considered the 'best cuts'. Each audio clip is about 2 to 15 seconds long and is available in waveform audio file format. The data is freely available for academic use.

## Methodology


## Python Script to Scrape and Download Audio Files

This Python script is designed to scrape a web page and download audio files located behind specified links. It employs the `requests` library to fetch web pages, `BeautifulSoup` for HTML parsing, and a progress bar to track the download progress. The script iterates through the initial webpage, extracts links, and downloads audio files into folders that mirror the structure of the initial website, naming the folders based on the text within the 'h3' tags of the links. Additionally, the script cleans up folder names to assist in data cleaning and processing.

#### How to Run

1. Install the required Python libraries if not already installed:

   ```bash
   pip install requests beautifulsoup4 progressbar2

2. Run the script from the root of the repository using Python:

   ```bash
   python scraper.py

The script will start scraping the website, create appropriate folders, and download audio files, showing a progress bar to track the process. Once the script completes, you'll find the downloaded audio files organized in folders corresponding to the structure of the initial website within the specified download directory.
