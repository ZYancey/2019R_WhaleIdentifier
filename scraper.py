import os
import time

import requests
import progressbar
from bs4 import BeautifulSoup

# Define the base URL of the initial website and the directory to save downloaded files
base_url = 'https://whoicf2.whoi.edu/science/B/whalesounds/'
download_dir = 'Data'
widgets = ['\x1b[36m Processed: \t',
           progressbar.Counter(),
           progressbar.GranularBar(markers=" ▁▂▃▄▅▆▇█", left=' | ', right=' |'),
           ' (', progressbar.ETA(), ') ',
           ]


# Function to transform a string for use as a folder name
def transform_folder_name(name):
    # Remove special characters and replace spaces with underscores
    name = ''.join(char if char.isalnum() or char.isspace() else '' for char in name)
    # Convert to lowercase
    name = name.lower()
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    return name


# Function to download a file and save it to a specified directory
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)


# Function to scrape a page, extract links, and download files
def scrape_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all links to follow on the page
    links = soup.find_all('a', href=True, )
    i = 0
    print('Scraping Page')

    for link in links:
        link_url = link['href']
        link_text = link.find('h3')  # Extract the text within the h3 tag

        if link_url and link_text:
            i = i + 1
            link_text = link_text.text
            folder_name = transform_folder_name(link_text)
            link_url = base_url + link_url

            folder_path = os.path.join(download_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            print('Downloading (' + str(i) + '/32) : ' + link_text)
            download_audio_files(link_url, folder_path)


# Function to download audio files for a given link and folder
def download_audio_files(url, folder_path):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find and download audio files
    audio_links = soup.find_all('a', href=True, string='Download')
    total_links = len(audio_links)

    with progressbar.ProgressBar(max_value=total_links,
                                 widgets=widgets, ).start() as bar:
        for i, audio_link in enumerate(audio_links, start=1):
            audio_url = 'https://whoicf2.whoi.edu' + audio_link['href']
            audio_name = os.path.basename(audio_url)
            download_path = os.path.join(folder_path, audio_name)

            # Check if the file already exists, and only download if it doesn't
            if not os.path.exists(download_path):
                download_file(audio_url, download_path)
            else:
                #Keeps the printing from showing weirdly when content is already downloaded
                time.sleep(0.01)
            bar.update(i)


# Main script to start the download process
if __name__ == '__main__':
    initial_url = 'https://whoicf2.whoi.edu/science/B/whalesounds/index.cfm'

    scrape_page(initial_url)

    print("Download completed.")
