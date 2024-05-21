---
layout: page
title: Apk Downloader
description: A repository that allows you to download APKs directly from Google Play by providing the package id of the app.
img: 
importance: 2
category: fun
giscus_comments: true
---


## :dart: About

This package helps you to directly download an APK from Google Play by providing the package id of the app. 
It also verifies the package ID of an app.
What the code does is to scrape a third party website to get the download link of the APK file.
Below is a detailed explanation of its functionality:

### Libraries and Modules
- `sys`: System-specific parameters and functions.
- `requests`: HTTP library for making requests.
- `BeautifulSoup` (from `bs4`): Library for parsing HTML and XML documents.
- `colored`: Module for coloring terminal text.
- `re`: Module for working with regular expressions.
- `progressbar`: Module for displaying progress bars.
- `itertools`: Module for creating iterators for efficient looping.
- `os`: Module for OS-dependent functionality.

### Base URLs
- `base_url`, `version_url`, `search_url`, `dl_url`, `g_play_url`: URLs for various actions like searching or downloading APKs.

## Headers
- Simulates a browser request to avoid being blocked by the server.

### Error Handling Functions
- `show_internet_error`: Checks internet connection.
- `show_arg_error`: Validates command-line argument format.
- `show_invalid_id_err`: Validates the package ID.

### Utility Functions
- `make_spinner`: Creates a spinner for the command line.
- `make_progress_bar`: Creates a progress bar for the download process.

### Main Function
- Validates command-line arguments.
- Greets the user and requests the package ID.
- Verifies the app's existence on the Google Play Store.
- Searches for the app on the APK website and finds download links.
- Requests a token for the download process.
- Downloads the APK file with a progress bar.


## :sparkles: Features

:heavy_check_mark: Feature 1.
Verifies the package ID of an app

:heavy_check_mark: Feature 2.
Downloads the latest apk version on Google Play

## :rainbow: Technologies

The following packages were used in this Python Project:

- [requests](https://pypi.org/project/requests/)
- [BeautifulSoup](https://pypi.org/project/beautifulsoup4/)
- [colored](https://pypi.org/project/colored/)
- [Regular Expressions (re)](https://docs.python.org/3/library/re.html)

## :white_check_mark: Requirements

Before starting :checkered_flag:, you need to have [Git](https://git-scm.com) and [Python🐍](https://www.python.org) installed.

## :checkered_flag: Starting

```bash
# Clone this project
$ git clone https://github.com/EngineerDanny/apk-downloader

# Access
$ cd apk-downloader

# Install dependencies
$ `pip install -r requirements.txt` or  `pip3 install -r requirements.txt`
```

## :rocket: Running

```bash
# Command
 $`python main.py {bundle identifier}` or `python3 main.py {bundle identifier}`

# Example
 $`python main.py com.imdb.mobile` or `python3 main.py com.imdb.mobile`
```
