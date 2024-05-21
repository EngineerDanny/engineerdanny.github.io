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
