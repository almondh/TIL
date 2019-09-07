# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:43:22 2019

@author: huiyeon
"""


# import packages
import requests as rq
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.parse import urlparse, parse_qs, urljoin

import os
import json
import re
import time
from datetime import datetime
import pandas as pd


"""
### 모르게써!!!!!!!!!!1
# 스크롤 내리기
browser = webdriver.Chrome("C:/Users/huiyeon/Desktop/twt crawling/chromedriver.exe")
browser = driver.get('https://twitter.com/septuor1')

for i in range(10):
    browser.execute_script("window.scrollBy(0,100000)")


"""
"""
# start
twt_url = urlopen('https://twitter.com/septuor1')
soup = BeautifulSoup(twt_url, 'html.parser')

# 웹 문서 전체 가져오기
print(soup)

# title 가져오기
print(soup.head.title)

"""    


# 크롤링 결과를 저장하기 위한 리스트 선언
uploader_list = []
date_list = []
contents_list = []
result = {}

# 엑셀로 저장하기 위한 변수
RESULT_PATH = 'C:/Users/huiyeon/Desktop/twt crawling/'
now = datetime.now()  # 파일 이름 현 시간으로 저장하기

def crawler() :
    """
    스크롤 내리는 부분 추가하기?
    """
    
    # url 주소 지정
    twt_url = urlopen('https://twitter.com/septuor1')
    
    # beautifulsoup 인자값 지정
    soup = BeautifulSoup(twt_url, 'html.parser')

    
    # uploader 추출
    uploader = soup.select('.FullNameGroup')
    for _ in uploader:
        uploader_list.append(_.text.strip())
    
    # date 추출
    timestamp = soup.select('span._timestamp.js-short-timestamp')
    for date in timestamp:
        date_list.append(date.text)

    # content 추출
    contents = soup.select('.js-tweet-text-container')
    for content in contents:
        contents_list.append(content.text.strip())
    
    # list -> dictionary
    result = {"date" : date_list, "uploader" : uploader_list, "content" : contents_list}
    df = pd.DataFrame(result)
    
    
    # 새로 만들 파일이름 지정
    outputFileName = '%s-%s-%s  %s시 %s분 %s초 merging.xlsx' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    df.to_excel(RESULT_PATH+outputFileName,sheet_name='sheet1')


crawler()
