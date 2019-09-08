# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:43:22 2019

@author: huiyeon

"""
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 16:10:42 2019

@author: huiyeon
"""

# import packages
import time
from datetime import datetime
import pandas as pd

from selenium import webdriver
from bs4 import BeautifulSoup
from urllib.request import urlopen


# url 주소
url = 'https://twitter.com/septuor1'

# chrome driver
driver = webdriver.Chrome("C:/Users/huiyeon/Desktop/twt crawling/chromedriver.exe")
browser = driver.get(url)

def scrolling() :
    # scroll 무한 내리기
    # https://stackoverflow.com/a/28928684/1316860
    SCROLL_PAUSE_TIME = 5
    
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    
        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)
    
        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

scrolling()

# url 주소 지정
html = driver.page_source

# beautifulsoup 인자값 지정
soup = BeautifulSoup(html, 'html.parser')

# 크롤링 결과를 저장하기 위한 리스트 선언
uploader_list = []
date_list = []
contents_list = []
result = {}

# 엑셀로 저장하기 위한 변수
RESULT_PATH = 'C:/Users/huiyeon/Desktop/twt crawling/'
now = datetime.now()  # 파일 이름 현 시간으로 저장하기

def crawler() :
    # uploader 추출
    uploader = soup.select('.FullNameGroup > strong')
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
    outputFileName = 'selenium %s-%s-%s  %s시 %s분 %s초.xlsx' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    df.to_excel(RESULT_PATH+outputFileName,sheet_name='sheet1')
    
crawler()
