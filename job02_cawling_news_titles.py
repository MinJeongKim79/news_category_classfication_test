from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
import datetime

category = ['Politiccs', 'Economic', 'Social', 'Culture', 'World', 'IT']
df_titles = pd.DataFrame()
xpath_roots = [
    'div[4]/div/div[2]',  # Politics
    'div[5]/div/div[2]',  # Economic
    'div[4]/div/div[2]',  # Social
    'div[4]/div/div[2]',  # Culture
    'div[4]/div/div[1]',  # World
    'div[4]/div/div[2]'   # IT
]
title_roots = [
    'div[4]/div/div[1]',  # Politiccs
    'div[5]/div/div[1]',  # Economic
    'div[4]/div/div[1]',  # Social
    'div[4]/div/div[1]',  # Culture
    'div[4]/div/div[1]',  # World
    'div[4]/div/div[1]'   # IT
]

options = ChromeOptions()

options.add_argument('lang=ko_KR')

service = ChromeService(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

for outer_i in range(2):  # 준택님 2,4     민정님 4,6  # 준택님 2,4     민정님 4,6
    root_path = xpath_roots[outer_i]
    title_root = title_roots[outer_i]
    url = 'https://news.naver.com/section/10{}'.format(outer_i)
    driver.get(url)
    titles = []
    button_xpath = '//*[@id="newsct"]/{}'.format(root_path)
    for i in range(200):
        time.sleep(0.5)
        try:
            driver.find_element(By.XPATH, button_xpath).click()
        except:
            print(i)
    time.sleep(5)
    for i in range(1, 300):
        for j in range(1, 7):
            title_path = '//*[@id="newsct"]/{}/div[{}]/ul/li[{}]/div/div/div[2]/a/strong'.format(title_root, i, j)
            try:
                title = driver.find_element(By.XPATH, title_path).text
                titles.append(title)
                print(title)
            except:
                print('error', i, j)

    df_section_titles = pd.DataFrame(titles, columns=['titles'])
    df_section_titles['category'] = category[outer_i]
    df_section_titles.to_csv('./crawling_data/naver_headline_{}_{}.csv'.format(
        category[outer_i],
        datetime.datetime.now().strftime('%y%m%d')), index=False)
