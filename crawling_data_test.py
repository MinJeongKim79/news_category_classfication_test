import pandas as pd
import glob
import datetime

# 모든 headline CSV 파일 경로 가져오기
file_list = glob.glob('./crawling_data/naver_headline_*.csv')

# 합칠 데이터프레임
df_total = pd.DataFrame()

# 파일 하나씩 읽어서 합치기
for file in file_list:
    df = pd.read_csv(file)
    df_total = pd.concat([df_total, df], axis=0, ignore_index=True)

# 현재 날짜
today = datetime.datetime.now().strftime('%y%m%d')

# 저장
df_total.to_csv(f'./crawling_data/naver_headline_total_{today}.csv', index=False)

print(f'총 {len(df_total)}개 뉴스 제목을 저장했습니다.')