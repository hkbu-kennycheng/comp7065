import pandas as pd
import requests
import urllib.parse
from bs4 import BeautifulSoup
import concurrent.futures
from tqdm import tqdm

url = 'https://resource.data.one.gov.hk/td/traffic-detectors/rawSpeedVol-all.xml'
columns = ['from', 'to', 'detector_id', 'direction', 'lane_id', 'speed', 'occupancy', 'volume', 'sd']
def get_urls(url, start, end):
    u = f'https://api.data.gov.hk/v1/historical-archive/list-file-versions?url={urllib.parse.quote_plus(url)}&start={start}&end={end}'
    r = requests.get(u)
    return [
        f'https://api.data.gov.hk/v1/historical-archive/get-file?url={urllib.parse.quote_plus(url)}&time={t}'
        for t in r.json()['timestamps']
    ]


def get_df(u):
    r = requests.get(u)
    soup = BeautifulSoup(r.text, features="xml")
    date = soup.find('date').get_text()
    d = pd.DataFrame(columns=columns)
    for period in soup.find_all('period'):
        for detector in period.find_all('detector'):
            for lane in detector.find_all('lane'):
                period_from = pd.to_datetime(f'{date} {period.find("period_from").get_text()}')
                period_to = pd.to_datetime(f'{date} {period.find("period_to").get_text()}')
                row = pd.DataFrame([[
                    period_from,
                    period_to,
                    detector.find("detector_id").get_text(),
                    detector.find("direction").get_text(),
                    lane.find("lane_id").get_text(),
                    int(lane.find('speed').get_text()),
                    int(lane.find('occupancy').get_text()),
                    int(lane.find('volume').get_text()),
                    float(lane.find('s.d.').get_text())
                ]], columns=columns)
                d = pd.concat([d, row], ignore_index=True)
    return d


urls = get_urls(url, '20240122', '20240126')

df = pd.DataFrame(columns=columns)

with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
    future_to_url = {executor.submit(get_df, u): u for u in urls}
    for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(urls)):
        url = future_to_url[future]
        try:
            r = future.result()
        except Exception as exc:
            print('%r generated an exception: %s' % (url, exc))
            pass
        else:
            df = pd.concat([df, r], ignore_index=True)

df.to_csv('lab4-20240122-20240126.csv', index=False)