import datetime
import time
from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()
driver.get("https://www.flightradar24.com/data/airports/hyd")
time.sleep(10)

try:
    cookies_accept = driver.find_element(By.ID, "onetrust-accept-btn-handler")
    cookies_accept.click()
    time.sleep(2)
except:
    print("Cookies already accepted or not found")

while True:
    try:
        load_earlier_button = driver.find_element(By.CSS_SELECTOR, "button.btn-flights-load")
        load_earlier_button.click()
        time.sleep(2)
    except:
        break

try:
    table = driver.find_element(By.CSS_SELECTOR, "table.table-condensed.table-hover.data-table.m-n-t-15")
except:
    print("Erroor: table element not found on page.")
    driver.quit()
    exit()

data,date,To,temperature_departures,wind_departures,direction_departures = [],[],[],[],[],[]

date_separators = table.find_elements(By.XPATH, "//tr[contains(@class, 'row-date-separator')]")
date_index = 0

for row in table.find_elements(By.TAG_NAME, "tr"):
    day = datetime.datetime.now().strftime("%A")
    if "row-date-separator hidden-xs hidden-sm " in row.get_attribute("class") and day in row.text:
        break
    if row in date_separators:
        date_text = row.text.strip()
        date_index += 1

    if "Time" in row.text:
        continue

    cols = [col.text.strip() for col in row.find_elements(By.TAG_NAME, "td")]
    if len(cols) == 7:
        data.append(cols)
        #date.append(date_text)

df = pd.DataFrame(data, columns=['TIME', 'FLIGHT', 'FROM', 'AIRLINE', 'AIRCRAFT', '','STATUS'])
# df['DATE'] = date
# df['TO'] = To

print(df)