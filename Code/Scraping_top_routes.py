from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--disable-dev-tools")
chrome_options.add_argument("--ignore-ssl-errors=yes")
chrome_options.add_argument("--ignore-certificate-errors")
driver = webdriver.Chrome(options=chrome_options)
driver.set_page_load_timeout(30)

driver.get("https://www.flightradar24.com/data/airports/tlv")

try:
    # Wait for up to 10 seconds for the element to be present
    top_routes = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "ul.top-routes"))
    )
except:
    print('Error: Top routes element not found on page.')
    driver.quit()
    exit()

routes = []
airport_codes = []
for route in top_routes.find_elements(By.TAG_NAME, "li"):
    link = route.find_element(By.TAG_NAME, "a")
    name = link.get_attribute("title")
    code = link.text.strip()
    flights_per_week = route.find_element(By.CSS_SELECTOR, "span.pull-right").text.strip()
    routes.append((name, code, flights_per_week))
    airport_codes.append(code)

driver.quit()
print(airport_codes)
