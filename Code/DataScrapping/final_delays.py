from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pandas as pd
from selenium.webdriver.chrome.options import Options
from datetime import datetime

def create_driver():
    options = Options()
    options.headless = True
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    options.add_argument(f'user-agent={user_agent}')
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

def accept_cookies(driver):
    try:
        cookies_accept = driver.find_element(By.ID, "onetrust-accept-btn-handler")
        cookies_accept.click()
        time.sleep(2)
    except Exception as e:
        print("Cookies already accepted or not found:", e)

def get_weather_data(driver, airport_code):
    try:
        driver.get(f"https://www.flightradar24.com/data/airports/{airport_code}")
        time.sleep(10)
        temperature_element = driver.find_element(By.CSS_SELECTOR, ".weather-value.ng-binding")
        wind_element = driver.find_elements(By.CSS_SELECTOR, ".weather-value.ng-binding")[1]
        direction_element = driver.find_elements(By.CSS_SELECTOR, ".weather-value.ng-binding")[2]

        weather_arrival = temperature_element.text
        wind_arrival = wind_element.text
        direction_arrival = direction_element.text

        return weather_arrival, wind_arrival, direction_arrival
    except Exception as e:
        print(f"Could not retrieve weather data for {airport_code}: {e}")
        return "N/A", "N/A", "N/A"

def parse_date(date_str, is_row_date=False):
    try:
        if is_row_date:
            return datetime.strptime(date_str, '%d %b %Y').date()
        else:
            return datetime.strptime(date_str, '%A, %b %d').replace(year=datetime.now().year).date()
    except ValueError:
        return None

def get_delay_data(driver, flight_code, target_date, flight_time):
    try:
        driver.get(f"https://www.flightradar24.com/data/flights/{flight_code}")
        time.sleep(10)
        accept_cookies(driver)
        rows = driver.find_elements(By.CSS_SELECTOR, 'table#tbl-datatable tbody tr.data-row')
        for row in rows:
            columns = row.find_elements(By.TAG_NAME, 'td')
            if len(columns) > 0:
                 row_date = columns[2].text.strip()
                 scheduled_departures = columns[7].text.strip()
                 actual_departures = columns[8].text.strip()
                 row_time = columns[6].text.strip()
                 actual_arrival = columns[9].text.strip()

        # table = driver.find_element(By.ID, "tbl-datatable")
        # for row in table.find_elements(By.TAG_NAME, "tr"):
        #     cells = row.find_elements(By.TAG_NAME, "td")
        #     if len(cells) >= 2:  # Ensure there are enough cells
        #         temp = row.text.split(" ")
        #         print(temp)
        #         if len(temp) >= 20:
        #             row_date = temp[2] + " " + temp[3] + " " + temp[4]
        #             scheduled_departures = temp[12] + " " + temp[13]
        #             actual_departures = temp[14] + " " + temp[15]
        #             row_time = temp[16] + " " + temp[17][:2]
        #             actual_arrival = temp[18] + " " + temp[19]

            if len(row_time) == 4:
                row_time = "0" + row_time

            print(f"row_date: {row_date}, target_date: {target_date}")
            if row_date == target_date:
                return scheduled_departures, actual_departures, row_time, actual_arrival
        return None, None, None, None
    except Exception as e:
        print(f"Could not retrieve delay data for {flight_code}: {e}")
        return None, None, None, None

# Initialize the Chrome driver
driver = create_driver()

# Open the main airports page
driver.get("https://www.flightradar24.com/data/airports/india")
time.sleep(10)
accept_cookies(driver)

# Extract airport codes and names
airports = []
try:
    td_elements = driver.find_elements(By.CSS_SELECTOR, 'td[colspan="2"]')
    if not td_elements:
        print("Table rows not found, the table might not have loaded correctly.")
    else:
        for td in td_elements:
            try:
                a_tag = td.find_element(By.TAG_NAME, "a")
                code = a_tag.get_attribute("data-iata")
                name = a_tag.get_attribute("title")
                if code and name:
                    airports.append((code, name))
            except Exception as e:
                print("Error extracting data-iata or title:", e)
except Exception as e:
    print("Error finding table rows:", e)

print(airports)

# Close the main browser
driver.quit()

# Initialize a DataFrame to store all data
all_data = pd.DataFrame()
airport_code = airports[:10]
i = 0
# Loop through each airport code and scrape the data
for code, name in airports:
    driver = create_driver()
    driver.get("https://www.flightradar24.com/data/airports/" + code + "/arrivals")
    time.sleep(10)
    accept_cookies(driver)

    # Extract arrival weather data
    weather_arrival, wind_arrival, direction_arrival = get_weather_data(driver, code)

    # Load earlier flights by clicking the button
    while True:
        try:
            load_earlier_button = driver.find_element(By.CSS_SELECTOR, "button.btn-flights-load")
            load_earlier_button.click()
            time.sleep(2)
        except:
            break

    # Scrape the table
    try:
        table = driver.find_element(By.CSS_SELECTOR, "table.table-condensed.table-hover.data-table.m-n-t-15")
    except:
        print(f"Error: table element not found on page for airport code {code}.")
        driver.quit()
        continue

    data = []
    date_text = datetime.now().strftime('%A, %b %d')

    for row in table.find_elements(By.TAG_NAME, "tr"):
        if "row-date-separator" in row.get_attribute("class"):
            date_text = row.text.strip()
            continue

        if "Time" in row.text:
            continue

        cols = [col.text.strip() for col in row.find_elements(By.TAG_NAME, "td")]
        if len(cols) == 7:
            cols.append(date_text)  # Add the date to the row data
            cols.append(code)  # Add the airport code to the row data
            cols.append(name)  # Add the airport name to the row data
            cols.append(weather_arrival)  # Add the arrival weather to the row data
            cols.append(wind_arrival)  # Add the arrival wind speed to the row data
            cols.append(direction_arrival)  # Add the arrival wind direction to the row data
            data.append(cols)

    # Create a DataFrame for the current airport's data
    df = pd.DataFrame(data, columns=[
        'TIME', 'FLIGHT', 'FROM', 'AIRLINE', 'AIRCRAFT', '', 'STATUS', 'DATE', 
        'ARRIVAL_AIRPORT_CODE', 'ARRIVAL_AIRPORT_NAME', 'WEATHER_ARRIVAL', 'WIND_ARRIVAL', 'DIRECTION_ARRIVAL'
    ])
    
    # Append to the main DataFrame
    all_data = pd.concat([all_data, df], ignore_index=True)
    driver.quit()

# Initialize columns for departure weather data
all_data['WEATHER_DEPARTURE'] = "N/A"
all_data['WIND_DEPARTURE'] = "N/A"
all_data['DIRECTION_DEPARTURE'] = "N/A"
all_data['Scheduled_departures'] = None
all_data['Actual_departures'] = None
all_data['Scheduled_arrival'] = None
all_data['Actual_arrival'] = None

# Process each "FROM" airport for additional weather data and delay data
driver = create_driver()
accept_cookies(driver)

for index, row in all_data.iterrows():
    from_airport_text = row['FROM']
    from_airport_code = from_airport_text[from_airport_text.find('(')+1:from_airport_text.find(')')]
    
    # Get the FROM airport weather data
    weather_departure, wind_departure, direction_departure = get_weather_data(driver, from_airport_code)
    
    # Update the DataFrame with the weather data
    all_data.at[index, 'WEATHER_DEPARTURE'] = weather_departure
    all_data.at[index, 'WIND_DEPARTURE'] = wind_departure
    all_data.at[index, 'DIRECTION_DEPARTURE'] = direction_departure

    # Convert the date to match the format
    date_value = str(row['DATE'])
    dt = datetime.strptime(date_value, '%A, %b %d')
    dt = dt.replace(year=datetime.now().year)
    date_value_converted = dt.strftime('%d %b %Y')
    flight_time_value = str(row['TIME'])
    print(date_value_converted)
    # Get delay data
    code_value = str(row['FLIGHT'])
    scheduled_departure, actual_departure, row_time, actual_arrival = get_delay_data(driver, code_value, date_value_converted, flight_time_value)
    
    all_data.at[index, 'Scheduled_departures'] = scheduled_departure
    all_data.at[index, 'Actual_departures'] = actual_departure
    all_data.at[index, 'Scheduled_arrival'] = row_time
    all_data.at[index, 'Actual_arrival'] = actual_arrival

# Save the combined data to an Excel file
final_output_filename = r"C:\Users\kiran\OneDrive\Desktop\Files\Step2\Flight_Delay_Predection\Data\Scraped_data_final1.xlsx"
all_data.to_excel(final_output_filename, index=False)

print(all_data)
