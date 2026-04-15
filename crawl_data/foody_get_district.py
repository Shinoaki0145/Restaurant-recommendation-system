import requests
import json
import pandas as pd
import urllib3 
def get_district_data():
    district_data = {
        "Name": [],
        "AsciiName": [],
        "Id": []
    }
    url = "https://www.foody.vn/ho-chi-minh/sang-trong"
    try:
        raw_data = requests.get(url, timeout=10).text
        
        start = raw_data.index("jsonData") + len("jsonData = ")
        end = raw_data.index("};", start) + 1      
        
        json_data = json.loads(raw_data[start:end])
        district_data_raw = json_data.get('districts', {})
        for district in district_data_raw:
            district_data["Name"].append(district['Name'])
            district_data["AsciiName"].append(district["AsciiName"])
            district_data["Id"].append(district['Id'])
        
    except (requests.exceptions.RequestException, urllib3.exceptions.ReadTimeoutError):
        print("Timeout while getting district data")
        
    return district_data



if __name__ == "__main__":
    district_data = get_district_data()
    df = pd.DataFrame(district_data)
    df.to_csv('foody_district.csv', index=False, encoding='utf-8')