import requests
import pandas as pd
import json

dict_data = {
    "Id": [],
    "Name": [],
    "AsciiName": [],
    "UrlRewriteName": []
}
if __name__ == "__main__":
    url = "https://www.foody.vn/__get/Directory/GetSearchFilter?q=&filter=cuisine&provinceId=217"
    
    try:
        response = requests.get(url, timeout=5, headers={"X-Requested-With":"XMLHttpRequest"}).text
        
        json_data = json.loads(response)
        for data in json_data["allCuisines"]:
            
            dict_data["Name"].append(data["Name"])
            dict_data["AsciiName"].append(data["AsciiName"])
            dict_data["Id"].append(data["Id"])
            dict_data["UrlRewriteName"].append(data["UrlRewriteName"])
    except json.JSONDecodeError as e:
        print(f"Fail to call api: {e}")
    df = pd.DataFrame(dict_data)
    df.to_csv("foody_cuisine.csv", index=False)
        