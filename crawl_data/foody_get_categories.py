import requests
import pandas as pd
import json
dict_data = {
    "CategoryGroup": [],
    "CategoriesName": [],
    "CategoriesAsciiName": [],
    "Id": []
}
if __name__ == "__main__":
    categories = ["food", "travel", "shop", "entertain", "service"]
    
    for category in categories:
        try:
            cookie = {"gcat" : category}
            response = requests.get("https://www.foody.vn/__get/Directory/GetSearchFilter?q=&filter=category&provinceId=217", 
                                    headers={"X-Requested-With": "XMLHttpRequest"}, cookies=cookie)
            json_data = json.loads(response.text)
            for item in json_data["allCategories"]:
                dict_data["CategoriesName"].append(item["Name"])
                dict_data["CategoriesAsciiName"].append(item["AsciiName"])
                dict_data["Id"].append(item["Id"])
                dict_data["CategoryGroup"].append(category)
            
        except requests.exceptions.Timeout or requests.exceptions.ReadTimeout:
            print(f"Fail to get category {category}")
        df = pd.DataFrame(dict_data)
        df.to_csv("foody_categories.csv", index=False)