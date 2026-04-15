import asyncio
import pandas as pd
from typing import Any
import os
from foody_class.QueryClass import SearchQuery, BranchQuery, DetailQuery, OpeningHourQuery, MAX_TOTAL_REQUESTS, STATE_FOLDER
import shutil
import time
# Controlling parameters
to_do_flag = [] #["food"] #["food", "travel", "shop", "entertain", "service"]
sequence_to_do = ["branch", "detail", "opening hour"] # ['branch', 'detail','opening hour']
# Store all failed URL
fail_url = []
newly_failed_urls = []
# List of all urls that we may not want to include as it, either is an always fails urls 
# or just non exsisting urls (i.e. foody report not found)
ignore_urls = [] 

# Locations to crawl
locations = ['ho-chi-minh']
    #, 'ha-noi', 'da-nang', 'khanh-hoa', 'can-tho', 'vung-tau']

# Store all the data
data = pd.DataFrame()
categories_group = {}
cuisines = []
district = pd.DataFrame()

# Filename to save crawl data
data_filename = "foody_search_data.csv"
detail_filename = "foody_detail_data.csv"
branch_filename = "foody_branch_data.csv"
store_opening_hours_filename = "foody_store_opening_hours.csv"
# Filename where we save url we want to ignore
ignore_urls_filename = "foody_ignore_urls.txt"

# State and debug (and sanity check) logs filenames
branch_url_filename = STATE_FOLDER + "foody_branch_urls.txt"
details_url_filename = STATE_FOLDER + "foody_detail_urls.txt"
store_id_filename = STATE_FOLDER + "foody_store_ids.csv"
fail_url_filename = STATE_FOLDER + "foody_fail_urls.txt"
newly_failed_urls_filename = STATE_FOLDER + "foody_recently_failed_urls.txt"
save_fail_url_every_category = 2

# List of other urls / save value
branch_url = []
detail_url = []
store_ids = []
districts = pd.DataFrame()

def setup():
    # The category and location we want to crawl
    districts = pd.read_csv("foody_district.csv")
    cuisines = pd.read_csv("foody_cuisine.csv")['Id'].values.tolist()
    categories_group = pd.read_csv("foody_categories.csv")\
                        .groupby('CategoryGroup')
    categories_group = categories_group.apply(format_group, include_groups=False)\
                        .to_dict()
    if os.path.exists(ignore_urls_filename):
        with open(ignore_urls_filename, 'r') as f:
            ignore_urls = f.read().splitlines()
    else:
        with open(ignore_urls_filename, 'w'):
            pass
        ignore_urls = []
    return 0, districts, cuisines, categories_group, ignore_urls

def read_search_state_data():
    # Initial data
    start_location = locations[0]
    start_district = districts['Id'].values[0]
    processed_cuisine = []
    
    # Check if there is any thing to do in the search page
    # If not then we set the start category group to None
    if len(to_do_flag) > 0:
        start_type_of_category = to_do_flag[0]
        start_category_group = list(categories_group[start_type_of_category].keys())[0]
    else:
        start_type_of_category = ''
        start_category_group = None
        
    if os.path.exists(STATE_FOLDER + "search_crawling_state.txt"):
        with open(STATE_FOLDER + "search_crawling_state.txt", "r", encoding="utf-8") as f:
            # Get the previous location
            start_location = f.readline().strip()
            if start_location == "":
                start_location = locations[0]
            
            # Get the previous district id 
            start_district = f.readline().strip()
            if start_district == "" or start_district == "None":
                start_district = int(districts['Id'].values[0])
            else:
                start_district = int(start_district)
            
            # Get the processed cuisines (so that we can exclude those in our search) 
            processed_cuisine = f.readline().strip().split(' ')
            if processed_cuisine == [''] or processed_cuisine == '' or processed_cuisine == "None":
                processed_cuisine = []
            processed_cuisine = [int(cuisine) for cuisine in  processed_cuisine]
            
            # Get the category group to search 
            start_type_of_category = f.readline().strip()
            if start_type_of_category == "":
                if len(to_do_flag) > 0:
                    start_type_of_category = to_do_flag[0]
                else:
                    start_type_of_category = ''
            
            # Get the sub category in the category group
            start_category_group = f.readline().strip()
            if start_category_group == "":
                if start_type_of_category != '':
                    start_category_group = list(categories_group[start_type_of_category].keys())[0]
                else:
                    start_category_group = None
    return start_location, start_district, start_category_group, start_type_of_category, processed_cuisine
def format_group(x: Any) -> dict:
    return x.set_index('CategoriesName')['Id'].to_dict()

def save_fail_url(fail_urls_list):
    # Check if the newly fail urls are the same urls from the previous run
    # If so then that urls may need to be (on very rare occasion) ignore
    # This is log so that we can see it more clearly
    recent_failed_urls = [url for url in fail_urls_list if url in fail_url]
    newly_failed_urls.extend(recent_failed_urls)
    with open(newly_failed_urls_filename, "w") as f:
        for url in set(newly_failed_urls):
            f.write(url + "\n")
    # We need to save all fail so that when trouble arised, we have something to see.
    fail_url.extend(fail_urls_list) 
    
    with open(fail_url_filename, "w") as f:
        for url in set(fail_url):
            f.write(url + "\n")

def retrive_data_with_obj(class_obj_name, state_filename, data_filename, num_request_so_far, url_list, to_do_name):
    # Get the previous processed urls
    processed_url = []
    if os.path.exists(state_filename):
        with open(state_filename, "r", encoding="utf-8") as f:
            processed_url = f.read().splitlines()
            
    # Read the previously crawl data
    try:
        obj_data_frame = pd.read_csv(data_filename) if os.path.exists(data_filename) else pd.DataFrame()
    except (pd.errors.InvalidColumnName, pd.errors.EmptyDataError):
        obj_data_frame = pd.DataFrame()
    # Fail safe(ish)
    if os.path.exists(data_filename):
        shutil.move(data_filename, data_filename + ".backup")
    # Here class_obj_name is the name of the class that we want to use to crawl data (e.g. BranchQuery, DetailQuery, etc.) 
    # and it should be a subclass of CommonQuery
    actual_obj = class_obj_name(num_request_so_far = num_request_so_far, pre_data=obj_data_frame, processed_url=processed_url)
    
    if to_do_name not in sequence_to_do:
        print(f"{to_do_name} data is not crawled")
        actual_obj.write_data([], write_all=True)
        return obj_data_frame, num_request_so_far
    ignore_list = processed_url.copy()
    ignore_list.extend(ignore_urls)
    obj_url = (actual_obj.get_url(value=url) for url in set(url_list) if actual_obj.get_url(value=url) not in ignore_list)
    asyncio.run(actual_obj.get_all_data(obj_url, result_filename=data_filename, verbose=True, no_return=True))
    # actual_obj.write_data(obj_data)
    obj_data_frame = pd.concat([obj_data_frame, actual_obj.data], ignore_index=True)
    actual_obj.write_data([], write_all=True)
    if actual_obj.fail_urls:
        save_fail_url(actual_obj.fail_urls)
    return obj_data_frame, actual_obj.num_request_so_far

# The main crawling process
if __name__ == "__main__":
    # Inital setup for auto-reload state
    num_of_request, districts, cuisines, categories_group, ignore_urls = setup()
    
    start_location, start_district, start_category_group, start_type_of_category, processed_cuisine = read_search_state_data()
    if os.path.exists(data_filename):
        data = pd.read_csv(data_filename)
        data.drop_duplicates(subset='Id',inplace=True)
    
    # Backing up data so that if we screw up then it not the end of the world 
    # (unless we immediately run the program without fixinng the issue)
    if os.path.exists(data_filename):
        shutil.move(data_filename, data_filename + ".backup")
    # init state folder
    if not os.path.exists(STATE_FOLDER):
        os.makedirs(STATE_FOLDER) 
    
    if os.path.exists(STATE_FOLDER + "foody_fail_urls.txt"):
        with open(STATE_FOLDER + "foody_fail_urls.txt", "r") as f:
            fail_url = f.read().splitlines()
    else:
        fail_url = []

    # Intial location index to start
    if start_category_group is not None and len(to_do_flag) > 0:
        i_type_of_category = to_do_flag.index(start_type_of_category)
    else:
        i_type_of_category = 0
    
    i_location = locations.index(start_location)
    cumulative_fail_urls = []
    for location in locations[i_location:]:
    
        if len(to_do_flag) == 0: # If there is nothing to do then we can just skip
            continue

        for flag in to_do_flag[i_type_of_category:]:
            
            category_list = list(categories_group[flag])
            i_start_category_group = category_list.index(start_category_group) 
            c = 0
            # while i_start_category_group < len(category_list):
            for category in category_list[i_start_category_group:]:
                
                obj = SearchQuery(location, districts, cuisines, categories_group, pre_data=data, num_request_so_far=num_of_request)
                results = asyncio.run(obj.get_all_data(category=category, type_of_category=flag,verbose=True,
                                                        start_district=start_district, starting_cuisine=processed_cuisine, result_filename=data_filename))
                # Save all the results from category
                result_data = [pd.DataFrame(r) for r in results if r is not None]
                result_data.append(data)
                if len(result_data) > 1:
                    data = pd.concat(result_data)
                if data is not None:
                    data.drop_duplicates(subset="Id",inplace=True) # Duplicate is remove as it is possible to have the same store appear many times in the search result
                    data.to_csv(data_filename, index=False)
                
                # Update the store branch url
                
                branch_url.extend(list(url for url in set(obj.get_branch_url()) if url not in branch_url and url != "")) # Remove any duplicate
                
                # Saving any url that we fail to gather data from
                if obj.fail_urls:
                    cumulative_fail_urls.extend(obj.fail_urls)
                    if (c + 1) % save_fail_url_every_category == 0:
                        save_fail_url(cumulative_fail_urls)
                        cumulative_fail_urls.clear()
                
                # Update number of request (
                    #                       so that we can stop calling the query repeatedly 
                    #                       - though the query class does terminated if number_of_request exceeded MAX_TOTAL_REQUESTS
                    #                       but this save time so we use it here
                    #                       )    
                num_of_request += obj.num_request_so_far
                print(f"Number of request so far: {num_of_request}, obj {obj.num_request_so_far}")
                if num_of_request >= MAX_TOTAL_REQUESTS:
                    break
                c += 1
                # Reset starting conditions
                processed_cuisine = []
                start_district = 0
            i_start_category_group = 0
        i_type_of_category = 0
    
    # Get detail urls and backup all detail and branch urls
    if data is not None:
        detail_url = data["DetailUrl"].dropna().to_list()

    if len(to_do_flag) > 0 :
        with open(branch_url_filename, "w", encoding="utf-8") as f:
            for url in branch_url:
                f.write(url + "\n")
        with open(details_url_filename, "w", encoding="utf-8") as f:
            for url in detail_url:
                f.write(url + "\n")
    else:
        if os.path.exists(branch_url_filename):
            with open(branch_url_filename, "r", encoding="utf-8") as f:
                for url in f:
                    branch_url.append(url.strip())
        if os.path.exists(details_url_filename):
            with open(details_url_filename, "r", encoding="utf-8") as f:
                for url in f:
                    detail_url.append(url.strip())
    # Save all data when we finish searching
    if data is not None:
        data.drop_duplicates(subset="Id",inplace=True)
        data.to_csv(data_filename, index=False)

    del data # Remove the search data so that we don't used up all memory
    
    print("Getting branch data")
    data_t, n = retrive_data_with_obj(BranchQuery, STATE_FOLDER + "branch_crawling_state.txt", branch_filename, num_of_request, branch_url, "branch")
    num_of_request += n
    data_t.drop_duplicates(subset="Id", inplace=True)
    data_t.to_csv(branch_filename, index=False)
    
    print("Getting details data")
    if not data_t.empty:
        detail_url.extend(data_t["Url"].dropna().to_list())
        with open(details_url_filename, "w", encoding="utf-8") as f:
             for url in set(detail_url):
                f.write(url + "\n")
        del data_t
        detail_url = list(url for url in set(detail_url) if url != "") # Remove duplicate
        data_t, n = retrive_data_with_obj(DetailQuery, STATE_FOLDER + "detail_crawling_state.txt", detail_filename, num_of_request, detail_url, "detail")
        num_of_request += n
        data_t.drop_duplicates(subset="RestaurantID", inplace=True)
        data_t.to_csv(detail_filename, index=False)
    
    print("Getting opening hours")
    if not data_t.empty:
        res_id = data_t["RestaurantID"] 
        res_id = res_id.apply(lambda x: int(x) if x is not pd.NA else pd.NA).dropna()
        del data_t
        data_t, n = retrive_data_with_obj(OpeningHourQuery, STATE_FOLDER + "opening_hour_crawling_state.txt", 
                                        store_opening_hours_filename, num_of_request, res_id, "opening hour")
        num_of_request += n
        data_t.drop_duplicates(subset="Id", inplace=True)
        data_t.to_csv(store_opening_hours_filename, index=False)
    
    # print("List of failed URLs:")
    # for item in fail_url:
    #     print(item)
    save_fail_url([])
    print()
    print("List repeated fails URLs:")
    for item in newly_failed_urls:
        print(item)
    print()
    # Printing some info out so that we can determine if we are done or not and what to do next
    if 'branch' not in sequence_to_do:
        print("Branch data is not crawled")
    if 'detail' not in sequence_to_do:
        print("Detail data is not crawled")
    if 'opening hour' not in sequence_to_do:
        print("Opening hour data is not crawled")
    print()
    if fail_url:
        print("There are fail url that we cant get but you can re run the program")
        print(" at another time so that foody (hopefully) give us data") 
        print()
    if newly_failed_urls:
        print("There are repeated fail urls")
        print("You may want to check if there is any non existing url and add those to the ignore list.")
        print()
    if num_of_request < MAX_TOTAL_REQUESTS:
        print("Crawling finished")
        print("You can stop running the program now")
    else:
        print("Not done yet, please rerun at another time when foody isnt instantly block this ip or suspect this ip")
        print(f"Current time is: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        print("Recommend wait time is around 40 minutets to 1 hour")