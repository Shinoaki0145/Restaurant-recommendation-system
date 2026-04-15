import asyncio
import random
import json
import numpy
import pandas as pd
import math
from time import sleep
from typing import Any, Iterable

from foody_class.WebRequest import get_raw_data, post_raw_data
from foody_class.util import merge_dict
from foody_class.__init__ import BranchResult, SearchResult, StoreDetails

BASE_URL = "https://www.foody.vn"

MAX_RETRIES = 3
RETRY_DELAY = 1

MAX_NUMBER_OF_CONCURRENT_REQUESTS = 15

MAX_TOTAL_REQUESTS = 11000

WARNING_CONCURRENT_THREADHOLD = 50

STATE_FOLDER = "crawling_state/"

class CommonQuery:
    SAVE_STATE_EVERY = 30
    num_request_so_far: int = 0
    start_num_request: int = 0
    fail_urls: list | None = None
    MAX_PAGE_ALLOWED: int = 166
    CRAWLING_LIMIT: int = MAX_PAGE_ALLOWED * 12

    data: pd.DataFrame | None = None
    result_filename: str | None = None
    first_write = True
    id_label_name = None
    def __init__(self, num_request_so_far=0, **kwargs):
        self.start_num_request = num_request_so_far
        self.data = pd.DataFrame()
        if kwargs.get("pre_data", None) is not None:
            self.data = kwargs.get("pre_data")
        self.fail_urls = []
        if WARNING_CONCURRENT_THREADHOLD < MAX_NUMBER_OF_CONCURRENT_REQUESTS:
            print(f"""
                  Warning: The number of concurrent requests ({MAX_NUMBER_OF_CONCURRENT_REQUESTS}) exceeds the warning threshold ({WARNING_CONCURRENT_THREADHOLD}). 
                  Consider reducing it to avoid rate limit blocking.
                  """)
            sleep(2)  # Sleep for a short time to allow the user to read the warning

    
    def get_url(self, *args, **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement this method")
    
    def parse_response(self, raw_response):
        raise NotImplementedError("Subclasses must implement this method")
    
    async def get_data(self, *args, **kwargs) -> dict[str, Any] | None:
        raise NotImplementedError("Subclasses must implement this method")
    
    def write_data(self, data: dict | list[dict | None], *args, **kwargs):
        if self.data is not None:
            data_frames = []
            if isinstance(data, list):
                try:
                    data_frames.extend([pd.DataFrame(r) for r in data if r is not None])
                except ValueError as e:
                    i= -1
                    try:
                        for i, r in enumerate(data):
                            data_frames = pd.DataFrame(r) 
                    except ValueError as e2:
                        if i >= 0:
                            print(f"Error processing data at index {i}: {data[i]}")
                            print(data)
                    raise ValueError(e)
            else:
                if not pd.DataFrame(data).empty:
                    data_frames.append(pd.DataFrame(data))
            data_frames = [df for df in data_frames if not df.empty]
            tmp_df = pd.concat(data_frames) if data_frames else pd.DataFrame()
            try:
                old_ids_list = self.data[self.id_label_name].copy() if not self.data.empty else pd.Series()
            except Exception as e:
                print(e)
                print(len(self.data))
                print(self.data)
                print(self.data.columns)
                raise e
            old_ids_list = set(old_ids_list.to_list())
            if self.data.empty:
                self.data = tmp_df
            elif not tmp_df.empty:
                self.data = pd.concat([tmp_df, self.data], ignore_index=True)
            self.remove_duplicate()
            
            added_ids_list = (set(self.data[self.id_label_name]) - old_ids_list) if not self.data.empty else set()
            newly_added = self.data.where(self.data[self.id_label_name].isin(added_ids_list)).dropna(how='all') if not self.data.empty else pd.DataFrame()
            if self.result_filename and not self.data.empty and not newly_added.empty:
                if self.first_write:
                    # There is a problem where if self.data is empty then we will not write header to file
                    # And the subsequent append will not have header 
                    self.data.to_csv(kwargs.get("result_filename", self.result_filename), mode=kwargs.get("mode", "w"), index=False, header=kwargs.get("write_header", True))
                    self.first_write = False
                else:
                    newly_added.to_csv(kwargs.get("result_filename", self.result_filename), mode=kwargs.get("mode", "a"), index=False, header=False)
                if kwargs.get("write_all", False):
                    self.data.to_csv(kwargs.get("result_filename", self.result_filename), mode="w", index=False, header=kwargs.get("write_header", True))
                    
    async def get_all_data(self, list_urls: Iterable[str], *args, **kwargs) -> list[dict[str, Any]| None]:
        global MAX_TOTAL_REQUESTS
        if self.num_request_so_far + self.start_num_request > MAX_TOTAL_REQUESTS:
            print("Reached maximum number of requests per hour. Stopping.")
            return []
        results = []
        added = []
        nrso = False
        saved = True
        
        write_every = kwargs.get("write_every", 30)
        
        if kwargs.get("save_data", True):
            if kwargs.get("no_return", False):
                nrso = True
        else:
            saved = False
        
        tasks = []
        list_urls = iter(list_urls)
        url = next(list_urls, None)
        i = 0
        c = 0
        while url is not None:
            
            if self.num_request_so_far + self.start_num_request > MAX_TOTAL_REQUESTS:
                print("Reached maximum number of requests per hour. Stopping now.")
                break
            
            # url = self.get_url(url, *args, **kwargs)
            
            if len(tasks) < MAX_NUMBER_OF_CONCURRENT_REQUESTS:
                if kwargs.get("verbose", False):
                    print(f"Processing URL: {url}")
                tasks.append(asyncio.create_task(self.get_data(url, *args, **kwargs)))
                url = next(list_urls, None)
                i += 1
            else:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                
                for task in done:
                    data = task.result()
                    if data is not None:
                        added.append(data)
                tasks = list(pending)
                
            if (c + 1) % self.SAVE_STATE_EVERY == 0:
                self.save_state(*args, **kwargs)
                
            if saved and (c + 1) % write_every == 0:
                
                data_to_write = [d for d in added if d is not None]
                results.extend(added)
                self.write_data(data_to_write, *args, **kwargs)
                if nrso:
                    results = []
                added = []
            c += 1
                    
        # Wait for remaining tasks to complete
        while len(tasks) > 0:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            for task in done:
                data = task.result()
                if data is not None:
                    added.append(data)
            tasks = list(pending)
        results.extend(added)
        results = [d for d in results if d is not None]
        if saved and len(results) > 0:
            self.write_data(added, *args, **kwargs)
            if nrso:
                results = []
        # if self.num_request_so_far >= MAX_TOTAL_REQUESTS:
        self.save_state(*args, **kwargs)
        return results
    
    def save_state(self, *args, **kwargs):
        """Save the current state of the crawling process to a file"""
        pass
    
    def remove_duplicate(self):
        if self.data is not None:
            self.data.drop_duplicates(subset=self.id_label_name, inplace=True)

        
class SearchQuery(CommonQuery):
    result_filename = "foody_search_results.csv"
    max_page_crawl_per_category = 500000
    max_page_crawl_per_district = 500000
    processed_cuisines:list | None = None
    expected_cuisines = None
    receive_cuisines = None
    id_label_name = "Id"
    def __init__(self, location, districts, cuisines, categories, pre_data = None, num_request_so_far=0):
        self.location = location
        self.districts = districts
        self.cuisines = cuisines
        self.categories = categories
        if self.processed_cuisines is None:
            self.processed_cuisines = []
        if self.expected_cuisines is None:
            self.expected_cuisines = {}
        if self.receive_cuisines is None:
            self.receive_cuisines = {}
        super().__init__(num_request_so_far=num_request_so_far, pre_data=pre_data)
    
    def get_url(self, *args, **kwargs) -> str:
        """
        Get the search URL given the filter parameters
        
        :param location: the location put in the url (ho-chi-minh, ha-noi, ...)
        :param districts: the id of districts to filter
        :param local_cuisines: the id of local cuisines to filter
        :param type_of_category: the type of category to filter (food, travel, shop, entertain, service)
        :param category: the specific category within the category type
        :param page: which page number 
        :return: the constructed URL
        """
        district_str = kwargs.get("district", "")
        categories_group = self.categories
        type_of_category = kwargs.get("type_of_category", "")
        category_str = str(categories_group[type_of_category][kwargs.get("category", "")])
        # cuisine filter is has a special format in the url
        # and it can also be multi-valued (by passing is a string with all the cuisine id separated by comma)
        cuisine_str = ""
        if kwargs.get("local_cuisines"):
            cuisine_str = "-phong-cach-" + str(kwargs.get("local_cuisines"))
        location = self.location
        
        # This is mostly for future proofing
        page = kwargs.get("page", 1)
        
        # Number 4 here represents sort by the best rated shop
        # Can set to other number
        stype = kwargs.get("stype", 4)
        
        url = BASE_URL + f"/{location}/dia-diem{cuisine_str}?c={category_str}&dtids={district_str}&categorygroup={type_of_category}&page={page}&st={stype}&append=true"
        
        return url
    
    async def get_data(self, url, *args, **kwargs) -> dict[str, Any] | None:
        """
        Get the search data from the URL
        
        :param url: the URL to get data from
        :return: a tuple of (number of requests made, text data)
        """
       
        num_of_retry = 0
        json_data = None
       
        while num_of_retry <= MAX_RETRIES:
            await asyncio.sleep(RETRY_DELAY + random.uniform(0, (num_of_retry + 1) * RETRY_DELAY))
            # Check and update number of requests made so far
            self.num_request_so_far += 1
            if self.num_request_so_far + self.start_num_request > MAX_TOTAL_REQUESTS:
                return None
            raw_response = await get_raw_data(url)
           
            if raw_response is not None:
                json_data = self.parse_response(raw_response, no_skip_duplicate=kwargs.get("no_skip_duplicate", False))
                if json_data is not None:
                    break
            num_of_retry += 1
            
        if num_of_retry > MAX_RETRIES and json_data is None and self.fail_urls is not None:
            self.fail_urls.append(url)
            print(f"Failed to get data from {url} after {MAX_RETRIES} retries.")
        if json_data is not None and json_data != -1:
            return json_data
        return None
    
    def get_retrive_cuisines(self, raw_response):
        try:
            start = raw_response.index("{", raw_response.index("jsonData"))
            end = raw_response.index("};", start) + 1
            data = json.loads(raw_response[start:end])
        except ValueError:
            return None
        
        if self.receive_cuisines is not None and data.get("selectedCuisines", None):
            cuisine = int(data.get("selectedCuisines")[0].get("Id"))
            value = self.receive_cuisines.get(cuisine, 0)
            self.receive_cuisines.update({cuisine:value + 1})
            
    
    def parse_response(self, raw_response, **kwargs):
        """
        Parsing the raw HTML search data to get the JSON data
        
        :param raw_data: HTML text data
        :return: dictionary of object(s) of the search data
        """
        try:
            start = raw_response.index("{", raw_response.index("jsonData"))
            end = raw_response.index("};", start) + 1
            if start == -1 or end == -1:
                return None
            data = json.loads(raw_response[start:end])
        except ValueError:
            return None
        
        
        if data is None:
            return None
        self.get_retrive_cuisines(raw_response)
        if len(data["searchItems"]) == 0:
            return -1
        
        list_data = []
        
        no_skip_duplicate = kwargs.get("no_skip_duplicate", False)
        for search_result in data["searchItems"]:
            # Remove duplicate data
            if not no_skip_duplicate and self.data is not None and\
                len(self.data) > 0 and search_result.get("Id") in self.data["Id"].values:
                continue
            
            tmp = SearchResult()
            for name in SearchResult.__get_attribute__():
                tmp[name] = search_result.get(name)
            
            # We only need Name, DetailUrl, Id for cuisines so we do a custom parse for Cuisines field
            if tmp.Cuisines:
                cuisines = []
                for item in tmp.Cuisines:
                    cuisines.append({"Name": item.get("Name"), "DetailUrl": item.get("DetailUrl"), "Id":item.get("Id")})
                tmp.Cuisines = cuisines
            # We only need Name, Id for Services field so we do a custom parse for Services field
            if tmp.Services:
                service = []
                for item in tmp.Services:
                    service.append({"Name": item.get("Text"), "Id": item.get("Id")})
                tmp.Services = service
            # We only need Name, Id for Categories field so we do a custom parse for Categories field
            if tmp.Categories:
                categories = []
                for item in tmp.Categories:
                    categories.append({"Name": item.get("Name"), "Id": item.get("Id")})
                tmp.Categories = categories
            
            list_data.append(tmp)
        res = merge_dict(*list(map(lambda x: x.__dict__, list_data)))
        if res:
            res.update({"totalResult": data.get("totalResult")})
            return res
        return -1
        
    def save_state(self, *args, **kwargs):
        """Save the current state of the crawling process to a file"""
        for cuisine in self.cuisines:
            if  self.processed_cuisines is not None and self.receive_cuisines is not None and self.expected_cuisines and\
                    self.receive_cuisines.get(cuisine, 0) >= self.expected_cuisines.get(cuisine, 0) and\
                    cuisine not in self.processed_cuisines:
                        self.processed_cuisines.append(cuisine)
        
        current_district = kwargs.get("district", None)
        category = kwargs.get("category", "")
        with open(STATE_FOLDER + "search_crawling_state.txt", "w", encoding="utf-8") as f:
            f.write(f"{self.location}\n")
            f.write(f"{current_district}\n")
            if self.processed_cuisines is not None:
                for cuisine in self.processed_cuisines:
                    f.write(f"{cuisine} ")
            f.write("\n")
            # f.write(f"{self.processed_cuisines}\n")
            f.write(f"{kwargs.get('type_of_category', '')}\n")
            f.write(f"{category}\n")
            
    
    async def get_all_data(self, *args, **kwargs):
        """Crawl all data with a the given prarameter"""
        if self.num_request_so_far + self.start_num_request > MAX_TOTAL_REQUESTS:
            print("Maximum number of requests reached before crawling. Stopping.2")
            return []
        first_url = self.get_url(*args, **kwargs)
        
        first_json_data = await self.get_data(first_url, *args, **kwargs)
        total_json_data = []
        
        if first_json_data is not None and first_json_data != -1:
            # The total result exceed the crawling limit so we added districts
            
            value = first_json_data.get("totalResult")
            if isinstance(value, int) and int(value) > self.CRAWLING_LIMIT:
                processed_cuisine = kwargs.get("starting_cuisine", [])
                if self.processed_cuisines is not None:
                    self.processed_cuisines.extend(processed_cuisine)
                
                # Starting ditrict is an integer
                start_district = kwargs.get("start_district", 0)
                
                districts = self.districts['Id'].values.tolist()
                i_district = districts.index(start_district) if start_district != 0 else 0
                for district in districts[i_district:]:
                    result = await self.districts_filter_crawling(
                                *args,
                                district=district,
                                **kwargs
                            )
                    result = [r for r in result if r is not None and r != -1]
                    total_json_data.extend(result)
                    
                    if self.num_request_so_far + self.start_num_request >= MAX_TOTAL_REQUESTS:
                        print("Reached maximum number of requests per hour. Stopping now.3")
                        break
                    
                    if len(total_json_data) >= self.max_page_crawl_per_category:
                        print(f"Reach max page crawl per category limit at district {district} for category {kwargs.get('category')}")
                        print(f"We breaking here if not we are going to go insane")
                        break
                    self.save_state(*args, district=district, **kwargs)
                    self.processed_cuisines = []
                    self.expected_cuisines = {}
                    self.receive_cuisines = {}
                    
            else:
                if isinstance(value, int) and int(value) > 0:
                    page_count = math.ceil(int(value) / 12)
                    all_page_list = (self.get_url(*args, page=i, **kwargs) for i in range(2, page_count + 1))
                    total_json_data.append(first_json_data)
                    value = await super().get_all_data(all_page_list, *args, **kwargs)
                    total_json_data.extend(value)
                    self.save_state(*args, **kwargs)
            self.write_data(total_json_data, *args, **kwargs)
        if kwargs.get("verbose", False):
            print()
            print("================================")
            print()
            print (f"Finish crawling category {kwargs.get('category')} at location {self.location} with type {kwargs.get('type_of_category')}")
            print(f'Total result crawled: {len(total_json_data)}')
            print()
            print("================================")
            print()
        return total_json_data
    
    async def districts_filter_crawling(self, *args, **kwargs) -> list[dict[str, Any] | None]:
        """Getting the crawling data for a specific district"""
        if self.num_request_so_far + self.start_num_request >= MAX_TOTAL_REQUESTS:   
            return []
        global data
        district = kwargs.get("district", "")
        first_url = self.get_url(*args, **kwargs)
        json_data = await self.get_data(first_url, *args, **kwargs)
        
        total_json_data = []
        
        if json_data is not None and json_data != -1:
            # The total result exceed the crawling limit so we added cuisines
            value = json_data.get("totalResult")
            if self.num_request_so_far + self.start_num_request < MAX_TOTAL_REQUESTS and isinstance(value, int) and int(value) > self.CRAWLING_LIMIT:
            
                
                # First determine which cuinse to group or to do a normal crawl
                # By checking the first page to find the total result need to crawl
                if self.processed_cuisines is not None:
                    filtered_cuisines = list(cuisine for cuisine in self.cuisines if cuisine not in self.processed_cuisines)
                else:
                    filtered_cuisines = list(cuisine for cuisine in self.cuisines)
                urls = list(
                    self.get_url(
                        local_cuisines=c, 
                        page=1, 
                        **kwargs) 
                    for c in filtered_cuisines
                    )
                
                first_page_data = []
                tasks = []
                i_url = 0
                # Here we use a new kwargs to set make sure we get the first page regrardless 
                # we have retrived it or not because we need the total page to determine whether we need to group it or not
                new_kwargs = kwargs.copy()
                new_kwargs["no_skip_duplicate"] = True
                
                # Since we need to match the total number of page to each cuisine so we do a manual crawl
                # as get_all_data method in the base class (CommonQuery) return a list of unorder value 
                # (i.e. the list order does not match with the order of the generated urls).
                
                while i_url < len(urls):
                # for url in urls:
                    if len(tasks) < MAX_NUMBER_OF_CONCURRENT_REQUESTS:
                        url = urls[i_url]
                        tasks.append(asyncio.create_task(self.get_data(url, *args, **new_kwargs)))
                        i_url += 1
                    else:
                        # Can't use asyncio.wait as that does not given any* info about what the cuisine is
                        # *: Technically, there is a way to (somewhat) get the cuisine 
                        #    (by using tasks.index(<done task>) and an additional array to hold the cuisine id)
                        #    but it may be complicated to do and the current implement is good enough for 55 cuisines  
                        
                        result = await tasks[0]
                        first_page_data.append(result)
                        done_task = tasks[0]
                        tasks.remove(done_task)
                # Handle the all unreceive data
                while len(tasks) > 0:
                    result = await tasks[0]
                    first_page_data.append(result)
                    done_task = tasks[0]
                    tasks.remove(done_task)
                pages_to_crawl = []
                pages_to_group = []
                
                for c, r in zip(filtered_cuisines, first_page_data):      
                    total_page = 0             
                    if r is not None and r != -1:
                        value = r.get("totalResult")
                        
                        if isinstance(value, int) and int(value) > 0:
                            total_json_data.append(r)
                            total_page = math.ceil(int(value) / 12)

                            if total_page > MAX_NUMBER_OF_CONCURRENT_REQUESTS:
                                pages_to_crawl.append({"cuisines": c, "total_page": total_page})
                            else:
                                if total_page > 1:
                                    pages_to_group.append({"cuisines": c, "total_page": total_page})
                                    
                    if self.expected_cuisines is not None:
                        self.expected_cuisines.update({c:total_page})

                self.save_state(*args, **kwargs)
                if self.num_request_so_far >= MAX_TOTAL_REQUESTS:
                    self.write_data(total_json_data, *args, **kwargs)
                    return total_json_data
                
                def all_page_category_gen(list_crawl):
                    for c in list_crawl:
                        for i in range(2, min(c["total_page"] + 1, self.MAX_PAGE_ALLOWED + 1)):
                            if self.num_request_so_far + self.start_num_request > self.max_page_crawl_per_district and \
                                self.num_request_so_far + self.start_num_request > self.max_page_crawl_per_category:
                                break
                            yield self.get_url(
                                local_cuisines=c["cuisines"],
                                page=i,
                                **kwargs)
                        if self.num_request_so_far + self.start_num_request > self.max_page_crawl_per_district and \
                            self.num_request_so_far + self.start_num_request> self.max_page_crawl_per_category:
                            if kwargs.get("verbose", False):
                                    print("Reached maximum number of requests per hour. Stopping.1")
                            break
                # Group small pages together to better parrallelize
                urls = all_page_category_gen(pages_to_group)
                
                group_result = await super().get_all_data(urls, *args, **kwargs)
                total_json_data.extend(group_result)
                self.save_state(*args, **kwargs)
                # Process pages that have many page
                urls = all_page_category_gen(pages_to_crawl)
                
                data = await super().get_all_data(urls, *args, **kwargs)
                
                total_json_data.extend(data)
                self.save_state(*args, **kwargs)
                print(f"District {district} has {len(total_json_data)} results")
                
            else:
                # Total result is within limit so we can crawl normally
                value = json_data.get("totalResult")
                if isinstance(value, int) and int(value) > 0:
                    page_count = math.ceil(int(value) / 12)
                    all_page_list = (
                        self.get_url(
                            page=i,
                            **kwargs) 
                        for i in range(2, page_count + 1)
                    )
                    total_json_data.append(json_data)
                    data = await super().get_all_data(all_page_list, *args, **kwargs)
                    total_json_data.extend(data)
            
            self.write_data(total_json_data, *args, **kwargs)
        return total_json_data
    
    def get_branch_url(self):
        branch_urls = []
        if self.data is not None and not self.data.empty:
            search_space = self.data[self.data["BranchUrl"].notna() & self.data["BranchUrl"].str.strip().ne("")]
            for item in search_space["BranchUrl"].unique():
                branch_urls.append(item)
        return branch_urls

    
class BranchQuery(CommonQuery):
    result_filename = "foody_branch_data.csv"
    processed_branch_url: list | None = None    
    id_label_name = "Id"
    def __init__(self, pre_data=None, num_request_so_far=0, processed_url=None):
        if self.processed_branch_url is None:
            self.processed_branch_url = []
        if processed_url is not None:
            self.processed_branch_url = processed_url
        super().__init__(num_request_so_far=num_request_so_far, pre_data=pre_data)
    
    def get_url(self, *args, **kwargs) -> str:
        """
        Get the branch URL given the filter parameters
        
        :param url: the url to get data from
        :return: the constructed URL
        """
        url = kwargs.get("value", "")
        return BASE_URL + url
    def parse_init_data_response(self, raw_response):
        try:
            start = raw_response.index("{", raw_response.index("initDataRes"))
            end = raw_response.index("};", start) + 1
            json_data_res = json.loads(raw_response[start:end])
        except (ValueError, json.decoder.JSONDecodeError):
            return None
        
        return json_data_res
    
    def parse_res_brand_response(self, raw_response):
        try:
            start = raw_response.index('{', raw_response.index("initDataBrand"))
            start = raw_response.index("'", raw_response.index("Brand:{Id:", start))
            end = raw_response.index("'", start + 1)
        except (ValueError):
            return None
        return raw_response[start+1:end] 
    
    async def get_data(self, url, *args, **kwargs) ->  dict[str, Any] | None:
        """
        Get the branch data from the URL
        
        :param url: the URL to get data from
        """
        
        num_of_retry = 0
        json_data = None
        actual_data = None
        post_url = None
        raw_response = None
        total_branch_count = 0

        while num_of_retry <= MAX_RETRIES:
            
            await asyncio.sleep(RETRY_DELAY + random.uniform(0, (num_of_retry + 1) * RETRY_DELAY))
            self.num_request_so_far += 1
            if self.num_request_so_far + self.start_num_request > MAX_TOTAL_REQUESTS:
                return None
            if post_url is None:
                # We need to get the branch data via get request because there is no brand id when searching
                raw_response = await get_raw_data(url)
                if raw_response is not None:
                    json_data = self.parse_init_data_response(raw_response)
                    
                if json_data is not None:
                    # There may be more than the max number of branches that we can get with a get request
                    # so we need to check if there are any other branches
                    if json_data.get("Total") > json_data.get("Count"):
                        # To get more we can use get request to get the next chunk of brances 
                        # But we can do better by using the website API (post request) to get the branch data faster
                        # 
                        # Why does this exist? Ans: Foody is a funny place
                        
                        brand_data = self.parse_res_brand_response(raw_response)
                        if brand_data is not None:
                            # Brand data is the brand id
                            num_of_retry = -1
                            total_branch_count = json_data['Total']
                            location = json_data.get("Items")[0].get("LocationUrlRewriteName")
                            try:
                                post_url = f"https://www.foody.vn/__post/LandingPage/ListResByBrandCityUrlRewrite?brandId={brand_data}&c={location}&sortType=null"
                            except ValueError as e:
                                print(f"Error parsing brand id from {brand_data}: {e}")
                                raise ValueError(e)
                    else:
                        actual_data = self.parse_response(json_data)
            else:
                # Why does paging works when the other api doesn't? Who knows :)
                raw_response = await post_raw_data(post_url, {'Count':total_branch_count}, {"X-Requested-With":"XMLHttpRequest"})
                # print(post_url, raw_response)
                if raw_response is not None:
                    try:
                        json_data = json.loads(raw_response)
                        actual_data = self.parse_response(json_data)
                    except json.decoder.JSONDecodeError:
                        actual_data = None
            
            if actual_data is not None:
                if self.processed_branch_url is not None and url not in self.processed_branch_url:
                    self.processed_branch_url.append(url)
                break
            num_of_retry += 1
            
        if num_of_retry > MAX_RETRIES and actual_data is None:
            print(f"Failed to get data from {url} after {MAX_RETRIES} retries.")
            if post_url is not None:
                print(post_url)
            if self.fail_urls is not None:
                self.fail_urls.append(url)
        return actual_data
        
    def parse_response(self, raw_response):

        """Parsing the branch data from JSON to a dictionary"""
        list_data = []
        
        for data in raw_response["Items"]:
            tmp = BranchResult()
            for name in BranchResult.__get_attribute__():
                tmp[name] = data.get(name)
                
            # We only need Name, Id for Services field so we do a custom parse for Services field
            if tmp.Services:
                service = []
                for item in tmp.Services:
                    service.append({"Name": item.get("Text"), "Id": item.get("Id")})
                tmp.Services = service
            
            list_data.append(tmp.__dict__)
        return merge_dict(*list_data)
    
    def save_state(self, *args, **kwargs):
        """Save the current state of the crawling process to a file"""
        with open(STATE_FOLDER + "branch_crawling_state.txt", "w", encoding="utf-8") as f:
            if self.processed_branch_url is not None:
                for url in self.processed_branch_url:
                    f.write(f"{url}\n")

            
class DetailQuery(CommonQuery):
    result_filename = "foody_detail_data.csv"
    processed_detail_url: list | None = None
    open_time_data: list | None = None
    id_label_name = "RestaurantID"
    def __init__(self, num_request_so_far=0, pre_data=None, processed_url=None):
        if self.processed_detail_url is None:
            self.processed_detail_url = []
        if self.open_time_data is None:
            self.open_time_data = []
        if processed_url is not None:
            self.processed_detail_url = processed_url
        super().__init__(num_request_so_far=num_request_so_far, pre_data=pre_data)
    def get_url(self, *args, **kwargs) -> str:
        """
        Get the detail URL given the filter parameters
        
        :param url: the url to get data from
        :return: the constructed URL
        """
        url = kwargs.get("value", "")
        return BASE_URL + url
    
    def get_store_ratings(self, raw_data: str):
        # Getting store ratings manually from raw HTML data
        level_strs = {
            'excelent' : '<b class="exellent">',
            'good': '<b class="good">',
            'average' :'<b class="average">',
            'bad' : '<b class="bad">'
        }
        result = {
            "excelent": 0,
            "good": 0,
            "average": 0,
            "bad": 0
        }
        
        try:
            start = raw_data.index('<div class="ratings-boxes">')
            end = raw_data.index('<div class="ratings-boxes-points">', start)
            for level in level_strs:
                start_level = raw_data.index(level_strs[level], start, end) + len(level_strs[level])
                end_level = raw_data.index("</b>", start_level)
                result[level] = int(raw_data[start_level:end_level].strip())
        except ValueError:
            pass
        return result
    
    def get_misc_info(self, raw_response):
        misc_info = []
        try:
            start_doc = raw_response.index('<div class="microsite-res-info">')
            start_doc = raw_response.index('<div class="new-detail-info-sec">', start_doc) + len('<div class="new-detail-info-sec">')
            end_doc = raw_response.index('<div class="microsite-res-info-properties">', start_doc)
            start_block = start_doc
            while start_block < end_doc:
                start_block = raw_response.index('<div class="new-detail-info-area">', start_block) + len('<div class="new-detail-info-area">')
                if start_block == -1 or start_block >= end_doc:
                    break
                start_label = raw_response.index('<div class="new-detail-info-label">', start_block) + len('<div class="new-detail-info-label">')
                end_label = raw_response.index("</div>", start_label)
                
                label = raw_response[start_label:end_label].strip()
                
                # Value is not well define so when we need to use it we need to parse it properly
                start_value = raw_response.index('<div>', end_label) + len('<div>')
                end_value = raw_response.index("</div>", start_value)
                value = raw_response[start_value:end_value].strip()
                misc_info.append({"Label": label, "Value": value})
        except ValueError:
            pass
        
        return misc_info
    def get_properties(self, raw_response):
        results = []
        num_of_property = 31
        try:
            start_doc = raw_response.index('<div class=\"microsite-res-info-properties\">')
            start_doc = raw_response.index("<ul ", start_doc) + len("<ul ")
            end_doc = raw_response.index("</ul>", start_doc)
            start = start_doc
            
            for _ in range(num_of_property):
                start = raw_response.index("<li>", start) + len("<li>")
                if start == -1 or start >= end_doc:
                    break
                start = raw_response.index("<a", start) + len("<a")
                start = raw_response.index(">", start) + len(">")
                end = raw_response.index("</a>", start)
                results.append({"Name": raw_response[start:end].strip(), "PropertyID": -1})
                start = end + len("</a>")
        except ValueError:
            pass
        return results
    
    def parse_response(self, raw_response):
        """Parsing the detail data from raw HTML to a dictionary"""
        try:
            start = raw_response.index('{', raw_response.index("initData = "))
            end = raw_response.index("};", start) + 1
            
            raw_json_data = json.loads(raw_response[start:end])
        except ValueError:
            return None
        if len(raw_json_data) > 0:
            tmp = StoreDetails()
            for name in StoreDetails.__get_attribute__():
                tmp[name] = raw_json_data.get(name, None)
                
            # Custom parse for PictureModel    
            if tmp.PictureModel is not None and not isinstance(tmp.PictureModel, tuple):
                tmp.PictureModel = {"imageUrl": tmp.PictureModel.get("ImageUrl"), "title": tmp.PictureModel.get("Title")}
                
            # Custom parse for Services
            if tmp.Services is not None:
                service = []
                for item in tmp.Services:
                    service.append({"Name": item.get("Text"), "Id": item.get("Id")})
                tmp.Services = service
                
            # Custom parse for Cuisines
            if tmp.AvgPointList is not None:
                points = []
                for item in tmp.AvgPointList:
                    points.append({"Label": item.get("Label"), "Point": item.get("Point")})
                tmp.AvgPointList = points
                
            # Custom parse for LstTargetAudience
            if tmp.LstTargetAudience is not None:
                names = []
                for item in tmp.LstTargetAudience :
                    names.append({"Name": item.get("Name")})
                tmp.LstTargetAudience  = names
                
            # Custom parse for LstCategory
            if tmp.LstCategory is not None:
                categories = []
                for item in tmp.LstCategory:
                    categories.append({"Name": item.get("Name"), "Id": item.get("Id"), "AsciiName":item.get("AsciiName")})
                tmp.LstCategory = categories
            
            if tmp.Properties is not None:
                properties = []
                for item in tmp.Properties:
                    properties.append({"Name": item.get("Name"), "PropertyID": item.get("PropertyID")})
                tmp.Properties = properties
            
            # Getting Ratings
            tmp.Ratings = self.get_store_ratings(raw_response)
            
            if not tmp.Properties:
                tmp.Properties = self.get_properties(raw_response)
            dict_result = tmp.__dict__
            dict_result["MiscInfo"] = self.get_misc_info(raw_response)
            return merge_dict(dict_result, StoreDetails().__dict__)

        return None
    
    async def get_data(self, url, *args, **kwargs) -> dict[str, Any] | None:
        """Getting the store details from a given URL"""
        # print("Getting detail data", url)
        retry_count = 0
        json_data = None
        while retry_count < MAX_RETRIES:
            await asyncio.sleep(1.25 * RETRY_DELAY + random.uniform(0, RETRY_DELAY* (retry_count + 1))) # Avoid too many requests at the same time
            self.num_request_so_far += 1
            if self.num_request_so_far + self.start_num_request > MAX_TOTAL_REQUESTS:
                return None
            raw_data = await get_raw_data(url)
            # print(1,raw_data)
            
            if raw_data:
                data = self.parse_response(raw_data)
                if data is not None:
                    if self.processed_detail_url is not None and url not in self.processed_detail_url:
                        self.processed_detail_url.append(url)
                        return data
                       
            retry_count += 1
        if self.fail_urls is not None and json_data is None and retry_count >= MAX_RETRIES:
            self.fail_urls.append(url)
            print(f"Fail to get store detail at {url}")
        return None
    
    def save_state(self, *args, **kwargs):
        """Save the current state of the crawling process to a file"""
        with open(STATE_FOLDER + "detail_crawling_state.txt", "w", encoding="utf-8") as f:
            if self.processed_detail_url is not None:
                for url in self.processed_detail_url:
                    f.write(f"{url}\n")
                  
class OpeningHourQuery(CommonQuery):
    result_filename = "foody_opening_hour_data.csv"
    processed_res_id: list[str] | None = None
    id_label_name = "Id"
    def __init__(self, num_request_so_far=0, pre_data=None, processed_url=None):
        if self.processed_res_id is None:
            self.processed_res_id = []
        if processed_url is not None:
            self.processed_res_id = processed_url
        super().__init__(num_request_so_far=num_request_so_far, pre_data=pre_data)
    
    def get_url(self, *args, **kwargs) -> str:
        """
        Get the opening hour URL given the filter parameters
        
        :param url: the url to get data from
        :return: the constructed URL
        """
        res_id = kwargs.get("value", "")
        return BASE_URL + f"/__get/Restaurant/GetOpeningTime?resId={res_id}"
    
    async def get_data(self, url, *args, **kwargs) -> dict[str, Any] | None:
        retry_count = 0
        
        res_id = url[url.index("resId=") + len("resId="):]
        json_data = None
        while retry_count < MAX_RETRIES:
            await asyncio.sleep(RETRY_DELAY + random.uniform(0, RETRY_DELAY * (retry_count + 1) )) # To avoid too many requests at the same time
            self.num_request_so_far += 1
            if self.num_request_so_far + self.start_num_request >= MAX_TOTAL_REQUESTS:
                return None
            raw_data = await get_raw_data(url, {"X-Requested-With":"XMLHttpRequest"})
            
            if raw_data is not None:
                json_data = self.parse_response(raw_data)
                if json_data is not None:
                    json_data.update({"Id": [res_id]})
                    if self.processed_res_id is not None and res_id not in self.processed_res_id:
                        self.processed_res_id.append(res_id)
            
            if json_data:
                break
            retry_count += 1
        if retry_count >= MAX_RETRIES and json_data is None:
            print(f"Failed to get opening hour data for restaurant id {res_id} after {MAX_RETRIES} retries.")
            if self.fail_urls is not None:
                self.fail_urls.append(url)
        return json_data
    
    def parse_response(self, raw_response):
        try: 
            json_data = json.loads(raw_response)
        except json.decoder.JSONDecodeError:
            return None
        
        result = {}
        for day_info in json_data['Items']:
            day = day_info.get('DayOfWeek')
            is_day_off = day_info.get("IsDayOff")
            is_time_off = day_info.get("TimeOffs")
            times = []
            for t in day_info.get('Times'):
                open_hours = t.get('TimeOpen')
                closing_hours = t.get('TimeClose')
                title = t.get('Title')
                
                times.append({"open": open_hours, "close": closing_hours, "Title": title})
            if len(times) == 0:
                times.append({"open": None, "close": None, "Title": None})
            result[day] = [{"time": times, "is_day_off": is_day_off, "is_time_off": is_time_off}]
        return result

    def save_state(self, *args, **kwargs):
        if self.processed_res_id is not None:
            with open(STATE_FOLDER + "opening_hour_crawling_state.txt", "w", encoding="utf-8") as f:
                for res_id in self.processed_res_id:
                    f.write(f"{res_id}\n")
