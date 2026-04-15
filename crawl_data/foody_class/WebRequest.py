import requests
from requests_html import AsyncHTMLSession
import urllib3.exceptions

get_timeout = 3
post_timeout = 3
async def get_raw_data(url: str, header: dict | None = None) -> str | None:
    """
    Create a get request and return the text response (Async)
    
    :param url: link to get data from
    :type url: str
    :param header: header for the request
    :type header: dict | None
    :return: The response text
    :rtype: str | None
    """
    try:
        asession = AsyncHTMLSession()
        response = await asession.get(url, timeout=get_timeout, headers=header)
        # response = requests.get(url, timeout=20)
        response.close()
        return response.text
    except (requests.exceptions.RequestException, urllib3.exceptions.ReadTimeoutError):
        return None
    
async def post_raw_data(url, payload, header = None) -> (str | None):
    try:
        asession = AsyncHTMLSession()
        response = await asession.post(url, json=payload, timeout=post_timeout, headers=header)
        response.close()
        return response.text
    except (requests.exceptions.RequestException, urllib3.exceptions.ReadTimeoutError):
        return None