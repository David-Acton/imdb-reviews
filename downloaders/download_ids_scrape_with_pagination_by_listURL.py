

## this code is for scrapng the list from the urls until the last page(pagination)
##url ="https://www.imdb.com/search/title/?title_type=feature&languages=hi&view=simple&count=250&start=12000"

import requests
from bs4 import BeautifulSoup
from pandas import DataFrame
import uuid,sys,os


def  scrapelist(soup) :
    ids =[]
    taglist = soup.findAll('div',{"class" : "col-title" })
    for t in taglist :
        try :
            ids.append(t.find('a')['href'].split('/')[2])
            print(t.find('a')['href'].split('/')[2])
        except Exception as e:          
            print(e)
    your_list=ids
    df = DataFrame (your_list)
    unique_filename = str(uuid.uuid4())
    df.to_csv("pagesId/"+ unique_filename+".csv" ,index=False)        
    

def start_pagination(next_page) :
    imdb = "https://www.imdb.com"
    while next_page!="" :
            print(next_page)
            r = requests.get(url=next_page)
            soup = BeautifulSoup(r.text, 'html.parser')
            try: 
                  next_page = imdb + soup.find('a',{'class': 'lister-page-next next-page'})['href']
            except Exception as e: 
                print(e)
                next_page=""
            
            
            try :
                scrapelist(soup)
                print(next_page)
            except Exception as e: 
                print(e)
                break