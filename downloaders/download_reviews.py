import requests
from bs4 import BeautifulSoup
from flask import jsonify
import json
import os


def scrape_reviews(soup, ImdbId):
    try:
        reviews = soup.find_all('div', {'class': 'imdb-user-review'})
    except:
        pass

    reviews_text = []
    for review in reviews:
        review_imdb = {}

        ################
        try:
            review_imdb['reviewer_name'] = review.find(
                'span', {'class': 'display-name-link'}).find('a').get_text().strip()
        except:
            review_imdb['reviewer_name'] = ""
        ###############
        try:
            review_imdb['reviewer_url'] = review.find(
                'span', {'class': 'display-name-link'}).find('a')['href']
        except:
            review_imdb['reviewer_url'] = ""
        ############
        try:
            review_imdb['data-review-id'] = review['data-review-id']
        except:
            review_imdb['data-review-id'] = ""

        #############
        try:
            short_review = review.find('a', {'class': 'title'})
            text = short_review.get_text().strip()
            review_imdb['short_review'] = text
        except:
            review_imdb['short_review'] = ""

        ######################
        try:
            full_review = review.find('div', {'class': 'show-more__control'})
            text = full_review.get_text().strip()
            review_imdb['full_review'] = text
        except:
            review_imdb['full_review'] = ""
        #############
        try:
            review_date = review.find('span', {'class': 'review-date'})
            text = review_date.get_text().strip()
            review_imdb['review_date'] = text
        except:
            review_imdb['review_date'] = ""
        #######
        try:
            ratings_span = review.find(
                'span', {'class': 'rating-other-user-rating'})
            text = ratings_span.find('span').get_text().strip()
            review_imdb['rating_value'] = text
        except:
            review_imdb['rating_value'] = ""
        ##########
        reviews_text.append(review_imdb)

    return reviews_text


def scrape(movie_url, ImdbId, all_data):
    r = requests.get(url=movie_url)
    soup = BeautifulSoup(r.text, 'html.parser')

    all_data = []
    data = scrape_reviews(soup, ImdbId)
    all_data.extend(data)

    try:
        pagination_key = soup.find(
            'div', {'class': 'load-more-data'})['data-key']
        movie_url = "https://www.imdb.com/title/"+ImdbId + \
            "/reviews/_ajax?&paginationKey="+pagination_key
        scrape(movie_url, ImdbId, all_data)
    except Exception as e:
        print("scraping done successfully")
    return all_data


def get_reviews(ImdbId):
    movie_url = "https://www.imdb.com/title/"+ImdbId+"/reviews/_ajax?"
    data = scrape(movie_url, ImdbId, [])
    return data
