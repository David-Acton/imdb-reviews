import requests
from bs4 import BeautifulSoup


def get_reviews(id, max_pages=50):
    '''Get the reviews for an IMDB film or series'''

    # Request the HTML from the URL provided.
    url = "https://www.imdb.com/title/"+id+"/reviews/_ajax?"
    request = requests.get(url=url)

    # Set up our HTML parser with the HTML content from the page.
    page = BeautifulSoup(request.text, 'html.parser')

    # Initally get the reviews, before we start using 'load more'.
    reviews = scrape_reviews(page, [])

    # Recursion is not real, contact your local computer scientist.
    page_index = 0
    while page_index < max_pages:
        # Get the pagination key of the next page.
        pagination_key = page.find('div', {'class': 'load-more-data'})

        # Check if the pagination exists.
        if pagination_key == None:
            # This key does not exist, we must be at the end.
            break
        else:
            # Request the HTML from the URL provided for the next page.
            url = "https://www.imdb.com/title/" + id + \
                "/reviews/_ajax?&paginationKey=" + pagination_key['data-key']
            request = requests.get(url=url)

            # Set up our HTML parser with the HTML content from the page.
            page = BeautifulSoup(request.text, 'html.parser')

            # Get the next set of reviews.
            reviews = scrape_reviews(page, reviews)
            page_index += 1

    # Return with all the reviews.
    return reviews


def scrape_reviews(page, reviews):
    '''Scrape all the reviews from the given page and array.'''

    # Find all the reviews on the page with the class 'imdb-user-review'.
    html_reviews = page.find_all('div', {'class': 'imdb-user-review'})

    # Loop through all the reviews found on the page.
    for review in html_reviews:
        # Set up a new dictionary to represent the review.
        review_imdb = {}

        # Get the reviewer's name from the review with the class 'display-name-link'.
        try:
            review_imdb['reviewer_name'] = review.find(
                'span', {'class': 'display-name-link'}).find('a').get_text().strip()
        except:
            review_imdb['reviewer_name'] = "NA"  # Failed to get any output

        # Get the reviewer's URL from the review with the class 'display-name-link'.
        try:
            review_imdb['reviewer_url'] = review.find(
                'span', {'class': 'display-name-link'}).find('a')['href']
        except:
            review_imdb['reviewer_url'] = "NA"  # Failed to get any output

        # Get the review ID from the review with the datatag 'data-review-id'.
        try:
            review_imdb['data-review-id'] = review['data-review-id']
        except:
            review_imdb['data-review-id'] = "NA"  # Failed to get any output

        # Get the short review (title) from the review with the class 'title'.
        try:
            review_imdb['short_review'] = review.find(
                'a', {'class': 'title'}).get_text().strip()
        except:
            review_imdb['short_review'] = "NA"  # Failed to get any output

        # Get the full review (main paragraph) from the review with the class 'show-more__control'.
        try:
            review_imdb['full_review'] = review.find(
                'div', {'class': 'show-more__control'}).get_text().strip()
        except:
            review_imdb['full_review'] = "NA"  # Failed to get any output

        # Get the review date from the review with the class 'review-date'.
        try:
            review_imdb['review_date'] = review.find(
                'span', {'class': 'review-date'}).get_text().strip()
        except:
            review_imdb['review_date'] = "NA"  # Failed to get any output

        # Get the review rating from the review with the class 'rating-other-user-rating'.
        try:
            review_imdb['rating_value'] = review.find(
                'span', {'class': 'rating-other-user-rating'}).find('span').get_text().strip()
        except:
            review_imdb['rating_value'] = "NA"  # Failed to get any output

        # Add the review to the reviews array.
        reviews.append(review_imdb)

    # Return the reviews array.
    return reviews
