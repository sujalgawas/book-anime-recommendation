from flask import Flask, render_template, request, jsonify
import os
import pickle
import numpy as np
from fuzzywuzzy import process
import jellyfish
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import logging
from flask import url_for
import ast


app = Flask(__name__, template_folder=os.path.abspath('templates'))

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

popular_df = pickle.load(open('popular.pkl','rb'))
pt = pickle.load(open('pt.pkl','rb'))
books = pickle.load(open('books.pkl','rb'))
similarity_score = pickle.load(open('similarity_score.pkl','rb'))

top_50_anime = pickle.load(open('top_50_anime.pkl','rb'))
animes = pickle.load(open('animes.pkl','rb'))
pivot_table = pickle.load(open('pivot_table.pkl','rb'))
anime_SC = pickle.load(open('anime_SC.pkl','rb'))


# Create a list of all book titles for searching
all_book_titles = list(pt.index)


@app.route('/')
def index():
    book_names = list(popular_df['Book-Title'].values)
    authors = list(popular_df['Book-Author'].values)
    images = list(popular_df['Image-URL-M'].values)
    votes = list(popular_df['num_ratings'].values)
    ratings = list(popular_df['avg_rating'].values)

    # Round ratings to 2 decimal places
    rounded_ratings = [round(rating, 2) for rating in ratings]

    return render_template('index.html',
                           book_name=book_names,
                           author=authors,
                           image=images,
                           votes=votes,
                           rating=rounded_ratings)
@app.route('/anime')
def anime():
    anime_names = list(top_50_anime['Name'].values)
    episodes = list(top_50_anime['Episodes'].values)
    images = list(top_50_anime['images'].values)
    scores = list(top_50_anime['Score'].values)

    # Convert JSON-like strings to dictionaries, handle non-string entries
    parsed_images = []
    for s in images:
        if isinstance(s, str):
            parsed_images.append(json.loads(s.replace("'", "\"")))
        else:
            # Handle non-string values (e.g., floats) appropriately
            parsed_images.append({})

    return render_template('anime.html',
                           anime_name=anime_names,
                           episodes=episodes,
                           images=parsed_images,
                           Score=scores)

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/anime_rec')
def anime_rec():
    return render_template('anime_rec.html')

import pandas as pd

@app.route('/recommend_anime', methods=['POST'])
def recommend_anime():
    user_in = request.form.get('user_in')
    logger.debug(f"Received input: {user_in}")
    
    logger.debug(f"Type of animes: {type(animes)}")
    
    try:
        # Find the anime in the DataFrame
        anime_num = animes.loc[animes['Name'] == user_in,'MAL_ID'].values[0]
        logger.debug(f"Found anime_num: {anime_num}")

        index = np.where(pivot_table.index == anime_num)[0][0]
        logger.debug(f"Found index in pivot_table: {index}")
        similar_items = sorted(list(enumerate(anime_SC[index])), key=lambda x: x[1], reverse=True)[1:6]
        logger.debug(f"Similar items: {similar_items}")
        data = []
        
        for i in similar_items:
            temp_df = animes[animes['MAL_ID'] == pivot_table.index[i[0]]].iloc[0]
            logger.debug(f"Processing anime: {temp_df.get('Name', 'Unknown')}")
            images = temp_df.get('images', {})
            logger.debug(f"Images data type: {type(images)}")
            logger.debug(f"Images data: {images}")

            parsed_images = {}
            if isinstance(images, dict):
                parsed_images = images
            elif isinstance(images, str):
                try:
                    parsed_images = json.loads(images.replace("'", '"'))
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON: {e}")
                    logger.error(f"Problematic string: {images}")
            else:
                logger.warning(f"Unexpected type for images: {type(images)}")

            item = [
                temp_df.get('Name', ''),
                temp_df.get('Score', ''),
                parsed_images,
                temp_df.get('Episodes', '')
            ]
            data.append(item)
        
        logger.info(f"Returning recommendations for: {user_in}")
        logger.debug(f"Data being sent to template: {data}")
        return render_template('anime_rec.html', data=data, input_book=user_in)
    except IndexError:
        logger.warning(f"Anime not found: {user_in}")
        return render_template('anime_rec.html', error='Anime not found.', input_book=user_in)
    except Exception as e:
        logger.exception(f"An error occurred while processing recommendations for {user_in}: {str(e)}")
        return render_template('anime_rec.html', error='An unexpected error occurred. Please try again.', input_book=user_in)


@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')
    
    # Check if the user_input exists in pt.index
    if user_input not in pt.index:
        return render_template('recommend.html', data=[], input_book=user_input, error="Book not found in database.")
    
    # Get the index of the user input book
    index = np.where(pt.index == user_input)[0][0]
    
    # Get similar items based on the similarity score
    similar_items = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]
    data = []
    
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        
        if not temp_df.empty:
            # Drop duplicates based on 'Book-Title'
            unique_books = temp_df.drop_duplicates(subset='Book-Title')
            item.append(unique_books['Book-Title'].values[0])
            item.append(unique_books['Book-Author'].values[0])
            item.append(unique_books['Image-URL-M'].values[0])
            data.append(item)
    
    return render_template('recommend.html', data=data, input_book=user_input)



@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    results = books[books['Book-Title'].str.contains(query, case=False, na=False)]['Book-Title'].head(10).tolist()
    return jsonify([{'title': title} for title in results])

@app.route('/search_anime', methods=['POST'])
def search_anime():
    query = request.form.get('query')
    results = animes[animes['Name'].str.contains(query, case=False, na=False)]['Name'].head(10).tolist()
    return jsonify([{'title': title} for title in results])

if __name__ == '__main__':
    app.run(debug=True)