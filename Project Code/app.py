from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, Response, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask_cors import CORS
import random
from youtubesearchpython import VideosSearch
import json
# from fetchyoutube import get_yt_video
from pytube import YouTube
import threading
import yt_dlp
import vlc
# import time
from urllib.parse import unquote

# creating the flask app
app = Flask(__name__)
CORS(app)

df = pd.read_csv('cleaned_dataset.csv')
# print(df)

df['Genre'] = df['Genre'].str.lower()

vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(df['Genre'].astype(str))
# print(genre_matrix)

cosine_sim = cosine_similarity(genre_matrix, genre_matrix)
# print(cosine_sim)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/playmusic')
def playmusic():
    return render_template('playmusic.html')

# search musics 
@app.route('/search')
def search():
    song_name = request.args.get('song')
    videos_search = VideosSearch(song_name + " song", limit=10)
    videos = videos_search.result()['result']
    return render_template('search.html', videos=videos, song_name=song_name)

# fetch youtube video data
@app.route('/get_video', methods=["GET"])
def get_video():
    song_name = request.args.get('song')
    if not song_name:
        return jsonify({'success': False, 'message': 'song name is required'})
    videos_search = VideosSearch(song_name + " song", limit=1)
    result = videos_search.result()
    video = result['result'][0]
    video_id = video['id']
    # about video
    video_type = video['type']
    video_title = video['title']
    video_Published_time = video['publishedTime']
    video_duration = video['duration']
    video_link = video['link']
    # video views
    video_full_views = video['viewCount']['text']
    video_short_views = video['viewCount']['short']

    # thumbnail links
    video_thumbnails_links = []
    for i in video['thumbnails']:
        video_thumbnails_links.append(i['url'])
    
    # rich thumbnail
    if 'richThumbnail' in video:
        video_rich_thumbnail = video['richThumbnail']['url']
    else:
        video_rich_thumbnail = None

    # channel information 
    channel = video['channel']
    channel_name = channel['name']
    channel_id = channel['id']
    channel_link = channel['link']
    channel_thumbnail = channel['thumbnails'][0]['url']

    # accessibility information
    accessibility_title = video['accessibility']['title']
    accessibility_duration = video['accessibility']['duration']

    return Response(json.dumps({
        'success': True, 
        'video_id': video_id,
        'video_type': video_type,
        'video_title': video_title,
        'video_Published_time': video_Published_time,
        'video_duration': video_duration,
        'video_link': video_link,
        'video_full_views': video_full_views,
        'video_short_views': video_short_views,
        'video_thumbnails_links': video_thumbnails_links,
        'rich_thumbnail': video_rich_thumbnail,
        'channel_name': channel_name,
        'channel_id': channel_id,
        'channel_link': channel_link,
        'channel_thumbnail': channel_thumbnail,
        'accessibility_title': accessibility_title,
        'accessibility_duration' : accessibility_duration
    }, ensure_ascii=False), content_type='application/json; charset=utf-8')

# recommend song function
def recommend_songs(genre_name):
    genre_name = genre_name.lower().strip()
    filtered_df = df[df['Genre'].apply(lambda x: genre_name in x.split())]
    if filtered_df.empty:
        return []
    filtered_indices = filtered_df.index.to_list()
    similarity_scores = [(i, cosine_sim[i][filtered_indices[0]]) for i in filtered_indices]
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    # top_songs = similarity_scores  #[:10]
    random.shuffle(similarity_scores)
    # top_songs = top_songs    #[:5]
    recommended_songs = []
    for i in similarity_scores:
        row = df.iloc[i[0]]
        recommended_songs.append({
            'Song': row['Song-Name'],
            'Artist': row['Singer/Artists'],
            'Genre': row['Genre'],
            'Album': row['Album/Movie'],
            'Rating': row['User-Rating'],
            # 'Video_id': get_yt_video(row['Song-Name'])['result'][0]['id']
        })
    return recommended_songs


@app.route('/recommend', methods=['GET'])
def recommend():
    genre = request.args.get('genre')
    if not genre:
        return jsonify({"error":"Genre Parameter is required."}), 400
    recommendations = recommend_songs(genre)
    if not recommendations:
        return jsonify({"message": "No songs found for this genre."}), 200
    return jsonify(recommendations)

# to play song using ytdlp and vlc
player = None
@app.route('/play', methods=['POST'])
def play_song_api():
    global player
    data = request.get_json()
    song_name = data.get('song', '')
    song_name = unquote(song_name.strip())
    if not song_name:
        return jsonify({'success': False, 'message': 'Song name is required.'})
    
    def play_song(song_name):
        global player
        ydl_opts ={
            'default_search': 'ytsearch1',
            'quiet': True,
            'format': 'bestaudio/best'
        }
        if player is not None:
            player.stop()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(song_name, download=False)
                if 'entries' in info:
                    info = info['entries'][0]
                url = info['url']
                print(f"Playing: {info['title']}")
                player = vlc.MediaPlayer(url)
                player.play()
            except Exception as e:
                print(f"Error: {e}")
    threading.Thread(target=play_song, args=(song_name,)).start()
    return jsonify({'success': True, 'message': f'Playing {song_name}...'})

@app.route('/pause', methods=['POST'])
def pause_song():
    global player
    if player is not None:
        player.pause()
        return jsonify({'success': True, 'message': 'Song paused.'})
    return jsonify({'success': False, 'message': 'No song is playing.'})


if __name__ == "__main__":
    app.run(debug=True)