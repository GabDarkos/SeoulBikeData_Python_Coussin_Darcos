# Discord Chatbot - HitBotx - DIA2

## Members
- Benjamin DEMOUGE
- Gabriel DARCOS
- Artémis COUSSIN
- Enzo CUNY

## Description

Going to work, trying to sleep, organizing a party, doing sport...
What do all these situations have in common ? A lot of people listen to music in them !
We wanted to create a chatbot capable to give informations about several musics, like their genre, their artist or even some statistics (danceability, energy...)
Furthermore, we wanted to create a recommendation system to recommend music to the users based on their tastes. 
On Discord, the user can look for a music based on its name. Once he searched it, our bot will not only return the informations about the music but also the youtube link if it exists. 
Then, the user can rate the music directly on Discord with emojis from 0 to 5. That's where our recommendation system can be used. 
After searching at least 3 musics, the user can ask the bot "What should I listen to ?" and our bot will send him the link of a video corresponding to his tastes. 


## Bot Info
- Chatbot platform: Discord
- [Chat with bot](https://discord.gg/uHeMfQTt) (You must download the Python code and launch it before starting to chat)
- [Working video of this bot](https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID_HERE) OUIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII


### Used API

First of all, we used a Discord API so that the users could chat with our bot on this very spread platform. 
It was really useful to use a Discord API because in our code it was easy to gather the data about the user talking to the chatbot and the musivcs he was looking for.

After that we had the idea of using a Youtube API to show the video clip of each music the user looks for. 
However, with this free Youtube API we can only look for 1000 music informations a day. 
Hence, we had to use 3 Youtube API keys in order to do as many queries as we wanted while testing our chatbot. 


### Used Dataset

The dataset we used already existed but we changed it a lot to keep only the informations that could interest the user when he looks for a particular music. 
It is called "Prediction of music genre", ou can find the dataset on this Kaggle page : [Kaggle] (https://www.kaggle.com/datasets/vicsuperman/prediction-of-music-genre?select=music_genre.csv)


## Recommender System

When the user asks something about a music to the bot, the bot sends him a message on Discord with the informations he asked and the video clip of the music. 
There, the user have the possibility to rate the music by reacting 0, 1, 2, 3, 4 or 5 to the message containing the informations.
Each time the user rates a music, his rating is added to a dataframe called "users", with the Discord username of the user reacting to the message, the rate he gaves and the informations about the music.

For our recommender system, we create another dataframe called "df_users" in which we want to have each username and some variables to understand what genre he prefers.
To do so, we code a one-hot-encoder with the music genres existing in the music dataset, which are :
'Electronic', 'Anime', 'Jazz', 'Alternative', 'Country', 'Rap', 'Blues', 'Rock', 'Classical', 'Hip-Hop'
For each username present in our "users" dataframe, we sum up the ratings by music genre with the function df_users = df_users.groupby(by = 'username').sum()
We also let the variables danceability, energy, popularity and tempo which can be useful to see what kind of music the user prefers (energic, calm, popular, etc...)

Then we use the following cosine functions to see the proximity between the user and the musics of the dataset : 
def cosine_distance(user_score,df):
	instance_id = df['instance_id']
	df_values = df['danceability','energy','tempo','popularity','Electronic','Anime','Jazz','Alternative','Country','Rap','Blues','Rock','Classical','Hip-Hop']
	return (int(instance_id),100*round(np.inner(user_score,df_values)/(np.linalg.norm(user_score)*np.linalg.norm(df_values)),3))

def df_cosine_similarity(user_score, df):
	df_values = df['danceability','energy','tempo','popularity','Electronic','Anime','Jazz','Alternative','Country','Rap','Blues','Rock','Classical','Hip-Hop']
	return df.apply(lambda x: cosine_distance(user_score, x), axis=1).sort_values(ascending=False).head(1)


## Language Processing

We used Deep Learning in our code to recognize the different intents even with some errors in the writing. 
You can find the algorithm in the file [chatbot.py] (https://github.com/BenjaminDemouge/Hitbotx/blob/master/chatbot.py)


### Intents and Entities


| Intent         | Entities                   |
|----------------|----------------------------|
| Artist         | "music_name" : [<br>".* of ( .* )" ,<br>".* wrote ( .* )",<br>".* composed ( .* )"] |
| genre          | "music_name" : [<br>".* of ( .* )" ]| 
| information    | "music_name" : [<br>".* about ( .* ) by.* "],<br>"artist_name" : [<br>". * by ( .* )"] |
| random_music   | "genre" : [<br>".* random ( .* ) music .* "<br>],<br>"information" : [<br>". * with ( .* )"] |



## Scenarios
Please describe all the possible scenarios in your chatbot.
### scenario 1:
| User | Bot                                                     |
|------|---------------------------------------------------------|
| Hi   | Welcome to my chatbot😊                                  |
|      | You can use this chat bot using the following examples: |
|      | Hi<br>How are you?<br>I want a pizza                    |

### scenario 2:
| User              | Bot                                                      |
|-------------------|----------------------------------------------------------|
| I want a pizza    | which kind of pizza do you like?<br> Margherita or Greek |
| Margherita please | ok your order is registered. <br> have a nice day.       |


| User                      | Bot                                                |
|---------------------------|----------------------------------------------------|                      
| I want a pizza margherita | ok your order is registered. <br> have a nice day. |

please enter three examples per each scenario




It could have been sufficient to give the informations on the music the user was looking for in a simple message on Discord. However, we found it more user-friendly to give out the youtube link of the song. This way, the user can immediatly listen to the song he was looking for. 

We have also put a lot of functions which were not asked, like a "random" function, where the user can juste ask to see a random music if he wants to discover new things. 
He can do personnalized researches by indicating some parameter values (energy, danceability, tempo, etc...). 
For example : "Give me a random rock or rap music with a high energy and low danceability ?"

Feel free to have fun with our bot, we recommend you to rate at least 5 songs so the recommender system can be useful to you ;)
