import os
import json
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Get the API key
SLACK_TOKEN = os.environ.get('SLACK_APP_TOKEN')

# Initialize the Slack API client
client = WebClient(token=SLACK_TOKEN)

# Set the channel ID
channel_id = os.environ.get('SLACK_CHANNEL_ID')

# Retrieve the conversation history of the channel
try:
    channels_history = client.conversations_history(channel=channel_id)
except SlackApiError as e:
    print(f"Error getting channel history: {e}")
    raise

# Preprocessing the data
messages = []
for message in channels_history["messages"]:
    messages.append(message["text"])

stop_words = set(stopwords.words("english"))
tokenized_messages = []
for message in messages:
    tokenized_messages.append([word for word in word_tokenize(message) if word not in stop_words])

# Vectorizing the data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([' '.join(tokens) for tokens in tokenized_messages])

# Clustering the data
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)

# Generating predefined responses
responses = {}
for index, label in enumerate(kmeans.labels_):
    if label not in responses:
        responses[label] = tokenized_messages[index]


# check if the token is valid
try:
    auth_test = client.auth_test()
    bot_mention = auth_test["user_id"]
    # retrieve all channels
    channels = client.conversations_list()

    # join a channel
    join_channel = client.conversations_join(channel=channel_id)

    # setting up a WebSocket connection to listen for incoming messages
    client.on(event="app_mention", callback=on_app_mention)
    client.start()
except SlackApiError as e:
    print("Error : {}".format(e))

def on_app_mention(web_client: WebClient, event: dict):
    message = event['text']
    tokenized_messages = [word for sublist in tokenized_messages for word in sublist]
    if bot_mention in message:
        # generate a response
        label = kmeans.predict(vectorizer.transform([message]))[0]
        response = ' '.join(responses[label])
        print(response)
        # send the response
        client.chat_postMessage(channel=channel_id, text=response)
        
except SlackApiError as e:
    print("Error : {}".format(e))
