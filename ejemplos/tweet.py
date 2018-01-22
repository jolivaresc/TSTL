from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
from credentials import *


class StdOutListener(StreamListener):

    def on_data(self, data):
        try:
            decoded = json.loads(data)
            if "retweeted_status" in decoded:
                pass
            else:
                print (data,end="")
                return True
        except:
            pass

    def on_error(self, status):
        pass
        


if __name__ == '__main__':
    l = StdOutListener()
    auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)
    stream = Stream(auth, l)
    stream.filter(locations = [-99.3227,19.2104,-98.8981,19.5931],languages=["es"])
    #stream.filter(track='djokovic')
