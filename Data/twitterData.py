import twitter
import pprint

with open('twitter.key', 'r') as f:
    lines = f.readlines()
    consumer_key = lines[0].strip()
    consumer_secret = lines[1].strip()
    access_token_key = lines[2].strip()
    access_token_secret = lines[3].strip()

t = twitter.Api(consumer_key=consumer_key,
                consumer_secret=consumer_secret,
                access_token_key=access_token_key,
                access_token_secret=access_token_secret)

search = t.GetSearch(term='Bitcoin', raw_query='l=&q=from%3AExcellion&src=typd')

pprint.pprint(search)
