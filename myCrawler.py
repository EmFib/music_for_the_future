# myCrawler.py

from fycharts.SpotifyCharts import SpotifyCharts
import sqlalchemy

api = SpotifyCharts()
connector = sqlalchemy.create_engine("sqlite:///spotifycharts.db", echo=False)
api.viral50Daily(output_file = "viral_50_daily.csv", output_db = connector, webhook = ["https://mywebhookssite.com/post/"], start = "2019-05-01", end = "2019-05-31", region = ["it", "de"])
