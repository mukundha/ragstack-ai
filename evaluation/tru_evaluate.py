from dotenv import load_dotenv
from trulens_eval import Tru
import os
import time

load_dotenv()

tru = Tru(database_url=os.getenv("TRULENS_DB_CONN_STRING"))
# tru = Tru()
tru.start_evaluator()

while True:
    time.sleep(0.1)
