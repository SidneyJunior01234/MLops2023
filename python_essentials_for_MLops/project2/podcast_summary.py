import json
import logging
import os
import requests
import xmltodict

from airflow.decorators import dag, task
import pendulum
from airflow.providers.sqlite.operators.sqlite import SqliteOperator
from airflow.providers.sqlite.hooks.sqlite import SqliteHook

from vosk import Model, KaldiRecognizer
from pydub import AudioSegment

logging.basicConfig(
    filename='podcast_summary.log',  # Especifica o arquivo de log
    level=logging.INFO,  # Define o nível de log para INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Formato do log
    datefmt='%Y-%m-%d %H:%M:%S'  # Formato da data/hora
)

PODCAST_URL = "https://www.marketplace.org/feed/podcast/marketplace/"
EPISODE_FOLDER = "episodes"
FRAME_RATE = 16000


@dag(
    dag_id='podcast_summary',
    schedule_interval="@daily",
    start_date=pendulum.datetime(2022, 5, 30),
    catchup=False,
)
def podcast_summary():

    create_database = SqliteOperator(
        task_id='create_table_sqlite',
        sql=r"""
        CREATE TABLE IF NOT EXISTS episodes (
            link TEXT PRIMARY KEY,
            title TEXT,
            filename TEXT,
            published TEXT,
            description TEXT,
            transcript TEXT
        );
        """,
        sqlite_conn_id="podcasts"
    )

    @task()
    def get_episodes():
        try:
            """
            Tarefa para obter dados do feed do podcast.

            Faz uma solicitação HTTP para o URL do feed do podcast, analisa os dados XML e retorna a lista de episódios.
            """
            data = requests.get(PODCAST_URL)
            feed = xmltodict.parse(data.text)
            episodes = feed["rss"]["channel"]["item"]
            print(f"Found {len(episodes)} episodes.")
            return episodes
        except Exception as e:
            logging.error("Error in get_episodes: %s", str(e))
            raise

    @task()
    def load_episodes(episodes):
        try:
            """
            Tarefa para carregar episódios novos no banco de dados.

            Compara os episódios obtidos com os já armazenados no banco de dados SQLite e adiciona novos episódios.
            """
            hook = SqliteHook(sqlite_conn_id="podcasts")
            stored_episodes = hook.get_pandas_df("SELECT * from episodes;")
            new_episodes = []

            for episode in episodes:
                if episode["link"] not in stored_episodes["link"].values:
                    filename = f"{episode['link'].split('/')[-1]}.mp3"
                    new_episodes.append([episode["link"], episode["title"],
                                        episode["pubDate"], episode["description"], filename])

            hook.insert_rows(table='episodes', rows=new_episodes, target_fields=[
                             "link", "title", "published", "description", "filename"])
            return new_episodes
        except Exception as e:
            logging.error("Error in load_episodes: %s", str(e))
            raise

    @task()
    def download_episode(episode):
        try:
            """
            Tarefa para baixar um episódio de áudio.

            Faz o download do arquivo de áudio do episódio, salva-o localmente e retorna informações sobre o episódio.
            """
            link = episode["link"]
            name_end = link.split('/')[-1]
            filename = f"{name_end}.mp3"
            audio_path = os.path.join(EPISODE_FOLDER, filename)

            if not os.path.exists(audio_path):
                print(f"Downloading {filename}")
                audio = requests.get(episode["enclosure"]["@url"])
                with open(audio_path, "wb+") as f:
                    f.write(audio.content)

            return {"link": link, "filename": filename}
        except Exception as e:
            logging.error("Error in download_episode: %s", str(e))
            raise

    @task()
    def transcribe_episode(episode):
        try:
            """
            Tarefa para transcrever um episódio de áudio para texto.

            Utiliza um modelo de reconhecimento de voz para transcrever o áudio do episódio em texto.
            """
            link = episode["link"]
            filename = episode["filename"]
            filepath = os.path.join(EPISODE_FOLDER, filename)

            model = Model(model_name="vosk-model-en-us-0.22-lgraph")
            rec = KaldiRecognizer(model, FRAME_RATE)
            rec.SetWords(True)

            mp3 = AudioSegment.from_mp3(filepath)
            mp3 = mp3.set_channels(1)
            mp3 = mp3.set_frame_rate(FRAME_RATE)

            step = 20000
            transcript = ""

            for i in range(0, len(mp3), step):
                print(f"Progress: {i/len(mp3)}")
                segment = mp3[i:i+step]
                rec.AcceptWaveform(segment.raw_data)
                result = rec.Result()
                text = json.loads(result)["text"]
                transcript += text

            return {"link": link, "transcript": transcript}
        except Exception as e:
            logging.error("Error in transcribe_episode: %s", str(e))
            raise

    episodes = get_episodes()
    new_episodes = load_episodes(episodes)
    downloaded_episodes = [download_episode(
        episode) for episode in new_episodes]
    transcribed_episodes = [transcribe_episode(
        episode) for episode in downloaded_episodes]

    create_database >> episodes >> new_episodes >> downloaded_episodes >> transcribed_episodes

    # Uncomment this to try speech to text (may not work)
    # speech_to_text(audio_files, new_episodes)

summary = podcast_summary()
