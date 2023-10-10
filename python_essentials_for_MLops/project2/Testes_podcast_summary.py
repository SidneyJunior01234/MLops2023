import os
import json
import requests
import xmltodict
import pytest
from podcast_summary import podcast_summary

@pytest.fixture
def dag():
    return podcast_summary()

@pytest.fixture
def episodes():
    data = requests.get("https://www.marketplace.org/feed/podcast/marketplace/")
    feed = xmltodict.parse(data.text)
    return feed["rss"]["channel"]["item"]

def test_get_episodes(dag, episodes):
    with dag:
        result = dag.get_task_instance(task_id='get_episodes').run(start_date='2022-05-30T00:00:00', end_date='2022-05-30T23:59:59', ignore_ti_state=True)
    assert len(result.output) == len(episodes)

def test_load_episodes(dag, episodes):
    with dag:
        result = dag.get_task_instance(task_id='load_episodes').run(start_date='2022-05-30T00:00:00', end_date='2022-05-30T23:59:59', ignore_ti_state=True, input_tensors={"episodes": episodes})
    assert len(result.output) == len(episodes)

def test_download_episode(dag, episodes):
    with dag:
        result = dag.get_task_instance(task_id='download_episode').run(start_date='2022-05-30T00:00:00', end_date='2022-05-30T23:59:59', ignore_ti_state=True, input_tensors={"episode": episodes[0]})
    assert os.path.exists(result.output["filename"])

def test_transcribe_episode(dag, episodes):
    with dag:
        result = dag.get_task_instance(task_id='transcribe_episode').run(start_date='2022-05-30T00:00:00', end_date='2022-05-30T23:59:59', ignore_ti_state=True, input_tensors={"episode": {"link": "episode1.mp3", "filename": "episode1.mp3"}})
    assert isinstance(result.output, dict)
    assert "transcript" in result.output
