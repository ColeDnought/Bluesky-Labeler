import re
import os
import csv
import time
import signal
import multiprocessing
from types import FrameType
from typing import Any, Generator, Optional

from atproto import CAR, FirehoseSubscribeReposClient, firehose_models, models, parse_subscribe_repos_message
from atproto_client.models.app.bsky.feed.post import Record

url_finder = r'(?:https?://|www\.)\S+'

def _filter_ops(commit: models.ComAtprotoSyncSubscribeRepos.Commit) -> Generator[tuple[Record, models.ComAtprotoSyncSubscribeRepos.Commit], None, None]:

    car = CAR.from_bytes(commit.blocks)
    for op in commit.ops:
        if op.action != 'create' or not op.cid:
            # we are only interested in created records
            continue

        # uri = AtUri.from_str(f'at://{commit.repo}/{op.path}')

        # create_info = {'uri': str(uri), 'cid': str(op.cid), 'author': commit.repo}

        record_raw_data = car.blocks.get(op.cid)
        if not record_raw_data:
            continue

        record = models.get_or_create(record_raw_data, strict=False)

        if record and models.is_record_type(record, models.AppBskyFeedPost):
            # yield {'record': record, **create_info}
            yield record, commit


def worker_main(cursor_value: multiprocessing.Value, pool_queue: multiprocessing.Queue, results_queue: multiprocessing.Queue) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # we handle it in the main process

    while True:
        message = pool_queue.get()

        commit = parse_subscribe_repos_message(message)
        if not isinstance(commit, models.ComAtprotoSyncSubscribeRepos.Commit):
            continue

        if commit.seq % 20 == 0:
            cursor_value.value = commit.seq

        if not commit.blocks:
            continue

        for post, op in _filter_ops(commit):
            match = re.search(url_finder, post.text)

            if match:
                
                try:
                    link = post.embed.external.uri
                except AttributeError:
                    link = match.group()

                results_queue.put([post.created_at, op.repo, link, post.text])


def file_writer(queue: multiprocessing.Queue, filename: str) -> None:
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'author', 'url', 'text'])
        
        while True:
            try:
                data = queue.get()
                if data is None:
                    break
                writer.writerow(data)
                f.flush()
            except Exception as e:
                print(f"Error writing to file: {e}")

def get_firehose_params(cursor_value: multiprocessing.Value) -> models.ComAtprotoSyncSubscribeRepos.Params:
    return models.ComAtprotoSyncSubscribeRepos.Params(cursor=cursor_value.value)


def measure_events_per_second(func: callable) -> callable:
    def wrapper(*args) -> Any:
        wrapper.calls += 1
        cur_time = time.time()

        if cur_time - wrapper.start_time >= 1:
            print(f'NETWORK LOAD: {wrapper.calls} events/second')
            wrapper.start_time = cur_time
            wrapper.calls = 0

        return func(*args)

    wrapper.calls = 0
    wrapper.start_time = time.time()

    return wrapper


def stream_firehose(results_queue: Optional[multiprocessing.Queue] = None, measure_performance: bool = False):
    start_cursor = None
    params = None
    cursor = multiprocessing.Value('i', 0)
    if start_cursor is not None:
        cursor = multiprocessing.Value('i', start_cursor)
        params = get_firehose_params(cursor)

    client = FirehoseSubscribeReposClient(params)

    workers_count = multiprocessing.cpu_count() * 2 - 1
    max_queue_size = 10000

    queue = multiprocessing.Queue(maxsize=max_queue_size)
    pool = multiprocessing.Pool(workers_count, worker_main, (cursor, queue, results_queue))

    def signal_handler(_: int, __: FrameType) -> None:
        print('Keyboard interrupt received. Waiting for the queue to empty before terminating processes...')

        # Stop receiving new messages
        client.stop()

        # Drain the messages queue
        while not queue.empty():
            print('Waiting for the queue to empty...')
            time.sleep(0.2)

        print('Queue is empty. Gracefully terminating processes...')

        pool.terminate()
        pool.join()

        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    def on_message_handler(message: firehose_models.MessageFrame) -> None:
        if cursor.value:
            # we are using updating the cursor state here because of multiprocessing
            # typically you can call client.update_params() directly on commit processing
            client.update_params(get_firehose_params(cursor))

        queue.put(message)

    if measure_performance:
        on_message_handler = measure_events_per_second(on_message_handler)

    client.start(on_message_handler)

if __name__ == '__main__':
    import sys

    filename = sys.argv[1] if len(sys.argv) > 1 else 'url_stream.csv'

    # Create results queue
    results_queue = multiprocessing.Queue()
    
    # Start writer process
    writer_process = multiprocessing.Process(target=file_writer, args=(results_queue, filename))
    writer_process.start()

    try:
        stream_firehose(results_queue, measure_performance=True)
    finally:
        # Cleanup
        results_queue.put(None)
        writer_process.join()