
import pytest
import numpy as np
from auturi.executor.shm.util import WaitingQueue
from auturi.test.utils import check_array
import time


@pytest.mark.skip
def test_queue_no_time():
    queue = WaitingQueue(10, False)
    assert queue.qsize == 0

    queue.insert(np.array([1, 2, 3]))
    assert queue.qsize == 3

    check_array(queue.pop(2), [1, 2])
    assert queue.qsize == 1

    queue.insert(np.array([4, 5]))
    assert queue.qsize == 3

    queue.insert(np.array([6, 7, 8]))
    assert queue.qsize == 6

    check_array(queue.pop(1), [3])
    assert queue.qsize == 5

    check_array(queue.pop("all"), [4, 5, 6, 7, 8])
    assert queue.qsize == 0


def test_queue_record_time():
    queue = WaitingQueue(10, True)

    queue.insert(np.array([1, 2]))
    time.sleep(0.1)

    queue.insert(np.array([3, 4]))
    time.sleep(0.1)

    check_array(queue.pop(3), [1, 2, 3])
    ret = np.sum(queue.q_resides[0]) # [0.2, 0.2, 0.1]
    assert ret >= 0.5 and ret <= 0.6

    queue.insert(np.array([5, 6]))
    time.sleep(0.1)

    queue.insert(np.array([7, 8]))
    time.sleep(0.1)

    check_array(queue.pop(1), [4])
    ret = np.sum(queue.q_resides[1]) # [0.3]
    assert ret >= 0.3 and ret <= 0.4

    check_array(queue.pop("all"), [5, 6, 7, 8])
    ret = np.sum(queue.q_resides[2]) # [0.2, 0.2, 0.1, 0.1]
    assert ret >= 0.6 and ret <= 0.7
