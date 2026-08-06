from time import sleep

import numpy as np

try:
    from std_msgs.msg import Float64MultiArray
except ImportError:
    Float64MultiArray = None


def publish_effort(pub, effort):
    if Float64MultiArray is None:
        raise RuntimeError("std_msgs is required to publish hardware effort commands")
    msg = Float64MultiArray()
    msg.data = np.asarray(effort, dtype=np.float64).tolist()
    pub.publish(msg)


def pose_pd(q_target, q, qdot, kp=0.285, kd=0.05, limit=0.25):
    err = np.asarray(q_target) - np.asarray(q)
    pd = kp * err - kd * np.asarray(qdot)
    return np.clip(pd, -limit, limit), err


def zero_effort(pub, count=20):
    zero = np.zeros(20, dtype=np.float64)
    for _ in range(count):
        publish_effort(pub, zero)
        sleep(0.01)
