from time import sleep

import numpy as np
from std_msgs.msg import Float64MultiArray


def publish_effort(pub, effort):
    msg = Float64MultiArray()
    msg.data = effort.tolist()
    pub.publish(msg)


def pose_pd(q_target, q, qdot, kp=0.4, kd=0.05, limit=0.25):
    err = q_target - q
    pd = kp * err - kd * qdot
    return np.clip(pd, -limit, limit), err


def zero_effort(pub, count=20):
    zero = np.zeros(20, dtype=np.float64)
    for _ in range(count):
        publish_effort(pub, zero)
        sleep(0.01)
