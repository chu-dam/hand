import mujoco
import numpy as np

from dg5f_grasp_control.hand_model import HAND_JOINT_NAMES


class MujocoGravityCompensator:
    def __init__(self, model_xml_path):
        self.model = mujoco.MjModel.from_xml_path(model_xml_path)
        self.data = mujoco.MjData(self.model)
        self.qadr, self.dadr = self._get_joint_addr()
        self.G = np.zeros(self.model.nv, dtype=np.float64)

    def _get_joint_addr(self):
        qadr, dadr = [], []

        for name in HAND_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"joint not found: {name}")

            qadr.append(self.model.jnt_qposadr[jid])
            dadr.append(self.model.jnt_dofadr[jid])

        return np.array(qadr, dtype=int), np.array(dadr, dtype=int)

    def compute(self, q, gravity=None):
        if gravity is not None:
            self.model.opt.gravity[:] = np.asarray(gravity, dtype=np.float64)

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.qpos[self.qadr] = q

        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_rne(self.model, self.data, 0, self.G)

        return self.G[self.dadr].copy()
