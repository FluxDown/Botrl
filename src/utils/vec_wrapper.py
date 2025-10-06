# src/utils/vec_wrapper.py
import numpy as np

class RunningMeanStd:
    def __init__(self, eps=1e-4, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var  = np.ones(shape,  np.float64)
        self.count = eps

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        bmean, bvar, bcount = x.mean(axis=0), x.var(axis=0), x.shape[0]
        self._update_from_moments(bmean, bvar, bcount)

    def _update_from_moments(self, bmean, bvar, bcount):
        delta = bmean - self.mean
        tot = self.count + bcount
        new_mean = self.mean + delta * bcount / tot
        m_a = self.var * self.count
        m_b = bvar * bcount
        M2 = m_a + m_b + delta**2 * self.count * bcount / tot
        new_var = M2 / tot
        self.mean, self.var, self.count = new_mean, new_var, tot

class SimpleVecNormalize:
    def __init__(self, gamma=0.99, eps=1e-8, training=True):
        self.gamma = float(gamma)
        self.eps = eps
        self.training = training
        self.obs_rms = None
        self.ret_rms = None
        self.ret = None
        self.num_envs = None

    def init(self, num_envs, obs_shape):
        self.num_envs = int(num_envs)
        self.ret = np.zeros(self.num_envs, dtype=np.float32)
        if self.obs_rms is None:
            self.obs_rms = RunningMeanStd(shape=obs_shape)
        if self.ret_rms is None:
            self.ret_rms = RunningMeanStd(shape=())

    def normalize_obs_array(self, obs_array):
        obs = np.asarray(obs_array, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        if self.obs_rms is None:
            self.init(num_envs=obs.shape[0], obs_shape=obs.shape[1:])
        if self.training:
            self.obs_rms.update(obs)
        std = np.sqrt(self.obs_rms.var + self.eps)
        norm = (obs - self.obs_rms.mean) / std
        return norm if obs_array.ndim > 1 else norm[0]

    def normalize_reward(self, rewards, dones):
        r = np.asarray(rewards, dtype=np.float32).reshape(-1)
        d = np.asarray(dones,   dtype=np.float32).reshape(-1)
        if self.ret is None or self.ret.shape[0] != r.shape[0]:
            self.init(num_envs=r.shape[0], obs_shape=(1,))
        self.ret = self.ret * (self.gamma * (1.0 - d)) + r
        if self.training:
            self.ret_rms.update(self.ret[:, None])
        std = np.sqrt(self.ret_rms.var + self.eps).astype(np.float32)
        return (r / std).astype(np.float32)

    def set_training(self, training: bool):
        self.training = bool(training)

    def save(self, path):
        np.savez(path,
                 mean=self.obs_rms.mean, var=self.obs_rms.var, count=self.obs_rms.count,
                 ret_var=self.ret_rms.var, ret_mean=self.ret_rms.mean, ret_count=self.ret_rms.count,
                 num_envs=self.num_envs, ret=self.ret)

    def load(self, path):
        z = np.load(path, allow_pickle=False)
        self.obs_rms = RunningMeanStd()
        self.obs_rms.mean  = z["mean"]
        self.obs_rms.var   = z["var"]
        self.obs_rms.count = z["count"]
        self.ret_rms = RunningMeanStd()
        self.ret_rms.mean  = z["ret_mean"]
        self.ret_rms.var   = z["ret_var"]
        self.ret_rms.count = z["ret_count"]
        self.num_envs = int(z["num_envs"])
        self.ret = z["ret"].astype(np.float32)
