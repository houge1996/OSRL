from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field


@dataclass
class BCPQTrainConfig:
    # wandb params
    project: str = "OSRL-baselines"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "BCPQ"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
    # dataset params
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float, float, float], ...] = None
    epsilon: float = None
    density: float = 1.0
    # training params
    task: str = "OfflineCarCircle-v0"
    dataset: str = None
    seed: int = 0
    device: str = "cpu"
    threads: int = 4
    reward_scale: float = 0.1
    cost_scale: float = 1
    actor_lr: float = 0.0001
    critic_lr: float = 0.001
    alpha_lr: float = 0.0001
    vae_lr: float = 0.001
    cost_limit: int = 10
    episode_len: int = 300
    batch_size: int = 512
    update_steps: int = 100_000
    num_workers: int = 8
    # model params
    a_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    c_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    vae_hidden_sizes: int = 400
    sample_action_num: int = 10
    gamma: float = 0.99
    tau: float = 0.005
    beta: float = 0.5
    num_q: int = 2
    num_qc: int = 2
    qc_scalar: float = 1.5
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 2500


@dataclass
class BCPQCarCircleConfig(BCPQTrainConfig):
    pass


@dataclass
class BCPQAntRunConfig(BCPQTrainConfig):
    # training params
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200


@dataclass
class BCPQDroneRunConfig(BCPQTrainConfig):
    # training params
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200


@dataclass
class BCPQDroneCircleConfig(BCPQTrainConfig):
    # training params
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300


@dataclass
class BCPQCarRunConfig(BCPQTrainConfig):
    # training params
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200


@dataclass
class BCPQAntCircleConfig(BCPQTrainConfig):
    # training params
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500


@dataclass
class BCPQBallRunConfig(BCPQTrainConfig):
    # training params
    task: str = "OfflineBallRun-v0"
    episode_len: int = 100


@dataclass
class BCPQBallCircleConfig(BCPQTrainConfig):
    # training params
    task: str = "OfflineBallCircle-v0"
    episode_len: int = 200


@dataclass
class BCPQCarButton1Config(BCPQTrainConfig):
    # training params
    task: str = "OfflineCarButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPQCarButton2Config(BCPQTrainConfig):
    # training params
    task: str = "OfflineCarButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPQCarCircle1Config(BCPQTrainConfig):
    # training params
    task: str = "OfflineCarCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCPQCarCircle2Config(BCPQTrainConfig):
    # training params
    task: str = "OfflineCarCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCPQCarGoal1Config(BCPQTrainConfig):
    # training params
    task: str = "OfflineCarGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPQCarGoal2Config(BCPQTrainConfig):
    # training params
    task: str = "OfflineCarGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPQCarPush1Config(BCPQTrainConfig):
    # training params
    task: str = "OfflineCarPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPQCarPush2Config(BCPQTrainConfig):
    # training params
    task: str = "OfflineCarPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPQPointButton1Config(BCPQTrainConfig):
    # training params
    task: str = "OfflinePointButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPQPointButton2Config(BCPQTrainConfig):
    # training params
    task: str = "OfflinePointButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPQPointCircle1Config(BCPQTrainConfig):
    # training params
    task: str = "OfflinePointCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCPQPointCircle2Config(BCPQTrainConfig):
    # training params
    task: str = "OfflinePointCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCPQPointGoal1Config(BCPQTrainConfig):
    # training params
    task: str = "OfflinePointGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPQPointGoal2Config(BCPQTrainConfig):
    # training params
    task: str = "OfflinePointGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPQPointPush1Config(BCPQTrainConfig):
    # training params
    task: str = "OfflinePointPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPQPointPush2Config(BCPQTrainConfig):
    # training params
    task: str = "OfflinePointPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCPQAntVelocityConfig(BCPQTrainConfig):
    # training params
    task: str = "OfflineAntVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCPQHalfCheetahVelocityConfig(BCPQTrainConfig):
    # training params
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCPQHopperVelocityConfig(BCPQTrainConfig):
    # training params
    task: str = "OfflineHopperVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCPQSwimmerVelocityConfig(BCPQTrainConfig):
    # training params
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCPQWalker2dVelocityConfig(BCPQTrainConfig):
    # training params
    task: str = "OfflineWalker2dVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCPQEasySparseConfig(BCPQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easysparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCPQEasyMeanConfig(BCPQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easymean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCPQEasyDenseConfig(BCPQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easydense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCPQMediumSparseConfig(BCPQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCPQMediumMeanConfig(BCPQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediummean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCPQMediumDenseConfig(BCPQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumdense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCPQHardSparseConfig(BCPQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCPQHardMeanConfig(BCPQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardmean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCPQHardDenseConfig(BCPQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-harddense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


BCPQ_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": BCPQCarCircleConfig,
    "OfflineAntRun-v0": BCPQAntRunConfig,
    "OfflineDroneRun-v0": BCPQDroneRunConfig,
    "OfflineDroneCircle-v0": BCPQDroneCircleConfig,
    "OfflineCarRun-v0": BCPQCarRunConfig,
    "OfflineAntCircle-v0": BCPQAntCircleConfig,
    "OfflineBallCircle-v0": BCPQBallCircleConfig,
    "OfflineBallRun-v0": BCPQBallRunConfig,
    # safety_gymnasium
    "OfflineCarButton1Gymnasium-v0": BCPQCarButton1Config,
    "OfflineCarButton2Gymnasium-v0": BCPQCarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": BCPQCarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": BCPQCarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": BCPQCarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": BCPQCarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": BCPQCarPush1Config,
    "OfflineCarPush2Gymnasium-v0": BCPQCarPush2Config,
    # safety_gymnasium: point
    "OfflinePointButton1Gymnasium-v0": BCPQPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": BCPQPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": BCPQPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": BCPQPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": BCPQPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": BCPQPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": BCPQPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": BCPQPointPush2Config,
    # safety_gymnasium: velocity
    "OfflineAntVelocityGymnasium-v1": BCPQAntVelocityConfig,
    "OfflineHalfCheetahVelocityGymnasium-v1": BCPQHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": BCPQHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": BCPQSwimmerVelocityConfig,
    "OfflineWalker2dVelocityGymnasium-v1": BCPQWalker2dVelocityConfig,
    # safe_metadrive
    "OfflineMetadrive-easysparse-v0": BCPQEasySparseConfig,
    "OfflineMetadrive-easymean-v0": BCPQEasyMeanConfig,
    "OfflineMetadrive-easydense-v0": BCPQEasyDenseConfig,
    "OfflineMetadrive-mediumsparse-v0": BCPQMediumSparseConfig,
    "OfflineMetadrive-mediummean-v0": BCPQMediumMeanConfig,
    "OfflineMetadrive-mediumdense-v0": BCPQMediumDenseConfig,
    "OfflineMetadrive-hardsparse-v0": BCPQHardSparseConfig,
    "OfflineMetadrive-hardmean-v0": BCPQHardMeanConfig,
    "OfflineMetadrive-harddense-v0": BCPQHardDenseConfig
}