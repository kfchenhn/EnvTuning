from dataclasses import dataclass


@dataclass
class SeetConfig:
    """SEET 训练控制配置。"""

    enabled: bool = False
    stage: int = 1
    retry_probability: float = 1.0
    max_retry_per_turn: int = 1

    @property
    def use_augmented_env(self) -> bool:
        return self.stage <= 2

    @property
    def allow_peer_anchor(self) -> bool:
        return self.stage >= 3

    @property
    def allow_historical_anchor(self) -> bool:
        return self.stage >= 3

    @property
    def allow_induced_anchor(self) -> bool:
        return self.stage == 2
