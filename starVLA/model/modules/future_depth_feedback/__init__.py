"""Future-depth prediction and action-feedback modules."""

from .losses import future_depth_feature_loss
from .modules import ConditionalFutureDepthFeedback
from .modules import FutureDepthFeedback
from .modules import FutureDepthPredictor
from .target import FrozenDA3FeatureTarget

__all__ = [
    "FrozenDA3FeatureTarget",
    "ConditionalFutureDepthFeedback",
    "FutureDepthFeedback",
    "FutureDepthPredictor",
    "future_depth_feature_loss",
]
