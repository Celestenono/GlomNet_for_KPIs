import yaml

from monai.transforms import (
    EnsureChannelFirstd,
    ScaleIntensityd,
    Compose,
    RandGaussianNoised,
    OneOf,
    MedianSmoothd,
    RandGaussianSmoothd,
)

from source.training.hover_trainer import Trainer_hover
import monai

if __name__ == "__main__":
    monai.utils.set_determinism(seed=0, additional_settings=None)
    with open("test_training_config_hover.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    train_transforms = Compose(
        [
            EnsureChannelFirstd(keys=["image"], channel_dim=2),
            ScaleIntensityd(keys=["image"]),
            OneOf(
                transforms=[
                    RandGaussianSmoothd(keys=["image"], sigma_x=(0.1, 1.1), sigma_y=(0.1, 1.1), prob=1.0),
                    MedianSmoothd(keys=["image"], radius=1),
                    RandGaussianNoised(keys=["image"], prob=1.0, std=0.05),
                ]
            ),

        ]
    )
    validation_transforms = Compose(
        [
            EnsureChannelFirstd(keys=["image"], channel_dim=2),
            ScaleIntensityd(keys=["image"]),
        ]
    )
    transforms = {}
    transforms["train"] = train_transforms
    transforms["val"] = validation_transforms

    trainer = Trainer_hover(cfg, transforms=transforms)
    trainer.prepare_data()
    trainer.prepare_training()
    trainer.train()