model = dict(
    type='PointRCNN',
    backbone=dict(
        type='PointNet2SAMSG',
        in_channels=6,
        num_points= (2048, 1024, 512, 256),
        radius= (0.2, 0.4, 0.8, 1.2),
        num_samples = (64, 32, 16, 16),
        sa_channels=(((16, 16, 32), (32, 32, 64)), ((64, 64, 128), (64, 96,
                                                                    128)),
                     ((128, 196, 256), (128, 196, 256)), ((256, 256, 512),
                                                          (256, 384, 512))),
    )
)