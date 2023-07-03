# dataset settings
data_type = "DrillingReal"
data_root = "/workspace/data/Data_2023_05_11"

batch_size = 4

disp_range = (32.0, 256.0)
flow_range = (0.0, 160.0)

crop_size = (384, 640)
train_pipeline = dict(type="StereoAugmentor",
                      crop_size=crop_size, bgr_to_rgb=True)
test_pipeline = dict(type="StereoNormalizor", divisor=64)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    train=dict(
        pipeline=train_pipeline,
        type=data_type,
        disp_range=disp_range,
        flow_range=flow_range,
        data_root=data_root,
        num_frames=2,
        split="PATH_TO_TRAINING_SPLIT",
    ),
    val=dict(
        pipeline=test_pipeline,
        type=data_type,
        disp_range=disp_range,
        flow_range=flow_range,
        data_root=data_root,
        num_frames=2,
        split="PATH_TO_VALIDATION_SPLIT",
    ),
    test=dict(
        pipeline=test_pipeline,
        type=data_type,
        disp_range=disp_range,
        flow_range=flow_range,
        data_root=data_root,
        num_frames=2,
        split="PATH_TO_TEST_SPLIT",
    ),
)
