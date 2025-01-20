## In Progress

## Training with Custom Data

For a particular degradation (say rain) the data stucture should look like:
```
<data_path>
└── data_awracle
    ├── Train
      └── Derain
        └── <clean_images_folder>
        └── <degraded_images_folder>
        └── derain_train_high.json
        └── derain_train_low.json
    └── Train_clip
      └── Derain
        └── <clean_images_embedding>
        └── <degraded_images_embedding>
```

```derain_train_high.json``` and ```derain_train_low.json``` contain the paths to clean images and degraded images, split by degradation intensity (as mentioned in Sec. S5 in supplementary). The ``.json``` is a list of dictionaries with the following structure:

```
[{'image_path': <path_to_image>, 'target_path': <path_to_target>, 'type': <optional>}, ...]
```

```<path_to_image>``` and ```<path_to_target>``` are relative to the path specified in ```--derain_dir```. ```Train_clip/``` contains precomputed CLIP image embeddings, which are used as context during training.

## Testing on Custom Data

Command for testing is as follows:
```

```
