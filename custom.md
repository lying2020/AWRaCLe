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

```derain_train_high.json``` and ```derain_train_low.json``` contain the paths to clean images and degraded images, split by degradation intensity (as mentioned in Sec. S5 in supplementary). The ``.json``` files have the following structure:

```
[{'image_path': <path_to_image>}]
```
