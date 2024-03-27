# Tracking, counting and speed estimation of vehicles using YOLO v9

This repository contains Python code for tracking vehicles (such as cars, buses, and bikes) as they enter and exit the road, thereby incrementing the counters for incoming and outgoing vehicles. and also speed estimation is calculated by mathematical calcualtion and video info like freame rate

## Installation

```bash
1. git clone https://github.com/Mahdijamebozorg/vehicles-track-count-and-speed-estimation.git
5. pip install ultralytics
15. pip install supervision
```

## Usage

Firstly set the crossing line co-ordinates inside the code i.e yolov8tracker.py for the incoming and outgoing vehicles. And then execute the python code as mentioned below.
### Linux

```bash
python video_process.py -i <input_video_path_with_format> -o <output_video_path_with_format>
```

https://github.com/sankalpvarshney/Track-And-Count-Object-using-YOLO/assets/41926323/bbeb35b4-3f0f-49cd-b222-2bf92ac001f7


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

