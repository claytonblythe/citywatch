

This is a project called citywatch created by Clayton Blythe on 2017/10/27

Email: claytondblythe@gmail.com

## Citywatch

Repository for experimenting with object detection and semantic segmentation in urban environments.

I downloaded a 4k video of someone walking through Times Square, and am using a pretrained SSD (Single Shot Detector) network. It can classify many different categories and is fast, so it is my initial experimentation. I am currently using PIL but that adds a lot of overhead, so I need to work on moving things toward OpenCV and maybe piping a video through ffmpeg to a python script for bounding box classification. 

Anyways here is a fun example of the results so far. 

![Alt Test](https://github.com/claytonblythe/citywatch/blob/master/figures/elmo.png)
![Alt Test](https://github.com/claytonblythe/citywatch/blob/master/figures/elmo_boxed.png)
