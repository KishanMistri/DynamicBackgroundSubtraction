# DynamicBackgroundSubtraction
Nowadays, we are using filters in picture more often. Many software and camera brands are flaunting upon that as well. In a later stage of filter, they are also applied in videoes. But this is tough because the detecting a perfect foreground/object is alway in motion and boundries are hard to distinguish in the moving foreground &/OR background.

There are multiple way:
1: Detect background, consider it as a noise. Replace it with background color by manipulating pixel color values.
You can use OpenCV for this on images. 
2: Detect foreground and using broders/edge detection take this object only.
-> There are libraries which can do this. (Keras)
3: Train your model with objects which will be part of the video and most concerning to your research. This is something known as object masking. This is used more in spefic areas where the object is known with 20% difference in shape.

We have many done fragmentation on the slides/frames of short videoes. This is creating 60 frames within 12 seconds (5 frames/sec) and then apply the custom algorithm to detect the edges of foreground object. It will detect the difference between frame's 9x9 pixels and compute it until a single frame is either a foreground or a background.(This is GPU extensive process.)

This is my student case study to understand the background substraction is working without using any library or methods used. This gives me how this algorithms work on the base and how can I enhance in specific scenario.
