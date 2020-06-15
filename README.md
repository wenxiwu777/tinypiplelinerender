# tinypiplelinerender
It is a simple and tiny and pure software 3d render which can draw models in point/line/triangle mode, and also it supports a bunch of classic and basic 3d engine features, please see REAME part for more details.

Thanks to Dmitry V. Sokolov, I've done this simple piple-line render through Sokolov's course(totally spent 15 days including both learning and programming), it's really a woundful and amzing course, I learned how to write a pure software piple-line render 3 years ago but I never do it in practice somehow, this lesson helped me review related algorithms like Bresenham’s Line, rasterizion of triangles, the back face removal, z-buffer and how to build up a uvw camera system and so on, I'd say the best part of this course is the tiny code sections which is easily to understand and debug, so I can do it right away step by step, it was a great time for doing this.

It supports the following features: 
1.three kinds of drawing mode, point, line and triangle
2.load and parse .obj model file(currently only obj file format)
3.texture mapping
4.per-vertex lighting(flat, Gouraud)
5.per-peixel normal lighting(normal mapping, tagnet space normal mapping)
6.phong speacular lighting model
7.shadow map and simple shadow on the model
8.a biult-in piple line, including vertex shader and fragment shader(aka pixel shader)
9.some transformation functions
10.some pre-created shaders, including TextureShader, ToonShader, FlatShader, GouraudShader, SmoothNormalShader,
NormalMappingShader, PhongSpecularShader, BiTagentNormalShader, DepthShader, ShadowShader

How to use it:
There are only a few key functions and classes,
ModelViewMatrix
ViewPortMatrix
ProjectionMatrix
IShader and its sub-classes
Rasterize
barycentric
Model
TGAImage

This render is flxeible so you can add some new features to it, you can write your own vertex and pixel shaders and do
something inside you want, and some other classic feature I've not implemented herein, like the cube map(skybox),
ambient occlusion(in ray tracer this one is very easy).

Anyway, you can see everything in code, mainly the main.cpp. And the rendering images are in the 'result' folder.
I wrote it on mac os, but I guess it has no problem running on win and linux due to all the codes use c/c++ standard 
library and the drawing surface is TGAImage which uses the std library iteself either.

