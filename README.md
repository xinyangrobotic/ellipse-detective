# fast_ellipse_detector
本程序是在Michele Fornaciari, Andrea Prati, Rita Cucchiara两位先生的代码基础上编写的

Original author: mikispace (https://sourceforge.net/projects/yaed/)

### How to compile:

```sh
g++ Main.cpp EllipseDetectorYaed.cpp common.cpp -o ellipse_det -std=c++11 `pkg-config --cflags --libs opencv`
```

### How to run:

```sh
./ellipse_det
```
# ellipse-detective