QT += core gui qml quick network
CONFIG += c++17
CONFIG += console
TEMPLATE = app
TARGET = qt_gpu_probe

SOURCES += \
    main.cpp \
    video_item.cpp \
    video_texture_item.cpp \
    tesla_scene_item.cpp \
    hud_control.cpp \
    ai_box_model.cpp \
    source_controller.cpp

HEADERS += \
    video_item.h \
    video_texture_item.h \
    tesla_scene_item.h \
    hud_control.h \
    ai_box_model.h \
    source_controller.h

RESOURCES += qml.qrc

QMAKE_CXXFLAGS += -O2 -Wall -Wextra
QMAKE_LFLAGS += -Wl,-rpath,/usr/lib/aarch64-linux-gnu/mali
LIBS += -L/lib/aarch64-linux-gnu -l:libmali.so.1 -ldl -lz
