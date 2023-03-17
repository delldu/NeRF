#/************************************************************************************
#***
#***    Copyright 2023 Dell(18588220928@163.com), All Rights Reserved.
#***
#***    File Author: Dell, 2023年 03月 10日 星期五 10:43:47 CST
#***
#************************************************************************************/
#
#! /bin/sh

# ./instant-ngp data/nerf/lego/transforms_train.json --load_model=/tmp/lego.msgpack \
#     --save_render_result --output lego_output

./instant-ngp data/nerf/lego --load_model=/tmp/lego.msgpack \
    --save_render_result --output lego_output
