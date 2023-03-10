#/************************************************************************************
#***
#***    Copyright 2023 Dell(18588220928@163.com), All Rights Reserved.
#***
#***    File Author: Dell, 2023年 03月 10日 星期五 10:43:47 CST
#***
#************************************************************************************/
#
#! /bin/sh

usage()
{
    echo "Usage: $0 [options] input output"
    echo "Options:"
    echo "    save_images"
    echo "    save_points"

    exit 1
}

save_images()
{
    ./instant-ngp data/nerf/lego --load_model=/tmp/lego.msgpack --save_images
}


save_points()
{
    ./instant-ngp data/nerf/lego --load_model=/tmp/lego.msgpack --save_points
}


if [ "$*" == "" ] ;
then
    usage
else
    eval "$*"
fi

