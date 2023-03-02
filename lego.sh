# ./instant-ngp data/nerf/lego \
#     --load_model=/tmp/lego.msgpack \
#     --save_model=/tmp/lego.msgpack
# --max_time 5

# ./instant-ngp data/nerf/lego --load_model=/tmp/lego.msgpack \
#     --save_model=/tmp/lego.msgpack --save_mesh "build/lego.obj"


# ./instant-ngp data/nerf/lego --load_model=/tmp/lego.msgpack \
#     --save_image 10,build/lego.png

# ./instant-ngp data/nerf/lego --load_model=/tmp/lego.msgpack --no-train \
#     --save_image 0,build/lego.png

./instant-ngp data/nerf/lego --load_model=/tmp/lego.msgpack \
    --save_points 10.0
