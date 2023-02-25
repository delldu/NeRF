# ./instant-ngp data/nerf/lego \
#     --load_model=/tmp/lego.msgpack \
#     --save_model=/tmp/lego.msgpack
# --max_time 5

# ./instant-ngp data/nerf/lego --load_model=/tmp/lego.msgpack \
#     --save_model=/tmp/lego.msgpack --save_mesh "build/lego.obj"

# ./instant-ngp data/nerf/lego --load_model=/tmp/lego.msgpack \
#     --save_point build/lego.ply

./instant-ngp data/nerf/lego --load_model=/tmp/lego.msgpack \
    --save_point build/lego.png
