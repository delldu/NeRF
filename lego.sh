# ./instant-ngp data/nerf/lego \
#     --load_model=/tmp/lego.msgpack \
#     --save_model=/tmp/lego.msgpack
# --max_time 5

# ./instant-ngp data/nerf/lego --load_model=/tmp/lego.msgpack \
#     --save_model=/tmp/lego.msgpack --save_mesh

# ./instant-ngp data/nerf/lego --load_model=/tmp/lego.msgpack --no-train \
#     --save_images

./instant-ngp data/nerf/lego --load_model=/tmp/lego.msgpack \
    --save_points
