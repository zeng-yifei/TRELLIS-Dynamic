import os

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from trellis.modules.attention.global_var import *
import torch


set_multiframe(True)
test = True
end_frame = 20
# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()
path = 'dataset/spiderman'
obj_id = path.split('/')[-1]
output_path = f'output/{obj_id}'
os.makedirs(output_path, exist_ok=True)
videos = []
videos_mesh = []
coords_list = []

for i, file in enumerate(sorted(os.listdir(path))):
    # Load an image
    print('--generating ', i,' frame --')

        
    
    image = Image.open(os.path.join(path, file))
    seed = 0
    preprocess_image = True
    sparse_structure_sampler_params={
         "steps": 25,
         "cfg_strength": 6,
     }

    num_samples = 1
    formats =  ['mesh', 'gaussian', 'radiance_field']
    if preprocess_image:
        image = pipeline.preprocess_image(image)
    cond = pipeline.get_cond([image])
    torch.manual_seed(seed)
    
    if i == 0:

        set_first_run(True)
        set_history_attentions([])
        set_attention_idx(0)
        set_history_attentions_sparse([])
        set_attention_idx_sparse(0)
    else:

        set_first_run(False)
        set_attention_idx(0)
        set_attention_idx_sparse(0)
    
    coords = pipeline.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
    
    coords_list.append(coords)
    
    print(f"Memory Usage: {process.memory_info().rss / (1024 * 1024)} MB")
    import gc;gc.collect()
    torch.cuda.empty_cache()
    print(f"Memory Usage after clean: {process.memory_info().rss / (1024 * 1024)} MB")

        
    if test and i==end_frame:
        break
    
    
del_history_attentions()
import gc;gc.collect()
print(f"Memory Usage after clean: {process.memory_info().rss / (1024 * 1024)} MB")

for i, file in enumerate(sorted(os.listdir(path))):
    
    if i == 0:

        set_first_run(True)
        set_history_attentions([])
        set_attention_idx(0)
        set_history_attentions_sparse([])
        set_attention_idx_sparse(0)
    else:

        set_first_run(False)
        set_attention_idx(0)
        set_attention_idx_sparse(0)
    
    
    print('--generating ', i,' frame --')


    slat_sampler_params={
             "steps": 30,
             "cfg_strength": 3,
         }
    if preprocess_image:
        image = pipeline.preprocess_image(image)
    
    image = Image.open(os.path.join(path, file))
    cond = pipeline.get_cond([image])
    
    slat = pipeline.sample_slat(cond, coords_list[i], slat_sampler_params)
    outputs = pipeline.decode_slat(slat, formats)
    # Run the pipeline

    if test and i==end_frame:
        break
    # outputs is a dictionary containing generated 3D assets in different formats:
    # - outputs['gaussian']: a list of 3D Gaussians
    # - outputs['radiance_field']: a list of radiance fields
    # - outputs['mesh']: a list of meshes

    # Render the outputs
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    #imageio.mimsave(output_path+f"/{obj_id}_gs{i}.mp4", video, fps=30)
    videos.append(video)
    video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
    videos_mesh.append(video_mesh)  
    
    imageio.mimsave(output_path+f"/{obj_id}_{i}_gs.mp4", video, fps=30)
    imageio.mimsave(output_path+f"/{obj_id}_{i}_mesh.mp4", video_mesh, fps=30)
    

    print(f"Memory Usage: {process.memory_info().rss / (1024 * 1024)} MB")
    import gc;gc.collect()
    torch.cuda.empty_cache()
    print(f"Memory Usage after clean: {process.memory_info().rss / (1024 * 1024)} MB")
    



n = 3  # Number of frames to take from each video before switching
frame_per_video = []
frame_per_video_mesh = []

len_videos = len(videos)
len_frame = len(videos[0])

for i in range(10000):
    frames_to_get = [3 * i, 3 * i + 1, 3 * i + 2]
    for idx in frames_to_get:
        if idx < len(videos[0]):
            frame_per_video.append(videos[i%len_videos][idx])
            frame_per_video_mesh.append(videos_mesh[i%len_videos][idx])
        else:
            break
imageio.mimsave(output_path+f"/{obj_id}_gs.mp4", frame_per_video, fps=30)
imageio.mimsave(output_path+f"/{obj_id}_mesh.mp4", frame_per_video_mesh, fps=30)
