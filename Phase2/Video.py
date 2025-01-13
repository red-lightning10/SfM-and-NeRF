from moviepy.editor import ImageSequenceClip


def create_video():
    result_path = 'C:/Users/DELL/Downloads/customdata/test/'
    clip = ImageSequenceClip(result_path, 25)
    clip.write_videofile('C:/Users/DELL/Downloads/customdata/test/custom.mp4')

    #result_path = 'C:\\Users\\lenovo\\Desktop\\cv_p2p2\\nerf-pytorch-master\\saved_images_ship_encoded'
    #clip = ImageSequenceClip(result_path, 25)
    #clip.write_videofile('./nerf_ship_encoded.mp4')

    #result_path = 'C:\\Users\\lenovo\\Desktop\\cv_p2p2 - lego unencoded\\nerf-pytorch-master\\saved_images_lego_unencoded'
    #clip = ImageSequenceClip(result_path, 25)
    #clip.write_videofile('./nerf_ship_unencoded.mp4')

create_video()
