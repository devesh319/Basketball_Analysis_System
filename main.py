from utils.video_utils import read_video, save_video


def main():

    # Read Video
    video_frames = read_video("./input/video_1.mp4")
    print(type(video_frames[0]))

    # Write Video
    save_video(video_frames, "./output/video_1.avi")
    print("Run Successful!")


if __name__ == "__main__":
    main()
