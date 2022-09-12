import cv2
import os
from src import detect_faces
from PIL import Image
import shutil
import warnings
warnings.filterwarnings("ignore")


def FrameCapture(path, dist_add):
    vidObj = cv2.VideoCapture(path)
    frameRate = vidObj.get(5)
    #print(frameRate)
    # Used as counter variable
    count = 0
  
    # checks whether frames were extracted
    success = 1
    save_frame = 0
    second = 1
    while success:
  
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()

        frame_counter = vidObj.get(1)
        if frame_counter % frameRate == 0:
            save_frame = 0
            second += 1
        save_frame += 1
        
        if save_frame % 7 == 0:
            #print(frame_counter)
            color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_coverted)
            bb, _ = detect_faces(pil_image)
            #print(bb)
            #cv2.imwrite(os.path.join(dist_add,str(count)+'.jpg'), image)
            face = pil_image.crop(box=(bb[0][0],bb[0][1],bb[0][2],bb[0][3]))
            face.save(os.path.join(dist_add,str(count)+'.jpg'))
            count += 1
        if count == 30: break

    print('Finish', os.path.join(dist_add,str(count)+'.jpg'))


def extract_feature():
    shutil.rmtree('./Dataset/')
    video_names_fake = {}
    count = 0
    fake_dataset = './faceforensic/manipulated_sequences'
    real_dataset = './faceforensic/original_sequences/youtube'
    if os.path.exists(os.path.join('.','Dataset')) == False:
        os.mkdir(os.path.join('.','Dataset'))
    for dir1 in os.listdir(fake_dataset):
        for dir2 in os.listdir(os.path.join(fake_dataset, dir1)):
            #print(os.path.join('./images',dir2,dir1))
            if os.path.exists(os.path.join('./Dataset',dir2)) == False:
                os.mkdir(os.path.join('./Dataset',dir2))
            os.mkdir(os.path.join('./Dataset',dir2,dir1))
            for dir3 in os.listdir(os.path.join(fake_dataset, dir1, dir2)):
                for filename in os.listdir(os.path.join(fake_dataset, dir1, dir2, dir3)):
                    video_add = os.path.join(fake_dataset, dir1, dir2, dir3, filename)
                    if filename.split('.')[0][:3] in video_names_fake.keys():
                        folder_name = video_names_fake[filename.split('.')[0][:3]]
                    else:
                        folder_name = str(count)
                        video_names_fake[filename.split('.')[0][:3]] = str(count)
                        count += 1
                    os.mkdir(os.path.join('./Dataset',dir2,dir1,folder_name))
                    dist_add = os.path.join('./Dataset',dir2, dir1, folder_name)
                    print('Start')
                    FrameCapture(video_add,dist_add)
                    print('Finish')
                
    print('*'*20)
    print('Extracting for Original Videos')
    for dir4 in os.listdir(real_dataset):
        os.mkdir(os.path.join('.','Dataset',dir4,'Original'))
        for dir5 in os.listdir(os.path.join(real_dataset,dir4)):
            for vid_name in os.listdir(os.path.join(real_dataset,dir4,dir5)):
                if vid_name.split('.')[0][:3] in video_names_fake.keys():
                    folder_name = video_names_fake[filename.split('.')[0][:3]]
                else:
                    print('Error')
                    print('{} is not in fake videos !!'.format(vid_name.split('.')[0][:3]))
                os.mkdir(os.path.join('./Dataset',dir4,'Original',folder_name))
                dist_add = os.path.join('./Dataset',dir4,'Original',folder_name)
                vid_add = os.path.join(real_dataset,dir4,dir5,vid_name)
                print('Start')
                FrameCapture(vid_add,dist_add)
                print('Finish')
    print(video_names_fake)
                
extract_feature()
print('Finish Extracting Frames')