import cv2
import os
import dlib

def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)

def FrameCapture(path, dist_add):
    # Function to extract frames
    # Path to video file
    #detector = dlib.get_frontal_face_detector()
    detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
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
        #print('second  ',second)
        # Saves the frames with frame-count
        # Extract Face
        rects = detector(image, 0)
        # Using mmod human face detector
        x, y, w, h = convert_and_trim_bb(image, rects[0].rect)
        # Using get_frontal_face_detector
        # x, y, w, h = convert_and_trim_bb(image, rects[0])
        #x, y, w, h = [convert_and_trim_bb(image, r) for r in rects]
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        image = image[y:y+h, x:x+w]
        if save_frame % 7 == 0:
            #print(frame_counter)
            cv2.imwrite(os.path.join(dist_add,str(count)+'.jpg'), image)
            count += 1
        if count == 30: break

    print('Finish', os.path.join(dist_add,str(count)+'.jpg'))


def extract_feature():
    video_names = {}
    count = 0
    fake_dataset = './faceforensic/manipulated_sequences'
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
                    if filename.split('.')[0] in video_names.keys():
                        folder_name = video_names[filename.split('.')[0]]
                    else:
                        folder_name = str(count)
                        video_names[filename.split('.')[0]] = str(count)
                        count += 1
                    os.mkdir(os.path.join('./Dataset',dir2,dir1,folder_name))
                    dist_add = os.path.join('./Dataset',dir2, dir1, folder_name)
                    FrameCapture(video_add,dist_add)
                    
                
extract_feature()
print('Finish Extracting Frames')