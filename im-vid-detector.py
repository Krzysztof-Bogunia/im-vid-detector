import os
import numpy as np
import cv2
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
from datetime import datetime
import ffmpeg
import time
import shutil 
import argparse

def detect_objects(model, image, DETECT_THRESHOLD=0.2):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.predict(image, conf=DETECT_THRESHOLD, verbose=False)
    b_mask = np.zeros(image.shape[:2], np.uint8)
    bbox = []
    score = 0.0
    # results[0].show()
    if(results[0].masks != None):
        contour = results[0].masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
        xyxy = results[0].boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
        if(len(xyxy.shape) == 1):
            bbox = xyxy
        else:
            bbox = [np.min(xyxy, axis=0)[0], np.min(xyxy, axis=0)[1], np.max(xyxy, axis=0)[2], np.max(xyxy, axis=0)[3]]
        _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
        score = results[0].boxes.conf[0].item()
    mask = b_mask
    return mask, bbox, score

def save_image(image, path):
    _path = path
    if os.path.exists(_path):
        filename, extension = os.path.splitext(_path)
        iter = 1
        while os.path.exists(_path):
            _path = filename + " (" + str(iter) + ")" + extension
            iter = iter + 1
    cv2.imwrite(_path, image)
    
def process_frame(model, image, DETECT_THRESHOLD=0.2):
    h,w = image.shape[0:2]
    image2 = image
    h2,w2 = image2.shape[0:2]
    MAX_RES = 640*2
    TARGET_HEIGHT = 640
    interpolation=cv2.INTER_NEAREST
    do_scaling = True
    if(w > (MAX_RES)):
        w2 = MAX_RES
        h2 = h*MAX_RES/w
    elif((w < (640)) and (h < (TARGET_HEIGHT))):
        w2 = 640
        h2 = h*640/w
    if(h2 > (TARGET_HEIGHT*2)):
        w2 = w2*TARGET_HEIGHT*2.0/h2
        h2 = TARGET_HEIGHT*2
    if(do_scaling):
        image2 = cv2.resize(image2, 
                            dsize=(int(w2), int(h2)), 
                            interpolation=interpolation)
    
    detected = False
    mask, bbox, score = detect_objects(model, image2, DETECT_THRESHOLD)
    if(score >= DETECT_THRESHOLD):
        detected = True
        mask = cv2.resize(mask, 
                            dsize=(w, h), 
                            interpolation=cv2.INTER_NEAREST)
        bbox[0] = int(bbox[0] * (w/w2))
        bbox[1] = int(bbox[1] * (h/h2))
        bbox[2] = int(bbox[2] * (w/w2))
        bbox[3] = int(bbox[3] * (h/h2))
    return detected, mask, bbox, score

def bboxToRange(CROP_SIZE_OFFSET, h, w, bbox):
    _CROP_SIZE_OFFSET = CROP_SIZE_OFFSET
    if((_CROP_SIZE_OFFSET > 0.0) and (_CROP_SIZE_OFFSET < 1.0)):
        _CROP_SIZE_OFFSET = int(_CROP_SIZE_OFFSET * min([w,h]))
    if((_CROP_SIZE_OFFSET < 0.0) and (_CROP_SIZE_OFFSET > -1.0)):
        _CROP_SIZE_OFFSET = int(_CROP_SIZE_OFFSET * min([w,h]))
    _CROP_SIZE_OFFSET = int(_CROP_SIZE_OFFSET)
    x1, y1, x2, y2 = bbox
    x1 = max({x1-_CROP_SIZE_OFFSET, 0})
    y1 = max({y1-_CROP_SIZE_OFFSET, 0})
    x2 = min({x2+_CROP_SIZE_OFFSET, w})
    y2 = min({y2+_CROP_SIZE_OFFSET, h})
    return x1,y1,x2,y2

def video_cut_and_merge_detections(MEDIA_PATH, file, detections, boxes, w, h, frame_rate, DO_CROP, CROP_SIZE_OFFSET=0, TEMP_PATH="./temp/", MAX_FRAMES_NO_CUT=300):
    filename, extension = os.path.splitext(file)
    frame_detection_ranges = []
    index_detection_ranges = []
    start_frame = 0
    end_frame = 0
    start_index = 0
    end_index = 0
    previous = False
    detection_boxes = []
    box1 = []
    if(not any(detections)):
        return
    
    for i in range(0, len(detections), 1):
        frame_n = i*FRAME_SKIP
        if((detections[i] == True) and (previous == False)):
            start_frame = frame_n
            start_index = i
            box1 = boxes[i]
        elif( ((detections[i] == False) and (previous == True)) or
            ( (detections[i] == True) and (i == (len(detections)-1))) ):
            end_frame = frame_n
            end_index = i-1
            frame_detection_ranges.append([start_frame, end_frame])
            index_detection_ranges.append([start_index, end_index])
            box2 = boxes[i-1]
            box_detection_range = np.array(boxes[start_index:(end_index+1)])
            box_detection_range = box_detection_range[~np.isnan(box_detection_range).any(axis=1)]
            bbox = [np.min(box_detection_range, axis=0)[0], np.min(box_detection_range, axis=0)[1], np.max(box_detection_range, axis=0)[2], np.max(box_detection_range, axis=0)[3]]
            detection_boxes.append(bbox)
        elif(DO_CROP and (detections[i] == True) and (len(frame_detection_ranges) > 0) and (previous == True) and 
             ((frame_n - start_frame) > MAX_FRAMES_NO_CUT)):
            previous = False
            end_frame = frame_n
            end_index = i
            frame_detection_ranges.append([start_frame, end_frame])
            index_detection_ranges.append([start_index, end_index])
            box2 = boxes[i-1]
            box_detection_range = np.array([b for b in boxes[start_index:(end_index+1)] if any(b)])
            box_detection_range = box_detection_range[~np.isnan(box_detection_range).any(axis=1)]
            bbox = [np.min(box_detection_range, axis=0)[0], np.min(box_detection_range, axis=0)[1], np.max(box_detection_range, axis=0)[2], np.max(box_detection_range, axis=0)[3]]
            detection_boxes.append(bbox)
            continue
        previous = detections[i]
        
    if not os.path.exists(TEMP_PATH):
        os.makedirs(TEMP_PATH)
    clips = []
    max_width = 0
    max_height = 0
    if(DO_CROP):
        for i in range(0, len(frame_detection_ranges)):
            bbox = detection_boxes[i]
            x1, y1, x2, y2 = bboxToRange(CROP_SIZE_OFFSET, h, w, bbox)
            width = (x2-x1)
            height = (y2-y1)
            max_width = max(max_width, width)
            max_height = max(max_height, height)
    # handle width,height must be divisible by 2 for ffmpeg
    max_width = int(int(max_width/2) * 2)
    max_height = int(int(max_height/2) * 2)
    temp_files = []
    has_audio = False
    has_video = False
    w2 = w
    h2 = h
    num_clips = 0
    for i in range(0, len(frame_detection_ranges)):
        t1 = frame_detection_ranges[i][0]/frame_rate
        t2 = frame_detection_ranges[i][1]/frame_rate
        try:
            clip = ffmpeg.input(MEDIA_PATH+file)
            video = clip.video.filter('trim', start=t1, end=t2).setpts('PTS-STARTPTS')
            audio = clip.audio
            has_audio = False
            has_video = False
            audio_probe = ffmpeg.probe(MEDIA_PATH+file, select_streams='a')
            video_probe = ffmpeg.probe(MEDIA_PATH+file, select_streams='v')
            if audio_probe['streams']:
                has_audio = True
            if video_probe['streams']:
                has_video = True
            if(not has_video):
                continue
            if(has_audio):
                audio = audio.filter('atrim', start=t1, end=t2).filter('asetpts', 'PTS-STARTPTS')
            if(DO_CROP):
                num_processing = 0
                bbox = detection_boxes[i]
                x1, y1, x2, y2 = bboxToRange(CROP_SIZE_OFFSET, h, w, bbox)
                width = (x2-x1)
                height = (y2-y1)
                # handle width,height must be divisible by 2 for ffmpeg
                width = int(int(width/2) * 2)
                height = int(int(height/2) * 2)
                centerx = x1+int(width/2)
                centery = y1+int(height/2)
                num_processing = num_processing+1
                video_path = TEMP_PATH+"part"+str(num_clips)+"_video"+str(num_processing)+extension
                video = video.crop(x1,y1,width,height)
                (
                    ffmpeg
                    .output( video, filename=video_path, loglevel="quiet")
                    .overwrite_output()
                    .run()
                )
                video = ffmpeg.input(video_path).video
                
                # if((width != max_width) and (height != max_height)):
                if True:
                    w2 = int(max_width)
                    h2 = int(height*max_width/width)
                    if(abs(max_width - width) > abs(max_height-height)):
                        h2 = int(max_height)
                        w2 = int(width*max_height/height)
                    if(w2 > max_width):
                        h2 = int(h2*max_width/w2)
                        w2 = int(max_width)
                    if(h2 > max_height):
                        w2 = int(w2*max_height/h2)
                        h2 = int(max_height)
                    # handle width,height must be divisible by 2 for ffmpeg
                    w2 = int(int(w2/2) * 2)
                    h2 = int(int(h2/2) * 2)
                    num_processing = num_processing+1
                    video_path = TEMP_PATH+"part"+str(num_clips)+"_video"+str(num_processing)+extension
                    time.sleep(1)
                    (
                        ffmpeg
                        .output( video, filename=video_path, vf="scale="+str(w2)+":"+str(h2)+",setsar="+str(1), loglevel="quiet")
                        .overwrite_output()
                        .run()
                    )
                    video = ffmpeg.input(video_path).video
                    
                if((width != max_width) or (height != max_height)):
                    video = video.filter('pad', width=int(max_width), height=int(max_height), x=int(-1), y=int(-1))
            if(has_audio):
                (
                    ffmpeg
                    .output(video, audio, TEMP_PATH+"part"+str(num_clips)+extension, loglevel="quiet")
                    .overwrite_output()
                    .run()
                )
            else:
                (
                    ffmpeg
                    .output(video, TEMP_PATH+"part"+str(num_clips)+extension, loglevel="quiet")
                    .overwrite_output()
                    .run()
                )
            num_clips = num_clips+1
        except:
            print("error when processing fragment of: "+file + ". Skipping clip.")
    #merge partial clips
    streams = []
    partial_files = []
    for i in range(0, num_clips):
        clip = ffmpeg.input(TEMP_PATH+"part"+str(i)+extension)
        streams.append(clip)
        partial_files.append(clip.video)
        if (has_audio):
            partial_files.append(clip.audio)
    if (has_audio):
        concatenated = (
            ffmpeg
            .concat(*partial_files, v=1, a=1)
            .node
        )
        (
            ffmpeg
            .output(concatenated[0], concatenated[1], OUTPUT_MEDIA_PATH + file, loglevel="quiet")
            .overwrite_output()
            .run()
        )
    else:
        concatenated = (
            ffmpeg
            .concat(*partial_files, v=1)
            .node
        )
        (
            ffmpeg
            .output(concatenated[0], OUTPUT_MEDIA_PATH + file, loglevel="quiet")
            .overwrite_output()
            .run()
        )
        
    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH) 
    return OUTPUT_MEDIA_PATH + file

if __name__ == "__main__":
    
    INPUT_PATH = "./input/"
    MASK_SAVE_PATH = "./output/masks/"
    OUTPUT_MEDIA_PATH = "./output/media/"
    TEMP_PATH = "./temp/"
    DETECTION_TEXTS = [""]
    DO_CROP = True
    DETECT_THRESHOLD = 0.7 # 0.02 0.05 0.17 0.7
    CROP_SIZE_OFFSET = 0.04
    FRAME_SKIP = 30
    MAX_FRAMES_NO_CUT = max(FRAME_SKIP*5, 48)
    MODEL_NAME = "yoloe-11m-seg-pf.pt" #"yoloe-11m-seg.pt" "yoloe-11m-seg-pf.pt"
    
    parser = argparse.ArgumentParser(description="img_vid_masker")
    parser.add_argument("--input", help="input media path. Default value: ./input/", default=INPUT_PATH)
    parser.add_argument("--masks", help="output masks path. Default value: ./output/masks/", default=MASK_SAVE_PATH)
    parser.add_argument("--output_media", help="output processed media path. Default value: ./output/media/", default=OUTPUT_MEDIA_PATH)
    parser.add_argument("--temp", help="output temporary media path (*CAN BE AUTOMATICALLY DELETED!*). Default value: ./temp/", default=TEMP_PATH)
    parser.add_argument("--prompt", help="target text description. Default value is empty so model should detect most likely class in input image", default=DETECTION_TEXTS[0])
    parser.add_argument("--crop", help="whether to crop input images to size matching bounding box of detection {0;1}. Default value: 1", default=DO_CROP)
    parser.add_argument("--threshold", help="detection confidence threshold <0; 1>. Default value: 0.7", default=DETECT_THRESHOLD)
    parser.add_argument("--crop_offset", help="detection bounding box crop size offset defined as ratio of pixels <-1; 1>. Default value: 0.04", default=CROP_SIZE_OFFSET)
    parser.add_argument("--frame_skip", help="how many video frames to skip in each iteration of detection. Default value: 30", default=FRAME_SKIP)
    parser.add_argument("--model", help="name of the model for detection. Default value: yoloe-11m-seg-pf.pt (without text prompt) or yoloe-11m-seg.pt (with text prompt)", default=MODEL_NAME)
    
    args = parser.parse_args()
    # parser.print_help()
    if args.input is not None:
        INPUT_PATH = str(args.input)
    if args.masks is not None:
        MASK_SAVE_PATH = str(args.masks)
    if args.output_media is not None:
        OUTPUT_MEDIA_PATH = str(args.output_media)
    if args.temp is not None:
        TEMP_PATH = str(args.temp)
    if args.prompt is not None:
        DETECTION_TEXTS = [str(args.prompt)]
    if args.crop is not None:
        DO_CROP = bool(int(args.crop))
    if args.threshold is not None:
        DETECT_THRESHOLD = float(args.threshold)
    if args.crop_offset is not None:
        CROP_SIZE_OFFSET = float(args.crop_offset)
    if args.frame_skip is not None:
        FRAME_SKIP = int(args.frame_skip)
    if args.model is not None:
        MODEL_NAME = str(args.model)
    
    start = datetime.now()
    model = None
    if(len(DETECTION_TEXTS[0]) > 0):
        MODEL_NAME = MODEL_NAME.replace("-pf", "")
        model = YOLOE(MODEL_NAME)
        model.set_classes(DETECTION_TEXTS, model.get_text_pe(DETECTION_TEXTS))
    else:
        MODEL_NAME = MODEL_NAME.replace(".pt", "-pf.pt")
        MODEL_NAME = MODEL_NAME.replace("-pf-pf", "-pf")
        model = YOLOE(MODEL_NAME)
    
    MEDIA_PATH = ""
    num_files = 0
    num_images = 0
    num_videos = 0
    num_detections = 0
    files = []
    if(os.path.isdir(INPUT_PATH)):
        MEDIA_PATH = INPUT_PATH + str("/")
        MEDIA_PATH = MEDIA_PATH.replace("//", "/")
        files = os.listdir(MEDIA_PATH)
        num_files = len(files)
        print("Input directory has "+str(num_files)+" files")
    else:
        MEDIA_PATH = os.path.dirname(INPUT_PATH) + str("/")
        MEDIA_PATH = MEDIA_PATH.replace("//", "/")
        files = [os.path.basename(INPUT_PATH)]
    for file in files:
        if (file.endswith(".png") or file.endswith(".jpg")):
            num_images = num_images+1
            image = cv2.imread(MEDIA_PATH+file)
            h,w = image.shape[0:2]
            detected, mask, bbox, score = process_frame(model, image, DETECT_THRESHOLD)
            if(detected):
                num_detections = num_detections+1
                
                if not os.path.exists(MASK_SAVE_PATH):
                    os.makedirs(MASK_SAVE_PATH)
                if not os.path.exists(OUTPUT_MEDIA_PATH):
                    os.makedirs(OUTPUT_MEDIA_PATH)
                output_image = image
                if(DO_CROP):
                    x1, y1, x2, y2 = bboxToRange(CROP_SIZE_OFFSET, h, w, bbox)
                    output_image = output_image[y1:y2, x1:x2]
                save_image(mask, MASK_SAVE_PATH + file)
                save_image(output_image, OUTPUT_MEDIA_PATH + file)
                
        if(file.endswith(".mp4") or file.endswith(".mkv")):
            num_videos = num_videos+1
            detections = []
            boxes = []
            cap = cv2.VideoCapture(MEDIA_PATH+file)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_rate = float(cap.get(cv2.CAP_PROP_FPS))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            for i in range(0, frame_count, FRAME_SKIP):
                frame_n = i
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n-1)
                res, image = cap.read()
                detected, mask, bbox, score = process_frame(model, image, DETECT_THRESHOLD)
                if(detected):
                    num_detections = num_detections+1
                    detections.append(True)
                    boxes.append(bbox)
                else:
                    detections.append(False)
                    boxes.append(bbox)

            if(len(boxes) > 0):
                video_cut_and_merge_detections(MEDIA_PATH, file, detections, boxes, w, h, frame_rate, DO_CROP, CROP_SIZE_OFFSET, "./temp/", MAX_FRAMES_NO_CUT)


    stop = datetime.now()
    print("Processed "+str(num_images)+" images")
    print("Processed "+str(num_videos)+" videos")
    print("Detected "+str(num_detections)+" objects")
    print("Elapsed time = "+str(stop-start)+" [h][m][s]")

    

