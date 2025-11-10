import os
import numpy as np
import cv2
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
from datetime import datetime
import ffmpeg
import time
import argparse
import gc

INPUT_PATH = "./input/"
MASK_SAVE_PATH = "./output/masks/"
OUTPUT_MEDIA_PATH = "./output/media/"
TEMP_PATH = "./temp/"
DETECTION_TEXTS = [""]
DO_CROP = True
DETECT_THRESHOLD = 0.7 # 0.02 0.05 0.17 0.7
CROP_SIZE_OFFSET = 0.04
FRAME_SKIP = 30
MAX_FRAMES_NO_CROP = max(FRAME_SKIP*10, 48)
MODEL_NAME = "yoloe-11m-seg-pf.pt" #"yoloe-11m-seg.pt" "yoloe-11m-seg-pf.pt"
VIDEO_CRF = 23
VIDEO_PRESET = "superfast"

arg_descriptions = {
    "INPUT_PATH": "input media path. Default value: ./input/",
    "MASK_SAVE_PATH": "output masks path. Default value: ./output/masks/",
    "OUTPUT_MEDIA_PATH": "output processed media path. Default value: ./output/media/",
    "TEMP_PATH": "output temporary media path (*CAN BE AUTOMATICALLY DELETED!*). Default value: ./temp/",
    "DETECTION_TEXTS": "target text description. Default value is empty so model should detect most likely class in input image",
    "DO_CROP": "whether to crop input images to size matching bounding box of detection {0;1}. Default value: 1",
    "DETECT_THRESHOLD": "detection confidence threshold <0; 1>. Default value: 0.7",
    "CROP_SIZE_OFFSET": "detection bounding box crop size offset. Value is ratio of image size <-1; 1>. Default value: 0.04",
    "FRAME_SKIP": "how many video frames to skip in each iteration of detection. Default value: 30",
    "MAX_FRAMES_NO_CROP": "name of the model for detection. Default value: yoloe-11m-seg-pf.pt (without text prompt) or yoloe-11m-seg.pt (with text prompt)",
    "MODEL_NAME": "maximum number of video frames before cutting video and applying different crop. Default value: max(FRAME_SKIP*10, 48)",
    "VIDEO_CRF": "ffmpeg argument for quality of output video (best quality is VIDEO_CRF=0). Default value: 23",
    "VIDEO_PRESET": "ffmpeg argument for encoding preset of output video. Default value: superfast"
}

class VideoSettings:
    def __init__(self, frame_rate, h, w, frame_count, VIDEO_CRF=23, VIDEO_PRESET="superfast"):
        self.frame_rate = frame_rate
        self.h = h
        self.w = w
        self.frame_count = frame_count
        self.VIDEO_CRF = VIDEO_CRF
        self.VIDEO_PRESET = VIDEO_PRESET
        
class VideoDetection:
    def __init__(self, frame_n, detected, bbox):
        self.frame_n = frame_n
        self.detected = detected
        self.bbox = bbox
    
def secondsToHHMMSS(seconds):
    hhmmss = time.strftime("%H:%M:%S", time.gmtime(int(seconds)))
    int_val = int(seconds)
    decimal_val = seconds-int(seconds)
    hhmmss = hhmmss[0:-2] + "{:02d}".format(int(hhmmss[-2:])) + "." + str(decimal_val).replace("0.", "")
    return hhmmss

def HHMMSSToSeconds(hhmmss):
    totalSeconds = 0.0
    values = hhmmss.split(':')
    ratios = [1.0, 60.0, 3600.0, 86400] # [seconds, minutes, hours, days]
    for i in range(len(values)):
        text_num = values[-i-1].lstrip('0')
        if(len(text_num) > 0):
            x = float(eval(text_num)) * ratios[i]
            totalSeconds = totalSeconds + x
    return totalSeconds

#get new path if file already exists
def suggest_path(output_file_path):
    suggested_path = output_file_path
    if os.path.exists(suggested_path):
        filename, extension = os.path.splitext(suggested_path)
        iter = 1
        while os.path.exists(suggested_path):
            suggested_path = filename + " (" + str(iter) + ")" + extension
            iter = iter + 1
    return suggested_path

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

def bbox_offset(bbox, CROP_SIZE_OFFSET, h, w):
    _CROP_SIZE_OFFSET = CROP_SIZE_OFFSET
    if((_CROP_SIZE_OFFSET > 0.0) and (_CROP_SIZE_OFFSET < 1.0)):
        _CROP_SIZE_OFFSET = int(_CROP_SIZE_OFFSET * min([w,h]))
    if((_CROP_SIZE_OFFSET < 0.0) and (_CROP_SIZE_OFFSET > -1.0)):
        _CROP_SIZE_OFFSET = int(_CROP_SIZE_OFFSET * min([w,h]))
    _CROP_SIZE_OFFSET = int(_CROP_SIZE_OFFSET)
    x1, y1, x2, y2 = bbox
    x1 = max({int(x1)-_CROP_SIZE_OFFSET, 0})
    y1 = max({int(y1)-_CROP_SIZE_OFFSET, 0})
    x2 = min({int(x2)+_CROP_SIZE_OFFSET, w})
    y2 = min({int(y2)+_CROP_SIZE_OFFSET, h})
    return x1,y1,x2,y2

def get_streams_id(path):
    video_index = None
    audio_index = None
    video_probe = ffmpeg.probe(path, select_streams='V')
    audio_probe = ffmpeg.probe(path, select_streams='a')
    if video_probe['streams']:
        video_index = int(video_probe['streams'][0]['index'])
    if audio_probe['streams']:
        audio_index = int(audio_probe['streams'][0]['index'])
    return video_index, audio_index

def merge_video_files(partial_files, output_file_path):
    dir_path = os.path.dirname(output_file_path) + str("/")
    dir_path = dir_path.replace("//", "/")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    output_file_path2 = suggest_path(output_file_path)

    paths_separated = []
    for file in partial_files:
        paths_separated.append("file "+"'file:"+os.path.abspath(file)+"'"+"\n")
    for i in range(len(paths_separated)):
        if(len(paths_separated[i].replace("\n", "")) <= 0):
            paths_separated.pop(i)
    temp_concat_file = TEMP_PATH+"concat.txt"
    open(temp_concat_file, 'w').writelines(paths_separated)
    (
        ffmpeg
        .input(temp_concat_file, format='concat', safe=0)
        .output(output_file_path2, c='copy', loglevel="quiet")
        .run()
    )
    if os.path.exists(temp_concat_file):
        os.remove(temp_concat_file)
    return output_file_path

def video_cut_and_merge_detections(MEDIA_PATH, file, detections, videoSettings, DO_CROP, CROP_SIZE_OFFSET=0, OUTPUT_MEDIA_PATH="./output/media/", TEMP_PATH="./temp/", MAX_FRAMES_NO_CROP=300):
    filename, extension = os.path.splitext(file)
    w = videoSettings.w
    h = videoSettings.h
    frame_rate = videoSettings.frame_rate
    frame_count = videoSettings.frame_count
    VIDEO_CRF = videoSettings.VIDEO_CRF
    VIDEO_PRESET = videoSettings.VIDEO_PRESET
    frame_detection_ranges = []
    start_frame = 0
    end_frame = 0
    start_index = 0
    end_index = 0
    previous = False
    forced_cut = False
    detection_boxes = []
    collected = gc.collect()

    if(not any(item.detected == True for item in detections)):
        return
    
    for i in range(0, len(detections), 1):
        frame_n = detections[i].frame_n
        if((detections[i].detected == True) and (previous == False)):
            start_frame = frame_n
            start_index = i
        elif( ((detections[i].detected == False) and (previous == True)) or
            ( (detections[i].detected == True) and (i == (len(detections)-1))) ):
            end_frame = frame_n
            end_index = i-1
            frame_detection_ranges.append([start_frame, end_frame])
            box_detection_range = np.array([b.bbox for b in detections[start_index:(end_index+1)] if any(b.bbox)])
            box_detection_range = box_detection_range[~np.isnan(box_detection_range).any(axis=1)]
            bbox = np.array([np.min(box_detection_range, axis=0)[0], np.min(box_detection_range, axis=0)[1], np.max(box_detection_range, axis=0)[2], np.max(box_detection_range, axis=0)[3]])
            detection_boxes.append(bbox)
        elif(DO_CROP and (detections[i].detected == True) and (len(frame_detection_ranges) > 0) and (previous == True) and 
             ((frame_n - start_frame) > MAX_FRAMES_NO_CROP) and 
             (not forced_cut) and
             (np.sum(np.absolute(detections[i].bbox) - detection_boxes[-1]) > 1)):
            forced_cut = True
            previous = False
            end_frame = frame_n-1
            end_index = i-1
            frame_detection_ranges.append([start_frame, end_frame])
            box_detection_range = np.array([b.bbox for b in detections[start_index:(end_index+1)] if any(b.bbox)])
            box_detection_range = box_detection_range[~np.isnan(box_detection_range).any(axis=1)]
            bbox = np.array([np.min(box_detection_range, axis=0)[0], np.min(box_detection_range, axis=0)[1], np.max(box_detection_range, axis=0)[2], np.max(box_detection_range, axis=0)[3]])
            detection_boxes.append(bbox)
            #reuse this frame
            i = i-1
            continue
        previous = detections[i].detected
        forced_cut = False
    #add last fragment of video
    if( (detections[-1].detected == True) and (frame_detection_ranges[-1][1] < (frame_count-2)) ):
        start_frame = frame_detection_ranges[-1][1]+1
        end_frame = frame_count
        frame_detection_ranges.append([start_frame, end_frame])
        detection_boxes.append(detection_boxes[-1])
        
    if not os.path.exists(TEMP_PATH):
        os.makedirs(TEMP_PATH)
    clips = []
    max_width = 0
    max_height = 0
    if(DO_CROP):
        for i in range(0, len(frame_detection_ranges)):
            bbox = detection_boxes[i]
            x1, y1, x2, y2 = bbox_offset(bbox, CROP_SIZE_OFFSET, h, w)
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
        t2 = min(frame_detection_ranges[i][1]+1.0, frame_count)/frame_rate
        try:
            video = None
            audio = None
            video_index = 0
            audio_index = 1
            has_audio = False
            has_video = False
            video_index, audio_index = get_streams_id(MEDIA_PATH+file)
            if audio_index is not None:
                if audio_index >= 0:
                    has_audio = True
                    audio = ffmpeg.input(MEDIA_PATH+file, ss=secondsToHHMMSS(t1), to=secondsToHHMMSS(t2))[str(audio_index)]
            if video_index is not None:
                if video_index >= 0:
                    has_video = True
                    video = ffmpeg.input(MEDIA_PATH+file, ss=secondsToHHMMSS(t1), to=secondsToHHMMSS(t2))[str(video_index)]
            if(not has_video):
                print("WARNING:video stream not detected when processing video clips")
                continue
            if(DO_CROP):
                num_processing = 0
                bbox = detection_boxes[i]
                x1, y1, x2, y2 = bbox_offset(bbox, CROP_SIZE_OFFSET, h, w)
                width = (x2-x1)
                height = (y2-y1)
                # handle width,height must be divisible by 2 for ffmpeg
                width = int(int(width/2-1) * 2)
                height = int(int(height/2-1) * 2)
                centerx = x1+int(width/2)
                centery = y1+int(height/2)
                num_processing = num_processing+1
                video_path = TEMP_PATH+"part"+str(num_clips)+"_video"+str(num_processing)+extension
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
                video = video.filter('crop', width, height, x1, y1)
                video = video.filter('scale', w2, h2)
                video = video.filter('pad', int(max_width), int(max_height), int(-1), int(-1))
                video = video.filter('setsar', 1)
                if(has_audio):                
                    (
                        ffmpeg
                        .output( video, audio, filename=TEMP_PATH+"part"+str(num_clips)+extension, loglevel="quiet", preset=VIDEO_PRESET, crf=VIDEO_CRF)
                        .overwrite_output()
                        .run()
                    )
                else:
                    (
                        ffmpeg
                        .output( video, filename=TEMP_PATH+"part"+str(num_clips)+extension, loglevel="quiet", preset=VIDEO_PRESET, crf=VIDEO_CRF)
                        .overwrite_output()
                        .run()
                    )
            temp_files.append(TEMP_PATH+"part"+str(num_clips)+extension)
            num_clips = num_clips+1
        except:
            print("error when processing fragment of: "+file + ". Skipping clip.")
    #merge partial clips
    partial_files = []
    for i in range(0, num_clips):
        video = None
        audio = None
        has_audio = False
        has_video = False
        video_index, audio_index = get_streams_id(TEMP_PATH+"part"+str(i)+extension)
        if audio_index is not None:
            if audio_index >= 0:
                has_audio = True
        if video_index is not None:
            if video_index >= 0:
                has_video = True
                partial_files.append(TEMP_PATH+"part"+str(i)+extension)
        if(not has_video):
            print("WARNING:video stream not detected when processing video clips")
            continue
    
    output_file_path = OUTPUT_MEDIA_PATH + file
    output_file_path = suggest_path(output_file_path)
    output_file_path = merge_video_files(partial_files, output_file_path)
        
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
    return output_file_path

def main(   INPUT_PATH=INPUT_PATH, 
            MASK_SAVE_PATH=MASK_SAVE_PATH, 
            OUTPUT_MEDIA_PATH=OUTPUT_MEDIA_PATH, 
            TEMP_PATH=TEMP_PATH, 
            DETECTION_TEXTS=DETECTION_TEXTS, 
            DO_CROP=DO_CROP,
            DETECT_THRESHOLD=DETECT_THRESHOLD, 
            CROP_SIZE_OFFSET=CROP_SIZE_OFFSET, 
            FRAME_SKIP=FRAME_SKIP, 
            MAX_FRAMES_NO_CROP=MAX_FRAMES_NO_CROP, 
            MODEL_NAME=MODEL_NAME, 
            VIDEO_CRF=VIDEO_CRF, 
            VIDEO_PRESET=VIDEO_PRESET   ):
    parser = argparse.ArgumentParser(description="Image and video detector. Program can scan all media in folder using AI model and return only those that match specified target. Processing is done locally.")
    parser.add_argument("--input", help=arg_descriptions["INPUT_PATH"], default=INPUT_PATH)
    parser.add_argument("--masks", help=arg_descriptions["MASK_SAVE_PATH"], default=MASK_SAVE_PATH)
    parser.add_argument("--output_media", help=arg_descriptions["OUTPUT_MEDIA_PATH"], default=OUTPUT_MEDIA_PATH)
    parser.add_argument("--temp", help=arg_descriptions["TEMP_PATH"], default=TEMP_PATH)
    parser.add_argument("--prompt", help=arg_descriptions["DETECTION_TEXTS"], default=DETECTION_TEXTS[0])
    parser.add_argument("--crop", help=arg_descriptions["DO_CROP"], default=DO_CROP)
    parser.add_argument("--threshold", help=arg_descriptions["DETECT_THRESHOLD"], default=DETECT_THRESHOLD)
    parser.add_argument("--crop_offset", help=arg_descriptions["CROP_SIZE_OFFSET"], default=CROP_SIZE_OFFSET)
    parser.add_argument("--frame_skip", help=arg_descriptions["FRAME_SKIP"], default=FRAME_SKIP)
    parser.add_argument("--model", help=arg_descriptions["MODEL_NAME"], default=MODEL_NAME)
    parser.add_argument("--max_frames_no_crop", help=arg_descriptions["MAX_FRAMES_NO_CROP"], default=MAX_FRAMES_NO_CROP)
    parser.add_argument("--crf", help=arg_descriptions["VIDEO_CRF"], default=VIDEO_CRF)
    parser.add_argument("--video_preset", help=arg_descriptions["VIDEO_PRESET"], default=VIDEO_PRESET)
    
    args = parser.parse_args()
    if args.input is not None:
        INPUT_PATH = str(args.input)
    if args.masks is not None:
        MASK_SAVE_PATH = str(args.masks)
        MASK_SAVE_PATH = MASK_SAVE_PATH + str("/")
        MASK_SAVE_PATH = MASK_SAVE_PATH.replace("//", "/")
    if args.output_media is not None:
        OUTPUT_MEDIA_PATH = str(args.output_media)
        OUTPUT_MEDIA_PATH = OUTPUT_MEDIA_PATH + str("/")
        OUTPUT_MEDIA_PATH = OUTPUT_MEDIA_PATH.replace("//", "/")
    if args.temp is not None:
        TEMP_PATH = str(args.temp)
        TEMP_PATH = TEMP_PATH + str("/")
        TEMP_PATH = TEMP_PATH.replace("//", "/")
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
    if args.max_frames_no_crop is not None:
        MAX_FRAMES_NO_CROP = int(args.max_frames_no_crop)
    if args.crf is not None:
        VIDEO_CRF = int(args.crf)
    if args.video_preset is not None:
        VIDEO_PRESET = str(args.video_preset)
        
    collected = gc.collect()

    start = datetime.now()
    model = None
    if(len(DETECTION_TEXTS[0]) > 0):
        MODEL_NAME = MODEL_NAME.replace("-pf", "")
        model = YOLOE(MODEL_NAME)
        for i in range(len(DETECTION_TEXTS)):
            DETECTION_TEXTS[i] = DETECTION_TEXTS[i].replace("\"", "")
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
                    x1, y1, x2, y2 = bbox_offset(bbox, CROP_SIZE_OFFSET, h, w)
                    output_image = output_image[y1:y2, x1:x2]
                save_image(mask, MASK_SAVE_PATH + file)
                save_image(output_image, OUTPUT_MEDIA_PATH + file)
                
        if(file.endswith(".mp4") or file.endswith(".mkv")):
            collected = gc.collect()
            video = None
            audio = None
            video_index = 0
            audio_index = 1
            has_video = False
            video_index, audio_index = get_streams_id(MEDIA_PATH+file)
            if video_index >= 0:
                has_video = True
            else:
                print("-skipping file without video stream")
                continue
            num_videos = num_videos+1
            detections = []
            cap = None
            frame_count = -1
            frame_rate = -1.0
            h = -1
            w = -1
            image = None
            if(file.endswith(".mp4")):
                try:
                    cap = cv2.VideoCapture(MEDIA_PATH+file)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_rate = float(cap.get(cv2.CAP_PROP_FPS))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    res, image = cap.read()
                except:
                    if(cap is not None):
                        try:
                            cap.release()
                        except:
                            pass
                        cap = None
            if((cap is None) or (image is None)):
                video_probe = ffmpeg.probe(MEDIA_PATH+file, select_streams='v')['streams'][video_index]
                frame_rate = float(eval(video_probe['r_frame_rate']))
                try:
                    frame_count = int(eval(video_probe['nb_frames']))
                except:
                    duration = video_probe['tags']['DURATION']
                    frame_count = int(HHMMSSToSeconds(duration) * frame_rate)
                h = int(video_probe['height'])
                w = int(video_probe['width'])
            
            videoSettings = VideoSettings(frame_rate,h,w,frame_count,VIDEO_CRF,VIDEO_PRESET)
            for i in range(0, frame_count-1, FRAME_SKIP):
                try:
                    frame_n = i
                    if(cap is not None):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n-1)
                        res, image = cap.read()
                    else:
                        t1 = float(frame_n)/frame_rate
                        t2 = float(min(frame_n+2, frame_count))/frame_rate
                        if((frame_n-1) >= frame_count):
                            video = ffmpeg.input(MEDIA_PATH+file, ss=secondsToHHMMSS(t1))[str(video_index)]
                        else:
                            video = ffmpeg.input(MEDIA_PATH+file, ss=secondsToHHMMSS(t1), to=secondsToHHMMSS(t2))[str(video_index)]
                        video = ffmpeg.input(MEDIA_PATH+file, ss=secondsToHHMMSS(t1))[str(video_index)]
                        buffer, _ = (
                            video
                            .filter('select', 'gte(n,{})'.format(1))
                            .output('pipe:', vframes=1, pix_fmt='bgr24', format='rawvideo', loglevel="quiet")
                            .run(capture_stdout=True)
                        )
                        image = np.frombuffer(buffer, np.uint8, count=h*w*3).reshape(h, w, 3)
                    detected, mask, bbox, score = process_frame(model, image, DETECT_THRESHOLD)
                    if(detected):
                        num_detections = num_detections+1
                        detections.append(VideoDetection(frame_n, True, bbox))
                    else:
                        detections.append(VideoDetection(frame_n, False, bbox))
                except:
                    break

            if(len(detections) > 0):
                video_cut_and_merge_detections(MEDIA_PATH, file, detections, videoSettings, DO_CROP, CROP_SIZE_OFFSET, OUTPUT_MEDIA_PATH, TEMP_PATH, MAX_FRAMES_NO_CROP)

    collected = gc.collect()
    stop = datetime.now()
    print("Processed "+str(num_images)+" images")
    print("Processed "+str(num_videos)+" videos")
    print("Detected "+str(num_detections)+" objects")
    print("Elapsed time = "+str(stop-start)+" [h][m][s]")

if __name__ == "__main__":
    main()
