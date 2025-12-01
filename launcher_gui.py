import FreeSimpleGUI as sg
import cv2
import numpy as np
import gc
import im_vid_detector as detector

INPUT_PATH = detector.INPUT_PATH
MASK_SAVE_PATH = detector.MASK_SAVE_PATH 
OUTPUT_MEDIA_PATH = detector.OUTPUT_MEDIA_PATH
TEMP_PATH = detector.TEMP_PATH
DETECTION_TEXTS = ' '.join(detector.DETECTION_TEXTS)
DO_CROP = bool(int(detector.DO_CROP))
DETECT_THRESHOLD = detector.DETECT_THRESHOLD
CROP_SIZE_OFFSET = detector.CROP_SIZE_OFFSET
FRAME_SKIP = int(detector.FRAME_SKIP)
MAX_FRAMES_NO_CROP = int(detector.MAX_FRAMES_NO_CROP)
MODEL_NAME = detector.MODEL_NAME
VIDEO_CRF = int(detector.VIDEO_CRF)
VIDEO_PRESET = detector.VIDEO_PRESET
ID_IMG = int(0)
PREVIEW_NUM = int(5)

def removeEmptyFromDict(dict1):
    dict2 = {k: v for k, v in dict1.items() if v is not None}
    return dict2

def checkTextValue(val):
    if( (val is not None) and (len(val)>0) ):
        return True
    else:
        return False
    
def update_preview(window, detections, ID_IMG_, DETECT_THRESHOLD, img_view_size):
    global ID_IMG
    ID_IMG = ID_IMG_
    ID_IMG = max(ID_IMG, 0)
    ID_IMG = min(ID_IMG, len(detections)-1)
    window['-ID_IMG-'].update(str(ID_IMG))
    if((len(detections) > 0) and (detections[ID_IMG].image is not None)):
        img_size = np.flip( np.array(detections[ID_IMG].image.shape[0:2]) )
        img_size_ratio = np.array(img_view_size) / img_size
        if(detections[ID_IMG].score >= float(DETECT_THRESHOLD)):
            img = detections[ID_IMG].drawBbox()
            img = cv2.resize(img, np.round(img_size*min(img_size_ratio)).astype(np.int64))
            window['-LEFT_IMG-'].update( data=cv2.imencode('.png', img)[1].tobytes() )
            window['-DETECTED-'].update(str(True))
            window['-SCORE-'].update(str(detections[ID_IMG].score))
        else:
            img = detections[ID_IMG].image
            img = cv2.resize(img, np.round(img_size*min(img_size_ratio)).astype(np.int64))
            window['-LEFT_IMG-'].update( data=cv2.imencode('.png', img)[1].tobytes() )
            window['-DETECTED-'].update(str(False))
            window['-SCORE-'].update(str(detections[ID_IMG].score))
    if((len(detections) > 0) and (detections[ID_IMG].mask is not None)):
        img_size = np.flip( np.array(detections[ID_IMG].mask.shape[0:2]) )
        img_size_ratio = np.array(img_view_size) / img_size
        mask = detections[ID_IMG].mask
        mask = cv2.resize(mask, np.round(img_size*min(img_size_ratio)).astype(np.int64))
        window['-RIGHT_IMG-'].update( data=cv2.imencode('.png', mask)[1].tobytes() )
        
def main():
    global INPUT_PATH 
    global MASK_SAVE_PATH 
    global OUTPUT_MEDIA_PATH 
    global TEMP_PATH 
    global DETECTION_TEXTS
    global DO_CROP 
    global DETECT_THRESHOLD 
    global CROP_SIZE_OFFSET 
    global FRAME_SKIP 
    global MAX_FRAMES_NO_CROP 
    global MODEL_NAME 
    global VIDEO_CRF 
    global VIDEO_PRESET 
    global ID_IMG
    global PREVIEW_NUM

    detections = []
    img_view_size = [round(1920*0.4), round(1080*0.4)]
    
    # TOP BAR
    # row1 = [    sg.Push(), 
    #             sg.Button('Run', font='Helvetica 18', button_color=('green','limegreen'), key="-RUN-"), 
    #             sg.Button('Preview', font='Helvetica 16', button_color=('MediumSpringGreen','MediumSeaGreen'), key="-PREVIEW-"), 
    #             sg.Push()   ]
    row1 = [    sg.Push(), 
                sg.Button('Run', font='Helvetica 18', button_color=('green','limegreen'), key="-RUN-"), 
                sg.Push()   ]
    
    # SETUP TAB
    button_media_file = sg.Button('Media file', font='Helvetica 14', tooltip=detector.arg_descriptions["INPUT_PATH"], key="-MEDIA-FILE-")
    button_media_folder = sg.Button('Media folder', font='Helvetica 14', tooltip=detector.arg_descriptions["INPUT_PATH"], key="-MEDIA-FOLDER-")
    button_output_folder = sg.Button('Output media folder', font='Helvetica 14', tooltip=detector.arg_descriptions["OUTPUT_MEDIA_PATH"], key="-OUTPUT-FOLDER-")
    button_mask_folder = sg.Button('Mask folder', font='Helvetica 14', tooltip=detector.arg_descriptions["MASK_SAVE_PATH"], key="-MASK-FOLDER-")
    button_temp_folder = sg.Button('Temp folder', font='Helvetica 14', tooltip=detector.arg_descriptions["TEMP_PATH"], key="-TEMP-FOLDER-")
    button_model_file = sg.Button('Model file', font='Helvetica 14', tooltip=detector.arg_descriptions["MODEL_NAME"], key="-MODEL-FILE-")
    input_column = [  [sg.Text('INPUT PATH SELECTION', justification='center', size=(30, 1), font='Helvetica 18')],                        
                        [button_media_file],
                        [button_media_folder],
                        [sg.Input(default_text=str(INPUT_PATH), tooltip=detector.arg_descriptions["INPUT_PATH"], key="-INPUT_PATH-")]  ]
    output_column = [  [sg.Text('OUTPUT PATH SELECTION', justification='center', size=(30, 1), font='Helvetica 18')],                        
                        [button_output_folder],
                        [sg.Input(default_text=str(OUTPUT_MEDIA_PATH), tooltip=detector.arg_descriptions["OUTPUT_MEDIA_PATH"], key="-OUTPUT_MEDIA_PATH-")],
                        [button_mask_folder],
                        [sg.Input(default_text=str(MASK_SAVE_PATH), tooltip=detector.arg_descriptions["MASK_SAVE_PATH"], key="-MASK_SAVE_PATH-")],
                        [button_temp_folder],
                        [sg.Input(default_text=str(TEMP_PATH), tooltip=detector.arg_descriptions["TEMP_PATH"], key="-TEMP_PATH-")] ]
    processing_column = [  [sg.Text('PROCESSING SETTINGS', justification='center', size=(30, 1), font='Helvetica 18')],                        
                            [button_model_file],
                            [sg.Text('model'), 
                            sg.Input(default_text=str(MODEL_NAME), enable_events=True, tooltip=detector.arg_descriptions["MODEL_NAME"], key="-MODEL_NAME-")],
                            [sg.Text('prompt'), 
                            sg.Input(default_text=str(DETECTION_TEXTS), enable_events=True, tooltip=detector.arg_descriptions["DETECTION_TEXTS"], key="-DETECTION_TEXTS-")],
                            [sg.Checkbox('crop', default=int(DO_CROP), enable_events=True, tooltip=detector.arg_descriptions["DO_CROP"], key="-DO_CROP_CBOX-"), 
                            sg.Input(default_text=int(DO_CROP), enable_events=True, tooltip=detector.arg_descriptions["DO_CROP"], key="-DO_CROP-")], 
                            [sg.Text('threshold'), 
                            sg.Slider((0.0,1.0), resolution=0.002, default_value=float(DETECT_THRESHOLD), enable_events=True, orientation='h', tooltip=detector.arg_descriptions["DETECT_THRESHOLD"], key="-DETECT_THRESHOLD_SLIDER-"), 
                            sg.Input(default_text=str(DETECT_THRESHOLD), enable_events=True, tooltip=detector.arg_descriptions["DETECT_THRESHOLD"], key="-DETECT_THRESHOLD-")],
                            [sg.Text('crop offset'), 
                            sg.Slider((-0.2,0.2), resolution=0.005, default_value=float(CROP_SIZE_OFFSET), enable_events=True, orientation='h', tooltip=detector.arg_descriptions["CROP_SIZE_OFFSET"], key="-CROP_SIZE_OFFSET_SLIDER-"),
                            sg.Input(default_text=str(CROP_SIZE_OFFSET), enable_events=True, tooltip=detector.arg_descriptions["CROP_SIZE_OFFSET"], key="-CROP_SIZE_OFFSET-")],
                            [sg.Text('frame skip'), 
                            sg.Slider((1,300), resolution=1, default_value=int(FRAME_SKIP), enable_events=True, orientation='h', tooltip=detector.arg_descriptions["FRAME_SKIP"], key="-FRAME_SKIP_SLIDER-"),
                            sg.Input(default_text=str(int(FRAME_SKIP)), enable_events=True, tooltip=detector.arg_descriptions["FRAME_SKIP"], key="-FRAME_SKIP-")],
                            [sg.Text('max frames no crop'), 
                            sg.Slider((0,2000), resolution=1, default_value=int(MAX_FRAMES_NO_CROP), enable_events=True, orientation='h', tooltip=detector.arg_descriptions["MAX_FRAMES_NO_CROP"], key="-MAX_FRAMES_NO_CROP_SLIDER-"), 
                            sg.Input(default_text=str(int(MAX_FRAMES_NO_CROP)), enable_events=True, tooltip=detector.arg_descriptions["MAX_FRAMES_NO_CROP"], key="-MAX_FRAMES_NO_CROP-")],
                            [sg.Text('crf'), 
                            sg.Slider((0,51), resolution=1, default_value=int(VIDEO_CRF), enable_events=True, orientation='h', tooltip=detector.arg_descriptions["VIDEO_CRF"], key="-VIDEO_CRF_SLIDER-"), 
                            sg.Input(default_text=str(int(VIDEO_CRF)), enable_events=True, tooltip=detector.arg_descriptions["VIDEO_CRF"], key="-VIDEO_CRF-")],
                            [sg.Text('video preset'), 
                            sg.Input(default_text=str(VIDEO_PRESET), enable_events=True, tooltip=detector.arg_descriptions["VIDEO_PRESET"], key="-VIDEO_PRESET-")]  ]
    setupTab = [    [sg.Frame(layout=input_column, element_justification='center', vertical_alignment="top", title=''), 
                    sg.Frame(layout=output_column, element_justification='center', vertical_alignment="top", title=''),
                    sg.Frame(layout=processing_column, element_justification='center', vertical_alignment="top", title='')] ]
    
    # PREVIEW TAB
    button_prev_img = sg.Button('<', font='Helvetica 14', key="-BTN_PREV_IMG-")
    button_next_img = sg.Button('>', font='Helvetica 14', key="-BTN_NEXT_IMG-")
    button_minus_preview = sg.Button('-', font='Helvetica 14', key="-BTN_MINUS_PREVIEW-")
    button_plus_preview = sg.Button('+', font='Helvetica 14', key="-BTN_PLUS_PREVIEW-")
    selection_row = [   sg.Text('IMAGES NUMBER'), 
                        sg.Input(default_text=str(PREVIEW_NUM), size=(5, 1), tooltip="number of total preview images", key="-PREVIEW_NUM-"), 
                        button_minus_preview, 
                        button_plus_preview,
                        sg.Text('SELECTION NUMBER'), 
                        sg.Input(default_text=str(ID_IMG), size=(5, 1), tooltip="selected number of preview image", key="-ID_IMG-"), 
                        button_prev_img, 
                        button_next_img, 
                        sg.Push(), 
                        sg.Text('DETECTION | SCORE '), 
                        sg.Text('False', key="-DETECTED-"), 
                        sg.Text('-1', key="-SCORE-"), 
                        sg.Push() ]
    images_row = [  sg.Frame(title='image', layout=[[sg.Image('', key='-LEFT_IMG-')]], size=img_view_size), 
                    sg.Frame(title='mask', layout=[[sg.Image('', key='-RIGHT_IMG-')]], size=img_view_size)    ]
    previewTab = [  [sg.Push(), sg.Button('Preview', font='Helvetica 16', button_color=('MediumSpringGreen','MediumSeaGreen'), key="-PREVIEW-"), sg.Push()], 
                    selection_row, 
                    images_row ]
    
    layout = [  row1,
                [sg.TabGroup([[sg.Tab('setup', setupTab), sg.Tab('previews', previewTab)]])],
                [sg.Output(size=(100, 12))]    ]
    window = sg.Window('Image-Video-Detector', layout, no_titlebar=False, location=(0, 0))
    
    while True:
        event, values = window.read(timeout=1000)
        if event in ('Exit', sg.WINDOW_CLOSED):
            break
        
        # load variables from UI
        if(checkTextValue( window['-INPUT_PATH-'].get())):
            INPUT_PATH = window['-INPUT_PATH-'].get()
        else:
            INPUT_PATH = None
        if(checkTextValue( window['-MASK_SAVE_PATH-'].get())):
            MASK_SAVE_PATH = window['-MASK_SAVE_PATH-'].get()
        else:
            MASK_SAVE_PATH = None
        if(checkTextValue( window['-OUTPUT_MEDIA_PATH-'].get())):
            OUTPUT_MEDIA_PATH = window['-OUTPUT_MEDIA_PATH-'].get()
        else:
            OUTPUT_MEDIA_PATH = None
        if(checkTextValue( window['-TEMP_PATH-'].get())):
            TEMP_PATH = window['-TEMP_PATH-'].get()
        else:
            TEMP_PATH = None
        if(checkTextValue( window['-DETECTION_TEXTS-'].get())):
            DETECTION_TEXTS = window['-DETECTION_TEXTS-'].get()
        else:
            DETECTION_TEXTS = None
        if(checkTextValue( window['-DO_CROP-'].get())):
            DO_CROP = window['-DO_CROP-'].get()
        else:
            DO_CROP = None
        if(checkTextValue( window['-DETECT_THRESHOLD-'].get())):
            DETECT_THRESHOLD = window['-DETECT_THRESHOLD-'].get()
        else:
            DETECT_THRESHOLD = None
        if(checkTextValue( window['-CROP_SIZE_OFFSET-'].get())):
            CROP_SIZE_OFFSET = window['-CROP_SIZE_OFFSET-'].get()
        else:
            CROP_SIZE_OFFSET = None
        if(checkTextValue( window['-FRAME_SKIP-'].get())):
            FRAME_SKIP = window['-FRAME_SKIP-'].get()
        else:
            FRAME_SKIP = None
        if(checkTextValue( window['-MAX_FRAMES_NO_CROP-'].get())):
            MAX_FRAMES_NO_CROP = window['-MAX_FRAMES_NO_CROP-'].get()
        else:
            MAX_FRAMES_NO_CROP = None
        if(checkTextValue( window['-MODEL_NAME-'].get())):
            MODEL_NAME = window['-MODEL_NAME-'].get()
        else:
            MODEL_NAME = None
        if(checkTextValue( window['-VIDEO_CRF-'].get())):
            VIDEO_CRF = window['-VIDEO_CRF-'].get()
        else:
            VIDEO_CRF = None
        if(checkTextValue( window['-VIDEO_PRESET-'].get())):
            VIDEO_PRESET = window['-VIDEO_PRESET-'].get()
        else:
            VIDEO_PRESET = None
        if(checkTextValue( window['-PREVIEW_NUM-'].get())):
            PREVIEW_NUM = int(window['-PREVIEW_NUM-'].get())
        if(checkTextValue( window['-ID_IMG-'].get())):
            ID_IMG = int(window['-ID_IMG-'].get())
            
        
        if event in ("-MEDIA-FILE-"):
            INPUT_PATH = sg.popup_get_file('Open input file')
            if not checkTextValue(INPUT_PATH):
                INPUT_PATH = None
            else:
                window['-INPUT_PATH-'].update(INPUT_PATH)
            
        if event in ("-MEDIA-FOLDER-"):
            INPUT_PATH = sg.popup_get_folder('Open input folder')
            if not checkTextValue(INPUT_PATH):
                INPUT_PATH = None
            else:
                window['-INPUT_PATH-'].update(INPUT_PATH)

        if event in ("-OUTPUT-FOLDER-"):
            OUTPUT_MEDIA_PATH = sg.popup_get_folder('Open output folder')
            if not checkTextValue(OUTPUT_MEDIA_PATH):
                OUTPUT_MEDIA_PATH = None
            else:
                window['-OUTPUT_MEDIA_PATH-'].update(OUTPUT_MEDIA_PATH)
        
        if event in ("-MASK-FOLDER-"):
            MASK_SAVE_PATH = sg.popup_get_folder('Open mask folder')
            if not checkTextValue(MASK_SAVE_PATH):
                MASK_SAVE_PATH = None
            else:
                window['-MASK_SAVE_PATH-'].update(MASK_SAVE_PATH)

        if event in ("-TEMP-FOLDER-"):
            TEMP_PATH = sg.popup_get_folder('Open temp folder')
            if not checkTextValue(TEMP_PATH):
                TEMP_PATH = None
            else:
                window['-TEMP_PATH-'].update(TEMP_PATH)
   
        if event in ("-MODEL-FILE-"):
            MODEL_NAME = sg.popup_get_file('Open model file')
            if not checkTextValue(MODEL_NAME):
                MODEL_NAME = None
            else:
                window['-MODEL_NAME-'].update(MODEL_NAME)
        
        if event in ("-DO_CROP-"):
            DO_CROP = window['-DO_CROP-'].get()
            if not checkTextValue(DO_CROP):
                DO_CROP = None
            else:
                DO_CROP = bool(int(DO_CROP))
                window['-DO_CROP_CBOX-'].update(DO_CROP)
        if event in ("-DO_CROP_CBOX-"):
            DO_CROP = values["-DO_CROP_CBOX-"]
            if not DO_CROP:
                DO_CROP = None
            else:
                DO_CROP = bool(int(DO_CROP))
                window['-DO_CROP-'].update(str(int(DO_CROP)))
        
        if event in ("-DETECT_THRESHOLD-"):
            DETECT_THRESHOLD = window['-DETECT_THRESHOLD-'].get()
            if not checkTextValue(DETECT_THRESHOLD):
                DETECT_THRESHOLD = None
            else:
                DETECT_THRESHOLD = float(DETECT_THRESHOLD)
                window['-DETECT_THRESHOLD_SLIDER-'].update(DETECT_THRESHOLD)
        if event in ("-DETECT_THRESHOLD_SLIDER-"):
            DETECT_THRESHOLD = values["-DETECT_THRESHOLD_SLIDER-"]
            if DETECT_THRESHOLD is None:
                DETECT_THRESHOLD = None
            else:
                DETECT_THRESHOLD = float(DETECT_THRESHOLD)
                window['-DETECT_THRESHOLD-'].update(str(DETECT_THRESHOLD))
                
        if event in ("-CROP_SIZE_OFFSET-"):
            CROP_SIZE_OFFSET = window['-CROP_SIZE_OFFSET-'].get()
            if not checkTextValue(CROP_SIZE_OFFSET):
                CROP_SIZE_OFFSET = None
            else:
                CROP_SIZE_OFFSET = float(CROP_SIZE_OFFSET)
                window['-CROP_SIZE_OFFSET_SLIDER-'].update(CROP_SIZE_OFFSET)
        if event in ("-CROP_SIZE_OFFSET_SLIDER-"):
            CROP_SIZE_OFFSET = values["-CROP_SIZE_OFFSET_SLIDER-"]
            if CROP_SIZE_OFFSET  is None:
                CROP_SIZE_OFFSET = None
            else:
                CROP_SIZE_OFFSET = float(CROP_SIZE_OFFSET)
                window['-CROP_SIZE_OFFSET-'].update(str(CROP_SIZE_OFFSET))
        
        if event in ("-FRAME_SKIP-"):
            FRAME_SKIP = window['-FRAME_SKIP-'].get()
            if not checkTextValue(FRAME_SKIP):
                FRAME_SKIP = None
            else:
                FRAME_SKIP = int(FRAME_SKIP)
                window['-FRAME_SKIP_SLIDER-'].update(FRAME_SKIP)
        if event in ("-FRAME_SKIP_SLIDER-"):
            FRAME_SKIP = values["-FRAME_SKIP_SLIDER-"]
            if FRAME_SKIP is None:
                FRAME_SKIP = None
            else:
                FRAME_SKIP = int(FRAME_SKIP)
                window['-FRAME_SKIP-'].update(str(FRAME_SKIP))
        
        if event in ("-MAX_FRAMES_NO_CROP-"):
            MAX_FRAMES_NO_CROP = window['-MAX_FRAMES_NO_CROP-'].get()
            if not checkTextValue(MAX_FRAMES_NO_CROP):
                MAX_FRAMES_NO_CROP = None
            else:
                MAX_FRAMES_NO_CROP = int(MAX_FRAMES_NO_CROP)
                window['-MAX_FRAMES_NO_CROP-'].update(MAX_FRAMES_NO_CROP)
        if event in ("-MAX_FRAMES_NO_CROP_SLIDER-"):
            MAX_FRAMES_NO_CROP = values["-MAX_FRAMES_NO_CROP_SLIDER-"]
            if MAX_FRAMES_NO_CROP is None:
                MAX_FRAMES_NO_CROP = None
            else:
                MAX_FRAMES_NO_CROP = int(MAX_FRAMES_NO_CROP)
                window['-MAX_FRAMES_NO_CROP-'].update(str(MAX_FRAMES_NO_CROP))
        
        if event in ("-VIDEO_CRF-"):
            VIDEO_CRF = window['-VIDEO_CRF-'].get()
            if not checkTextValue(VIDEO_CRF):
                VIDEO_CRF = None
            else:
                VIDEO_CRF = int(VIDEO_CRF)
                window['-VIDEO_CRF_SLIDER-'].update(VIDEO_CRF)
        if event in ("-VIDEO_CRF_SLIDER-"):
            VIDEO_CRF = values["-VIDEO_CRF_SLIDER-"]
            if VIDEO_CRF is None:
                VIDEO_CRF = None
            else:
                VIDEO_CRF = int(VIDEO_CRF)
                window['-VIDEO_CRF-'].update(str(VIDEO_CRF))
                
        if event in ("-PREVIEW_NUM-"):
            temp = window['-PREVIEW_NUM-'].get()
            if checkTextValue(temp):
                PREVIEW_NUM = int(temp)
                window['-PREVIEW_NUM-'].update(str(PREVIEW_NUM))
        if event in ("-BTN_PLUS_PREVIEW-"):
            PREVIEW_NUM = PREVIEW_NUM+1
            window['-PREVIEW_NUM-'].update(str(PREVIEW_NUM))
        if event in ("-BTN_MINUS_PREVIEW-"):
            PREVIEW_NUM = PREVIEW_NUM-1
            window['-PREVIEW_NUM-'].update(str(PREVIEW_NUM))

        if event in ("-ID_IMG-"):
            temp = window['-ID_IMG-'].get()
            if checkTextValue(temp):
                ID_IMG = int(temp)
                window['-ID_IMG-'].update(str(ID_IMG))
        if event in ("-BTN_NEXT_IMG-"):
            ID_IMG = ID_IMG+1
            update_preview(window, detections, ID_IMG, DETECT_THRESHOLD, img_view_size)
        if event in ("-BTN_PREV_IMG-"):
            ID_IMG = ID_IMG-1
            update_preview(window, detections, ID_IMG, DETECT_THRESHOLD, img_view_size)
            
        # handle tasks
        if event in ("-RUN-"):
            args = {  "INPUT_PATH_": INPUT_PATH, 
                        "MASK_SAVE_PATH_": MASK_SAVE_PATH, 
                        "OUTPUT_MEDIA_PATH_": OUTPUT_MEDIA_PATH, 
                        "TEMP_PATH_": TEMP_PATH, 
                        "DETECTION_TEXTS_": DETECTION_TEXTS, 
                        "DO_CROP_": DO_CROP,
                        "DETECT_THRESHOLD_": DETECT_THRESHOLD, 
                        "CROP_SIZE_OFFSET_": CROP_SIZE_OFFSET, 
                        "FRAME_SKIP_": FRAME_SKIP, 
                        "MAX_FRAMES_NO_CROP_": MAX_FRAMES_NO_CROP, 
                        "MODEL_NAME_": MODEL_NAME, 
                        "VIDEO_CRF_": VIDEO_CRF, 
                        "VIDEO_PRESET_": VIDEO_PRESET   }
            args = removeEmptyFromDict(args)
            if("DETECTION_TEXTS_" in args):
                args["DETECTION_TEXTS_"] = [args["DETECTION_TEXTS_"]]
            print("\nRunning detector with input arguments:\n"+str(args))
            detector.main(**args)
            collected = gc.collect()
            
        if event in ("-PREVIEW-"):
            args = {    "N": PREVIEW_NUM,
                        "INPUT_PATH_": INPUT_PATH, 
                        "DETECTION_TEXTS_": DETECTION_TEXTS, 
                        "DO_CROP_": DO_CROP,
                        "DETECT_THRESHOLD_": DETECT_THRESHOLD, 
                        "CROP_SIZE_OFFSET_": CROP_SIZE_OFFSET, 
                        "MODEL_NAME_": MODEL_NAME    }
            args = removeEmptyFromDict(args)
            if("DETECTION_TEXTS_" in args):
                args["DETECTION_TEXTS_"] = [args["DETECTION_TEXTS_"]]
            print("\nRunning detector on samples with input arguments:\n"+str(args))
            detections = detector.sample_n_files(**args)
            window['-DETECTED-'].update(str(False))
            window['-SCORE-'].update(str(-1))
            print("Preview detection returned "+str(len(detections))+" images")
            update_preview(window, detections, ID_IMG, DETECT_THRESHOLD, img_view_size)
            collected = gc.collect()
            
    window.close()


    
if __name__ == "__main__":
    main()
