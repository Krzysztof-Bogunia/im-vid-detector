import FreeSimpleGUI as sg
import cv2
import argparse
import im_vid_detector as detector

def removeEmptyFromDict(dict1):
    dict2 = {k: v for k, v in dict1.items() if v is not None}
    return dict2

def checkTextValue(val):
    if( (val is not None) and (len(val)>0) ):
        return True
    else:
        return False

def main():
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
    
    button_media_file = [sg.Button('Media file', font='Helvetica 14', tooltip=detector.arg_descriptions["INPUT_PATH"], key="-MEDIA-FILE-")]
    button_media_folder = [sg.Button('Media folder', font='Helvetica 14', tooltip=detector.arg_descriptions["INPUT_PATH"], key="-MEDIA-FOLDER-")]
    button_output_folder = [sg.Button('Output media folder', font='Helvetica 14', tooltip=detector.arg_descriptions["OUTPUT_MEDIA_PATH"], key="-OUTPUT-FOLDER-")]
    button_mask_folder = [sg.Button('Mask folder', font='Helvetica 14', tooltip=detector.arg_descriptions["MASK_SAVE_PATH"], key="-MASK-FOLDER-")]
    button_temp_folder = [sg.Button('Temp folder', font='Helvetica 14', tooltip=detector.arg_descriptions["TEMP_PATH"], key="-TEMP-FOLDER-")]
    button_model_file = [sg.Button('Model file', font='Helvetica 14', tooltip=detector.arg_descriptions["MODEL_NAME"], key="-MODEL-FILE-")]

    row1 = [sg.Push(), sg.Button('Run', font='Helvetica 18', button_color=('green','limegreen'), key="-RUN-"), sg.Push()]
    input_column = [  [sg.Text('INPUT PATH SELECTION', justification='center', size=(30, 1), font='Helvetica 18')],                        
                        button_media_file,
                        button_media_folder,
                        [sg.Input(default_text=str(INPUT_PATH), tooltip=detector.arg_descriptions["INPUT_PATH"], key="-INPUT_PATH-")]  ]
    output_column = [  [sg.Text('OUTPUT PATH SELECTION', justification='center', size=(30, 1), font='Helvetica 18')],                        
                        button_output_folder,
                        [sg.Input(default_text=str(OUTPUT_MEDIA_PATH), tooltip=detector.arg_descriptions["OUTPUT_MEDIA_PATH"], key="-OUTPUT_MEDIA_PATH-")],
                        button_mask_folder,
                        [sg.Input(default_text=str(MASK_SAVE_PATH), tooltip=detector.arg_descriptions["MASK_SAVE_PATH"], key="-MASK_SAVE_PATH-")],
                        button_temp_folder,
                        [sg.Input(default_text=str(TEMP_PATH), tooltip=detector.arg_descriptions["TEMP_PATH"], key="-TEMP_PATH-")] ]
    processing_column = [  [sg.Text('PROCESSING SETTINGS', justification='center', size=(30, 1), font='Helvetica 18')],                        
                        button_model_file,
                        [sg.Text('model'), 
                         sg.Input(default_text=str(MODEL_NAME), enable_events=True, tooltip=detector.arg_descriptions["MODEL_NAME"], key="-MODEL_NAME-")],
                        [sg.Text('prompt'), 
                         sg.Input(default_text=str(DETECTION_TEXTS), enable_events=True, tooltip=detector.arg_descriptions["DETECTION_TEXTS"], key="-DETECTION_TEXTS-")],
                        [sg.Checkbox('crop', default=int(DO_CROP), enable_events=True, tooltip=detector.arg_descriptions["DO_CROP"], key="-DO_CROP_CBOX-"), 
                         sg.Input(default_text=int(DO_CROP), enable_events=True, tooltip=detector.arg_descriptions["DO_CROP"], key="-DO_CROP-")], 
                        [sg.Text('threshold'), 
                         sg.Slider((0.0,1.0), resolution=0.005, default_value=float(DETECT_THRESHOLD), enable_events=True, orientation='h', tooltip=detector.arg_descriptions["DETECT_THRESHOLD"], key="-DETECT_THRESHOLD_SLIDER-"), 
                         sg.Input(default_text=str(DETECT_THRESHOLD), enable_events=True, tooltip=detector.arg_descriptions["DETECT_THRESHOLD"], key="-DETECT_THRESHOLD-")],
                        [sg.Text('crop offset'), 
                         sg.Slider((0.0,1.0), resolution=0.005, default_value=float(CROP_SIZE_OFFSET), enable_events=True, orientation='h', tooltip=detector.arg_descriptions["CROP_SIZE_OFFSET"], key="-CROP_SIZE_OFFSET_SLIDER-"),
                         sg.Input(default_text=str(CROP_SIZE_OFFSET), enable_events=True, tooltip=detector.arg_descriptions["CROP_SIZE_OFFSET"], key="-CROP_SIZE_OFFSET-")],
                        [sg.Text('frame skip'), 
                         sg.Slider((1,2000), resolution=1, default_value=int(FRAME_SKIP), enable_events=True, orientation='h', tooltip=detector.arg_descriptions["FRAME_SKIP"], key="-FRAME_SKIP_SLIDER-"),
                         sg.Input(default_text=str(int(FRAME_SKIP)), enable_events=True, tooltip=detector.arg_descriptions["FRAME_SKIP"], key="-FRAME_SKIP-")],
                        [sg.Text('max frames no crop'), 
                         sg.Slider((0,2000), resolution=1, default_value=int(MAX_FRAMES_NO_CROP), enable_events=True, orientation='h', tooltip=detector.arg_descriptions["MAX_FRAMES_NO_CROP"], key="-MAX_FRAMES_NO_CROP_SLIDER-"), 
                         sg.Input(default_text=str(int(MAX_FRAMES_NO_CROP)), enable_events=True, tooltip=detector.arg_descriptions["MAX_FRAMES_NO_CROP"], key="-MAX_FRAMES_NO_CROP-")],
                        [sg.Text('crf'), 
                         sg.Slider((0,51), resolution=1, default_value=int(VIDEO_CRF), enable_events=True, orientation='h', tooltip=detector.arg_descriptions["VIDEO_CRF"], key="-VIDEO_CRF_SLIDER-"), 
                         sg.Input(default_text=str(int(VIDEO_CRF)), enable_events=True, tooltip=detector.arg_descriptions["VIDEO_CRF"], key="-VIDEO_CRF-")],
                        [sg.Text('video preset'), 
                         sg.Input(default_text=str(VIDEO_PRESET), enable_events=True, tooltip=detector.arg_descriptions["VIDEO_PRESET"], key="-VIDEO_PRESET-")]  ]
    layout = [  row1, 
                [sg.Frame(layout=input_column, element_justification='center', vertical_alignment="top", title=''), 
                sg.Frame(layout=output_column, element_justification='center', vertical_alignment="top", title=''),
                sg.Frame(layout=processing_column, element_justification='center', vertical_alignment="top", title='')],
                [sg.Output(size=(100, 12))]    ]
    window = sg.Window('Image-Video-Detector', layout, no_titlebar=False, location=(0, 0))
    
    while True:
        event, values = window.read(timeout=1000)
        if event in ('Exit', sg.WINDOW_CLOSED):
            break
        
        #load variables from UI
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
        
        if event in ("-RUN-"):
            args = {  "INPUT_PATH": INPUT_PATH, 
                        "MASK_SAVE_PATH": MASK_SAVE_PATH, 
                        "OUTPUT_MEDIA_PATH": OUTPUT_MEDIA_PATH, 
                        "TEMP_PATH": TEMP_PATH, 
                        "DETECTION_TEXTS": DETECTION_TEXTS, 
                        "DO_CROP": DO_CROP,
                        "DETECT_THRESHOLD": DETECT_THRESHOLD, 
                        "CROP_SIZE_OFFSET": CROP_SIZE_OFFSET, 
                        "FRAME_SKIP": FRAME_SKIP, 
                        "MAX_FRAMES_NO_CROP": MAX_FRAMES_NO_CROP, 
                        "MODEL_NAME": MODEL_NAME, 
                        "VIDEO_CRF": VIDEO_CRF, 
                        "VIDEO_PRESET": VIDEO_PRESET   }
            args = removeEmptyFromDict(args)
            if("DETECTION_TEXTS" in args):
                args["DETECTION_TEXTS"] = [args["DETECTION_TEXTS"]]
            print("\nRunning detector with input arguments:\n"+str(args))
            detector.main(**args)
        
    window.close()
    
if __name__ == "__main__":
    main()
