# others
import cv2
from random import randint
import time
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import numpy as np
from PIL import Image
import mode

# YOLOv8
from ultralytics import YOLO
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils import ops
from copy import deepcopy
import matplotlib.pyplot as plt
import argparse

# gaze
import argparse, os
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mymodel.gaze.model import ModelSpatial # type: ignore
from mymodel.gaze.utils import imutils, evaluation # type: ignore
from mymodel.gaze.config import * # type: ignore

#dbデシベル
from moviepy.editor import AudioFileClip
import librosa
import soundfile as sf

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from mymodel.wav_split import cut_wav # type: ignore
from mymodel.write_wav_result import write_wav_result # type: ignore
from reazonspeech.nemo.asr import transcribe, audio_from_path, load_model # type: ignore


emotion_labels = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad","surprised", "unknown"]

# protoFile : to generate code that can read and write the data in the programming language of your choice
protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"
nPoints = 18
# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16]]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

pose_colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

def _pt_is_in_box(pt, box):
    _is_in = True
    box_min_x = min(box[0], box[2])
    box_min_y = min(box[1], box[3])
    box_max_x = max(box[0], box[2])
    box_max_y = max(box[1], box[3])

    if  box_min_x > pt[0] or box_max_x < pt[0]:
        _is_in = False 
    if  box_min_y > pt[1] or box_max_y < pt[1]:
        _is_in = False 
    return _is_in

def getKeypoints(probMap, threshold=0.1):

    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []

    #find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


# Find valid connections between the different joints of a all persons present
def getValidPairs(output,frameWidth,frameHeight,detected_keypoints):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid

        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
            # print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs



# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs, invalid_pairs,keypoints_list):
    # the last number in each row is the overall score
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score
                    print("2",len(keypoints_list))
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints

def compute_head_tall(head, height):
    # Detection adult or child from keypoints ratio
    #1.4 is constants for fine tuning
    detect_ratio = (head*1.4/height)
    detect_ratio = round(detect_ratio, 3)
    return detect_ratio

# detect==================================================================

def preprocess(img, size=640):
        img = LetterBox(size, True)(image=img)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)  # contiguous
        img = torch.from_numpy(img)
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img.unsqueeze(0)

def postprocess(preds, img, orig_img,confthres,iouthres):
    preds = ops.non_max_suppression(preds,
                                    conf_thres=confthres, #confidenceがこれ以上大きれば描画
                                    iou_thres=iouthres, #ボックスがフィルタアウトされるIoUしきい値
                                    classes = [0],
                                    agnostic=False,
                                    max_det=100)

    for i, pred in enumerate(preds):
        shape = orig_img.shape
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

    return preds

def drow_bbox(pred, names, annotator):
    for *xyxy, conf, cls in reversed(pred):
        c = int(cls)  # integer class
        label =  f'{names[c]} {conf:.2f}'
        annotator.box_label(xyxy, label, color=colors(c, True))

# ==================================================================

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image
def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution))) # type: ignore
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)

def expand_box(head_box,width,height,expand_size):
    expand_head_box = head_box
    expand_head_box[0] = head_box[0] - expand_size if head_box[0] > expand_size else 0
    expand_head_box[1] = head_box[1] - expand_size if head_box[1] > expand_size else 0
    expand_head_box[2] = head_box[2] + expand_size if width - head_box[2] > expand_size else width
    expand_head_box[3] = head_box[3] + expand_size if width - head_box[3] > expand_size else height
    return expand_head_box

def expand_box_ratio(head_box,width,height,expand_ratio):
    '''
    expand_ratio:元のhead_boxのwidth,heightのどれくらいを増やすか
    '''
    expand_head_box = head_box
    box_w = head_box[2]- head_box[0]
    box_h = head_box[3]- head_box[1]
    expand_size_w = int(box_w * expand_ratio)
    expand_size_h = int(box_h * expand_ratio)
    expand_head_box[0] = head_box[0] - expand_size_w if head_box[0] > expand_size_w else 0
    expand_head_box[1] = head_box[1] - expand_size_h if head_box[1] > expand_size_h else 0
    expand_head_box[2] = head_box[2] + expand_size_w if width - head_box[2] > expand_size_w else width
    expand_head_box[3] = head_box[3] + expand_size_h if width - head_box[3] > expand_size_h else height
    return expand_head_box

def get_angle(x0,y0,x1,y1,x2,y2):
    # 150度の視野に対象のbboxが入っているかを確認
    #角度の中心位置
    # x0,y0
    #方向指定1
    # x1,y1
    #方向指定2
    # x2,y2

    #角度計算開始
    vec1=[x1-x0,y1-y0]
    vec2=[x2-x0,y2-y0]
    absvec1=np.linalg.norm(vec1)
    absvec2=np.linalg.norm(vec2)
    inner=np.inner(vec1,vec2)
    cos_theta=inner/(absvec1*absvec2)
    theta=math.degrees(math.acos(cos_theta))
    return theta


def detect(opt):
    # out, source, yolo_weights, = opt.output, opt.source, opt.yolo_weights
    out, source, confthres, iouthres, device = opt.output, opt.source, opt.confthres, opt.iouthres, opt.device
    # mode================================================================
    is_gaze = opt.gaze
    is_count_person = opt.person_count
    is_to_csv = opt.to_csv
    is_db = opt.db
    if is_to_csv == "True":
        is_gaze = "True"
        is_count_person = "True"


    # for all =====================================================================
    video_capture = cv2.VideoCapture(source)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    all_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
    writer = cv2.VideoWriter(out, fmt, 10, (width, height)) # ライター作成 10fps設定、もともとは30fps
    frame_number = 1
    time_fromStart = 0
    print(width,height,all_frame_count,fps)
    # ======================================================================

	# GPU check
	# 
	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device check")
    print(device)

    # for person_detection=========================================================
    person_model = YOLO("best.pt")
    # model = YOLO("mymodel/yolov8n-pose.pt")
    if device == "gpu":
        person_model.to('cuda:0')
    person_color = (0, 255, 0)
    
    # for openpose=========================================================
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    if device == "cpu":
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
    elif device == "gpu":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")
    
    # Fix the input Height and get the width according to the Aspect Ratio
    inHeight = 368
    inWidth = int((inHeight/height)*width)
    threshold = 0.1
    # A child is considered to be a child when the ratio of head length to height is 18% or more
    CHILD_THRESHOLD = 0.18

    # for gaze==============================================================
    if is_gaze == "True":    
        # チューニングしたやつ
        face_model = YOLO("mymodel/yolov8face2.pt")
        if device == "gpu":
            face_model.to('cuda:0')
        # set up data transformation
        test_transforms = _get_transform()

        model = ModelSpatial()
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.model_weights, map_location=torch.device('cpu') )
        pretrained_dict = pretrained_dict['model']
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        if device == "gpu":
            model.cuda()
        model.train(False)

        # 視野角は150ど
        rotation_angle = 75
        # そのなかで注目してるであろう30ど
        rotation_angle_default = 15
        default_diameter = 200
    face_color = (200,0,0)
    
    # for db=================================================================
    if is_db == "True":     
        # 動画からAudioFileClipオブジェクトを生成
        audio = AudioFileClip(source) 
        wav_path = source.replace('.mp4', '.wav')
        # .wavファイルとして保存
        audio.write_audiofile(wav_path)
        y,sr = librosa.load(wav_path)
        #波形の生成
        rms=librosa.feature.rms(y=y) #RMSを計算　
        # デシベルを表示---------------------------------
        # 大きいほど音は大きい．最小は-100.
        # 音圧レベル (dB) = 20 * log10(RMS振幅 / 基準振幅)
        db=librosa.amplitude_to_db(rms) #dBを計算
        time=librosa.times_like(db,sr=sr) #時間軸の生成
        time_fromStart_db = 0
        time_num = 0
        db_atThisSecond = 0
        db_max_atThisSecond = -100
        db_perSecond = []
        db_maxPerSecond = []
        for i,t in enumerate(time):
            
            if time_fromStart_db <= t < time_fromStart_db + 1:
                db_atThisSecond += db[0][i]
                if db[0][i] > db_max_atThisSecond:
                    db_max_atThisSecond = db[0][i]
                time_num += 1
            else:
                db_ave_atThisSecond = db_atThisSecond / time_num
                db_perSecond.append(db_ave_atThisSecond)
                db_maxPerSecond.append(db_max_atThisSecond)
                db_atThisSecond = 0
                time_num = 0
                db_max_atThisSecond = -100
                time_fromStart_db += 1
        
        # 感情分析
        wave_file_list = cut_wav(wav_path, 5)
        emotion_list = []
        emotion_per_second = {label: [] for label in emotion_labels}
        
        for cut_wav_path in wave_file_list:
            inference_pipeline = pipeline(
                task=Tasks.emotion_recognition,
                model="iic/emotion2vec_base_finetuned", model_revision="v2.0.4")
            rec_result = inference_pipeline(cut_wav_path, output_dir="./output", granularity="utterance", extract_embedding=False)
            emotion_list.append(rec_result)
            scores = rec_result[0]["scores"]
            for key, value in zip(emotion_labels, scores):
                emotion_per_second[key].append(value)
                        
        trans_model = load_model()
        audio = audio_from_path(wav_path)
        ret = transcribe(trans_model, audio)
        
        write_wav_result(opt, ret, emotion_list, opt.output_audio_result)
        
        

# for 注目度----------------------------------------------- 
    # あるidの人が見られている総時間
    personbeingWatchedTime = defaultdict(int)
    # あるidの人がカメラの画角のなかにいる総時間
    personbeingWTime = defaultdict(int)
    # 1秒あたりの注目度を時系列でまとめた配列 
    degreeofAttentionatSecond = []
    # ある1秒での注目度の合計
    degreeofAttentionatthissec = 0
    # いつ，どのIDがみられたか
    id_attention = defaultdict(list)
    # ある一秒で，あるIDの人がみられるか
    id_attention_at_this_time = defaultdict(int)
# for 目配せど----------------------------------------------- 
    faceWatchingTime = defaultdict(int)
    facebeingTime = defaultdict(int)
# for 写っているID(人物)----------------------------------------------- 
    # いつ，どのIDがいるか
    id_exist = defaultdict(list)
    # ある一秒で，あるIDの人がいるか
    id_exist_at_this_time = defaultdict(int)
     # 1秒あたり写っている人の数を時系列でまとめた配列 
    personatSecond = []
    # ある1秒での写っている人の数の合計
    personcountatthissec = 0
    #ある人物が大人か子供か
    personIsChild = {}
    idBox = {}

    frame = 0

    # 動画をフレームごとに処理()
    while video_capture.isOpened():
        # フレームを1枚ずつ読み込む
        ret, img = video_capture.read()

        if frame_number % fps < 1:
            id_attention[time_fromStart] = id_attention_at_this_time
            id_exist[time_fromStart] = id_exist_at_this_time
            time_fromStart += 1
            # ある一秒で，あるIDの人がみられるか：毎秒初期化，毎秒追加
            id_attention_at_this_time = defaultdict(int)
            # ある一秒で，あるIDの人がいるか：毎秒初期化，毎秒追加
            id_exist_at_this_time = defaultdict(int)

        # 動画の再生が終了した場合、ループを抜ける
        if not ret:
            break
        frame += 1

        origin = deepcopy(img)
        output_img = deepcopy(img)

        # track person===============================================================================
        # resuresults_personlts = person_model.track(img,tracker = "bytetrack.yaml",persist=True,conf=confthres,iou=iouthres,classes=[0])
        results_person = person_model.track(img,persist=True,conf=confthres,iou=iouthres,classes=[0])

        if is_count_person == "True":
            personcountatthissec += len(results_person[0])
            if frame_number % fps < 1:
                avePersonCountatSecond = int(personcountatthissec / fps)
                personatSecond.append(avePersonCountatSecond)
                personcountatthissec = 0
            person_count = "person_count: " + str(len(results_person[0]))
            cv2.putText(output_img, person_count, (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3, cv2.LINE_AA)
        else: # count_personをしない時
            pass

        if is_gaze == "True":
            # track gaze===============================================================================
            results_face = face_model.track(img,persist=True,conf=confthres,iou=iouthres)
        # # Check if there are any detections
        # if results_face[0].boxes is not None:

            # 全ての人がみている場所の中心座標
            gaze_list = []

            with torch.no_grad():
                frame_raw = cv2pil(output_img)
                frame_raw = frame_raw.convert('RGB')
                width, height = frame_raw.size
                print(width,height)
                # Extract IDs if they exist
                ids = results_face[0].boxes.id.cpu().numpy().astype(int) if results_face[0].boxes.id is not None else []

                # Annotate frame with boxes and IDs
                for i, box in enumerate(results_face[0].boxes.xyxy.cpu().numpy().astype(int)):
                    id = ids[i] if len(ids)!=0 else None
                    head_box = [int(box[0]),int(box[1]),int(box[2]),int(box[3])]
                    
                    is_in_adult = False
                    head_pt1 = (head_box[0], head_box[1])
                    head_pt2 = (head_box[2], head_box[3])
                    for p_id, p_box in idBox.items():
                        expand_p_box = expand_box_ratio(p_box, width, height, 0.1)
                        if not personIsChild[p_id] and _pt_is_in_box(head_pt1, expand_p_box) and _pt_is_in_box(head_pt2, expand_p_box):
                            is_in_adult = True
                            break
                    
                    if not is_in_adult:
                        continue
                    
                    head_box = expand_box_ratio(head_box,width,height,0.6)
                    head = frame_raw.crop((head_box)) # head crop

                    head = test_transforms(head) # transform inputs
                    frame = test_transforms(frame_raw)
                    head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], width, height, resolution=input_resolution).unsqueeze(0)


                    head = head.unsqueeze(0)
                    frame = frame.unsqueeze(0)
                    head_channel = head_channel.unsqueeze(0)

                    if device == "gpu":
                        head = head.cuda()
                        frame = frame.cuda()
                        head_channel = head_channel.cuda()

                    # forward pass
                    raw_hm, _, inout = model(frame, head_channel, head)

                    # heatmap modulation
                    raw_hm = raw_hm.cpu().detach().numpy() * 255
                    raw_hm = raw_hm.squeeze()
                    inout = inout.cpu().detach().numpy()
                    inout = 1 / (1 + np.exp(-inout))
                    inout = (1 - inout) * 255

                    norm_map= np.array(Image.fromarray(obj=raw_hm, mode='F').resize(size=(width, height), resample=Image.BICUBIC))

                    gray_image = np.zeros((norm_map.shape[0], norm_map.shape[1]), dtype=np.uint8)
                    cv2.normalize(norm_map, gray_image, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    heatmap = gray_image

                    max_index = np.argmax(heatmap)
                    h = 1

                    max_index += 1
                    while max_index - width > 0:
                        max_index = max_index - width
                        h += 1
                    
                    # 視線の中心
                    h_point = (1.0 / (height -1)) * (h-1)
                    w_point = (1.0 / (width -1)) * (max_index-1)

                    gaze_center_h = h_point*height
                    gaze_center_w = w_point*width
                    head_w = int((head_box[2]+head_box[0])/2)
                    head_h = int((head_box[3]+head_box[1])/2)
                    gaze_list.append((head_w,head_h,gaze_center_h,gaze_center_w))

                    # ヒートマップ表示するならこれ---------------
                    # カラーマップを適用
                    # heatmap_color = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
                    # threshold,alpha = 200,0.7
                    # mask = np.where(heatmap<=threshold, 1, 0)
                    # mask = np.reshape(mask, (height, width, 1))
                    # mask = np.repeat(mask, 3, axis=2)
                    # marge = output_img*mask + heatmap_color*(1-mask)
                    # marge = marge.astype("uint8")   
                    # output_img = cv2.addWeighted(output_img, 1-alpha, marge,alpha,0)                      

                    original_output_image = output_img.copy()

                    center_h = height * h_point
                    center_w = width * w_point
                    diameter = int(np.sqrt(abs(center_h - head_h) ** 2 + abs(center_w - head_w)** 2))+40
                    # ラジアン単位を取得
                    radian = math.atan2((center_h - head_h),(center_w - head_w))
                    # ラジアン単位から角度を取得
                    degree = radian * (180 / math.pi)

                    # 人間の視野である150度をプロット
                    cv2.ellipse(output_img, (head_w, head_h), (diameter*10, diameter*10), degree,-rotation_angle, rotation_angle, face_color, thickness=-1)
                    output_img = cv2.addWeighted(original_output_image, 0.8, output_img, 0.2, 0)
                    # 距離が200は扇形の角度は15*2=30度にする
                    _rotation_angle = int(rotation_angle_default * default_diameter / diameter)
                    cv2.ellipse(output_img, (head_w, head_h), (diameter*10, diameter*10), degree,-_rotation_angle, _rotation_angle, face_color, thickness=-1)
                    output_img = cv2.addWeighted(original_output_image, 0.5, output_img, 0.5, 0)

                    # 顔の描画
                    cv2.rectangle(output_img, (head_box[0],head_box[1]), (head_box[2],head_box[3]), face_color, 2, cv2.LINE_AA)

                    if id is not None:
                        # 画面内に現れる総時間を計算
                        facebeingTime[id] += 1 
                        # 人をどれくらい見てるか
                        if results_person[0].boxes is not None:
                            for i, box in enumerate(results_person[0].boxes.xyxy.cpu().numpy().astype(int)):
                                boxx_center = int((box[2]+box[0])/2)
                                boxy_center = int((box[3]+box[1])/2)

                                theta = get_angle(head_w,head_h,gaze_center_w,gaze_center_h,boxx_center,boxy_center)
                                if theta <= rotation_angle:
                                # if ((box[0] - 50) < gaze_center_w < (box[2] + 50)) and ((box[1] - 50) < gaze_center_h < (box[3] + 50)) and head_box[1] < box[1]:
                                    faceWatchingTime[id] += 1
                                    break
                            degreeofWatch = int((faceWatchingTime[id]/facebeingTime[id])*100)

                        # cv2.putText(output_img, f"ID {id}", (head_box[0], head_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, face_color, 2)
                        cv2.putText(output_img, f"faceID {id} Watch {degreeofWatch}", (box[0], box[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,face_color, 2)

            # draw_person depend on gaze--------------------------------------------------------------------------
            # openpose--------------------------------------------------------------------------
            inpBlob = cv2.dnn.blobFromImage(origin, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)
            net.setInput(inpBlob)
            output = net.forward()
            detected_keypoints = []
            keypoints_list = np.zeros((0,3))
            keypoint_id = 0
            threshold = 0.1

            for part in range(nPoints):
                probMap = output[0,part,:,:]
                probMap = cv2.resize(probMap, (origin.shape[1], origin.shape[0]))
                keypoints = getKeypoints(probMap, threshold)
                # print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
                keypoints_with_id = []
                for i in range(len(keypoints)):
                    keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                    keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                    keypoint_id += 1

                detected_keypoints.append(keypoints_with_id)
            for i in range(nPoints):
                for j in range(len(detected_keypoints[i])):
                    cv2.circle(output_img, detected_keypoints[i][j][0:2], 5, pose_colors[i], -1, cv2.LINE_AA)
            # cv2.imshow("Keypoints",frameClone)
            valid_pairs, invalid_pairs = getValidPairs(output,width,height,detected_keypoints)
            personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs,keypoints_list)

            # Check if there are any detections
            if results_person[0].boxes is not None:
                
                # Extract IDs if they exist
                ids = results_person[0].boxes.id.cpu().numpy().astype(int) if results_person[0].boxes.id is not None else []

                # Annotate frame with boxes and IDs
                for i, box in enumerate(results_person[0].boxes.xyxy.cpu().numpy().astype(int)):
                    id = ids[i] if len(ids)!=0 else None
                    this_person_color = person_color

                    # 画面内に現れる総時間を計算
                    if id is not None:  
                        personbeingWTime[id] += 1    
                        id_exist_at_this_time[id] = 1   
                        if id not in personIsChild.keys():
                            personIsChild[id] = False
                        
                        for n in range(len(personwiseKeypoints)):
                            if personwiseKeypoints[n][1] < 0 or personwiseKeypoints[n][10] < 0 or personwiseKeypoints[n][16] < 0 or personwiseKeypoints[n][17] < 0:
                                continue

                            neck = keypoints_list[int(personwiseKeypoints[n][1])]
                            # rhip = detected_keypoints[8]    
                            # rknee = detected_keypoints[9]
                            rankle = keypoints_list[int(personwiseKeypoints[n][10])]
                            r_ear = keypoints_list[int(personwiseKeypoints[n][16])]
                            l_ear = keypoints_list[int(personwiseKeypoints[n][17])]
                            
                            if _pt_is_in_box(neck, box) and _pt_is_in_box(rankle, box) and _pt_is_in_box(r_ear, box) and _pt_is_in_box(l_ear, box):
                                                                                            
                                head = l_ear[0] - r_ear[0]
                                p_height = (rankle[1] - neck[1]) + head
                                detect_ratio= compute_head_tall(head, p_height)
                                if detect_ratio > CHILD_THRESHOLD:
                                    personIsChild[id] = True                        
                        
                    # みられている場合は...色の変更，見られている時間を加算
                    for (face_x,face_y,center_h,center_w) in gaze_list:
                        # print(center_w,box[0],box[2])
                        boxx_center = int((box[2]+box[0])/2)
                        boxy_center = int((box[3]+box[1])/2)
                        theta = get_angle(face_x,face_y,center_w,center_h,boxx_center,boxy_center)

                        if theta <= rotation_angle:
                        # if ((box[0] - 50) < center_w < (box[2] + 50)) and ((box[1] - 50) < center_h < (box[3] + 50)) and face_y < box[1]:
                            this_person_color = (180,180,180)
                            if id is not None:
                                personbeingWatchedTime[id] += 1
                                id_attention_at_this_time[id] = 1
                            break
                    cv2.rectangle(output_img, (box[0], box[1]), (box[2], box[3]), this_person_color, 2)
                    if id is not None:
                        degreeofAttention = int((personbeingWatchedTime[id]/personbeingWTime[id])*100)
                        degreeofAttentionatthissec += degreeofAttention
                        
                        idBox[id] = box

                        if personIsChild[id]: # 元々child⇒person
                            cv2.putText(output_img, f"ID {id}: child  ATTENTION {degreeofAttention}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,this_person_color, 2)
                        else: # 元々adult⇒person
                            cv2.putText(output_img, f"ID {id}: adult", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,this_person_color, 2)

            # 一秒あたりの注目度の合計を求める
            if frame_number % fps < 1:
                aveDegreeofAttentionatSecond = int(degreeofAttentionatthissec / fps)
                degreeofAttentionatSecond.append(aveDegreeofAttentionatSecond)
                degreeofAttentionatthissec = 0
            
            
            for i in range(17):
                for n in range(len(personwiseKeypoints)):
                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                    if -1 in index:
                        continue
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    A = np.int32(keypoints_list[index.astype(int), 1])
                    cv2.line(output_img, (B[0], A[0]), (B[1], A[1]), pose_colors[i], 3, cv2.LINE_AA)

        elif is_gaze == "False": # gaze情報を表示しない時
            # draw_person depend on gaze--------------------------------------------------------------------------
            
            # Check if there are any detections
            if results_person[0].boxes is not None:
                # Extract IDs if they exist
                ids = results_person[0].boxes.id.cpu().numpy().astype(int) if results_person[0].boxes.id is not None else []

                # Annotate frame with boxes and IDs
                for i, box in enumerate(results_person[0].boxes.xyxy.cpu().numpy().astype(int)):
                    id = ids[i] if len(ids)!=0 else None
                    cv2.rectangle(output_img, (box[0], box[1]), (box[2], box[3]), person_color, 2)
                    if id is not None:
                        cv2.putText(output_img, f"ID {id}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, person_color, 2)

        else : # とりあえずgaze情報を表示しない時
            # draw_person depend on gaze--------------------------------------------------------------------------
            
            # Check if there are any detections
            if results_person[0].boxes is not None:
                # Extract IDs if they exist
                ids = results_person[0].boxes.id.cpu().numpy().astype(int) if results_person[0].boxes.id is not None else []

                # Annotate frame with boxes and IDs
                for i, box in enumerate(results_person[0].boxes.xyxy.cpu().numpy().astype(int)):
                    id = ids[i] if len(ids)!=0 else None
                    cv2.rectangle(output_img, (box[0], box[1]), (box[2], box[3]), person_color, 2)
                    if id is not None:
                        cv2.putText(output_img, f"ID {id}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,person_color, 2)



        # OpenCVで表示＆キー入力チェック
        # cv2.imshow("YOLOv8 Tracking", output_img)
        writer.write(output_img)

        # 'q'キーが押された場合、ループを抜ける
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
        frame_number += 1

    # 使用したリソースを解放する
    #
    #video_capture.release()
    #writer.release()
    #cv2.waitKey(1)
    #cv2.destroyAllWindows()
    #cv2.waitKey(1)

    # if is_count_person == "True":    
    #     # データをグラフにプロット
    #     plt.plot(personatSecond)
    #     # グラフのタイトル
    #     plt.title('person count at sec')
    #     # X軸およびY軸のラベル
    #     plt.xlabel('time from start')
    #     plt.ylabel('the number of person')
    #     # グラフを表示
    #     plt.show()

    # デシベルのcsvと図を保存
    if is_db == "True":   
        data = {'max_db': db_maxPerSecond,
                'ave_db': db_perSecond,
                }
        df = pd.DataFrame(data)
        df_t = df.T
        df_t.to_csv(opt.output_db_csv, index=True) 

        plt.xlabel("Time(s)")
        plt.ylabel("dB")
        plt.plot(time,db[0])
        plt.savefig(opt.output_db_png)  

        df = pd.DataFrame(emotion_per_second)
        df_t = df.T
        df_t.to_csv(opt.output_emotion_csv, index=True) 


    if is_to_csv == "True":
        # defaultdictを通常の辞書に変換
        regular_dict = {}
        for key, inner_dict in id_exist.items():
            regular_dict[key] = dict(inner_dict)
        # 辞書をPandas DataFrameに変換
        df = pd.DataFrame(regular_dict).fillna(0).astype(int)
        # 列の値をソート
        df = df.reindex(sorted(df.columns), axis=1)
        # 行のインデックスをソート
        df = df.reindex(range(df.index.min(), df.index.max() + 1), fill_value=0)
        df_index = []
        for row in range(len(df)):
            df_index.append("person_" + str(row))
        df = df.set_axis(df_index, axis=0)
        person_id_num = len(df)
        # # CSVファイルへの書き込み
        df.to_csv(opt.output_personid_csv, index=True)


        # defaultdictを通常の辞書に変換
        regular_dict = {}
        for key, inner_dict in id_attention.items():
            regular_dict[key] = dict(inner_dict)
        # 辞書をPandas DataFrameに変換
        df = pd.DataFrame(regular_dict).fillna(0).astype(int)
        # 列の値をソート
        df = df.reindex(sorted(df.columns), axis=1)
        # 行のインデックスをソート
        df = df.reindex(range(df.index.min(), df.index.max() + 1), fill_value=0)
        df_index = []
        for row in range(len(df)):
            df_index.append("注目度_ID_" + str(row))
        df = df.set_axis(df_index, axis=0)
        while len(df) < person_id_num:
            row += 1
            row_name = "注目度_ID_" + str(row)
            df.loc[row_name] = 0
        # # CSVファイルへの書き込み
        df.to_csv(opt.output_attentionid_csv, index=True)            

        # DataFrameの作成
        df = pd.DataFrame({
            'person_count': personatSecond,
        })
        df = df.T
        # # CSVファイルへの書き込み
        df.to_csv(opt.output_personcount_csv, index=True)   


# 実行例
# track_person_yolov8.py --source <<動画のパス>> --output <<出力先のパス>>
# track_person_yolov8.py --source "D:/train_code/yolov8_0203/data/MB/movie_test.mp4" --output "D:/train_code/yolov8_0203/data/kojima/1.mp4"

if __name__ == '__main__':
    t1 = time.localtime()
    start_time = time.strftime("%H:%M:%S", t1)
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str, default="data/movie_test.mp4", help='source')
    
    parser.add_argument('--output', type=str, default="output/test.mp4", help='outputpath')  # output folder  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--confthres', type=float, default=0.15, help='object confidence threshold')
    parser.add_argument('--iouthres', type=float, default=0.6, help='IOU threshold for NMS')
    #parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    
    # gaze-------------------------------------
    parser.add_argument('--model_weights', type=str, help='model weights', default='mymodel/gaze/model_demo.pt')

    # db---------------------------------------
    parser.add_argument('--output_db_csv', type=str, default="output/db.csv", help='outputpath')  # output folder
    parser.add_argument('--output_emotion_csv', type=str, default="output/emotion.csv", help='outputpath')  # output folder
    parser.add_argument('--output_db_png', type=str, default="output/db.png", help='outputpath')  # output folder
    parser.add_argument('--output_audio_result', type=str, default="output/audio_result.mp4", help='outputpath')  # output folder
    # tocsv-------------------------------------
    parser.add_argument('--output_personid_csv', type=str, default="output/person_id.csv", help='outputpath')  # output folder
    parser.add_argument('--output_attentionid_csv', type=str, default="output/attention_id.csv", help='outputpath')  # output folder
    parser.add_argument('--output_personcount_csv', type=str, default="output/personcount.csv", help='outputpath')  # output folder
    
    # mode-------------------------------------
    parser.add_argument('--gaze', type=str, help='do you need gaze information?', default='True')
    parser.add_argument('--person_count', type=str, help='do you need person-count information?', default='True')
    parser.add_argument('--to_csv', type=str, help='do you need csv information?', default='True')
    # 
    parser.add_argument('--db', type=str, help='do you need db csv and db graph?', default='False')
    parser.add_argument("--device", default="gpu", help="Device to inference on")


    

    args = parser.parse_args()
    detect(args)
    t2 = time.localtime()
    end_time = time.strftime("%H:%M:%S", t2)
    print(start_time)
    print(end_time)


# AI 検索する

# 説明

# 翻訳

