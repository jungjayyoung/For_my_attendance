# attendance를 video version으로!
# streaming version
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import os
import numpy as np
import torch
import joblib
from torchvision import datasets, transforms
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
from datetime import datetime
import time
from flask import Flask, render_template, Response, request
from flask import stream_with_context

## db 연동을 위한 패키지 추가
import pymysql

app = Flask(__name__)

# db 연동

#host: 접속할 DB 주소, port: RDBMS는 주로 3306 포트를 통해 연결됨, user: DB에 접속할 사용자 ID, passwd: 사용자 비밀번호,
#db: 사용할 DB 이름, charset: 한글이나 유니코드 데이터가 깨지는 것을 막기위한 인코딩 방식 utf8로 설정
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='639528',
    db='test_database',
    charset='utf8'
)

#커서 객체 생성:
#커서는 SQL 문을 실행하거나, 결과를 돌려받는 통로이다.
cursor = conn.cursor()

#연결한 데이터베이스에 새 테이블 생성, 만들어져 있지 않으면 생성하도록 한다.
cursor.execute("""CREATE TABLE IF NOT EXISTS Attendance
               (
               id int PRIMARY KEY not null AUTO_INCREMENT COMMENT '인덱스',
               num varchar(20) not null COMMENT '학번',
               major varchar(20) not null COMMENT '전공',
               name varchar(10) not null COMMENT '이름',
               time int not null COMMENT '출석시간',
               frequency float not null COMMENT '빈도',
               result varchar(20) not null COMMENT '출석',
               day TIMESTAMP not null DEFAULT CURRENT_TIMESTAMP COMMENT '출석일자',
               professor varchar(10) not null DEFAULT '배성일' COMMENT '교수님'
               );"""
                )
# 데이터 입력: 여기서 %s는 일반 문자열 포팅에 사용하는 %d,%ld,%c 등과는 다른 것이다.
# MySQL에서 이것을 Parameter Placeholder라고 하는데 문자열이건 숫자이건 모두 %s를 사용한다.
sql = 'INSERT INTO Attendance(num, major, name, time, frequency, result) VALUES (%s, %s, %s, %s, %s, %s);'




##############################
# attendance
class Person(object):
  def __init__(self,SN,major,name):
    self.SN= SN
    self.major = major
    self.name = name
    self.appear = 0
    self.percent = 0
    self.result = 0


person1 = Person(11111111,'기계공학과','IU')
person2 = Person(22222222,'전자공학과','JAEYOUNG')
person3 = Person(33333333,'전자공학과','JUNE')
person4 = Person(44444444,'화생공학과','KEY')
person5 = Person(55555555, '방송연예학', 'MINHO')
person6 = Person(66666666, '방송연예학', 'MYUNGHO')
person7 = Person(77777777, '방송연예학', 'ONEW')
person7 = Person(77777777, '방송연예학', 'TAEMIN')

person_list = [person1, person2, person3, person4, person5, person6, person7]

# anum : streaming에서 각 class 사람들이 인식된 frame의 개수
anum=[ 0, 0, 0, 0, 0, 0, 0, 0]
# 각 사람의 번호
aboard = {0:'IU',1:'JAEYOUNG',2:'JUNE',3:'Key',4:'MINHO',5:'MYUNGHO',6:'ONEW',7:'TAEMIN'}

###############################
t = time.localtime()


def Acheck_TR(anum, atime, person_list, framess, total_time, absence_per, attend_per):
    for i in range( len( anum ) ) :
        # frame/total_time : 시간 당 frame의 수
        # 인식되었던 시간(frame 단위였던 것을 time으로 변환)
        atime[ i ]=round( anum[ i ] / (framess/total_time) , 1 )
    for i in range( len( atime ) ) :
        # person_list의 appear 항목에 출석한 총 시간 넣어주기
        person_list[ i ].appear=atime[ i ]
    for i in range( len( anum ) ) :
        person_list[ i ].percent=round( anum[i]/framess * 100 , 1 )
        if person_list[i].percent > 100:
          person_list[i].percent = 100

    # 총 시간, 첫째자리까지 표현하기
    total_time_r = round(total_time,1)
    if total_time >=60 :
        total_time_r = "{}분 {}초".format(total_time_r//60,total_time_r%60)
    else :
        total_time_r = "{}초".format(total_time_r)
    print('\033[96m'+"총 시간: "+total_time_r+ '\033[0m')

    # percent로 출석과 출튀 여부 가리기
    for i in range(len(atime)):
        if person_list[i].percent >= attend_per:
            if person_list[i].result == '지각(late)':
                continue
            else:
                person_list[i].result = '출석(attend)'
        elif person_list[i].percent < attend_per and person_list[i].percent >= absence_per:
            person_list[i].result = '출튀(escape)'
        elif person_list[i].percent < absence_per:
            person_list[i].result = '결석(absence)'
            #person_list[i].result = '\033[31m' + '결석(absence)'+ '\033[0m'

    print('결석(absence) : percent < {}'.format(absence_per))
    print('출튀(escape) : {} < percent < {}'.format(absence_per, attend_per))
    print('출석(attend) : {} < percent'.format(attend_per))
    print( '\033[32m'+'|ㅡㅡ학번ㅡㅡ|ㅡㅡㅡ전공ㅡㅡㅡ|ㅡㅡ이름ㅡㅡ|ㅡ출석시간ㅡ|ㅡㅡ빈도ㅡㅡ|ㅡㅡ결과ㅡㅡ|' '\033[0m')
    for i in range( len( person_list ) ) :
        # 총 시간, 첫째자리까지 표현하기
        appear_time_r = round(person_list[i].appear)
        if person_list[i].appear >= 60:
            appear_time_r = "{}분 {}초".format(appear_time_r//60, appear_time_r%60)
        else:
            appear_time_r = "    {}초".format(appear_time_r)

        print('  {}'.format(person_list[i].SN) + '   {}'.format(person_list[i].major) + '    {}'.format(
                person_list[i].name) + '   ' + appear_time_r + '       {}%'.format(
                person_list[i].percent) + '    {}'.format(
                person_list[i].result))
        #DB에 저장할 데이터 입력 id, num, major, name, time, frequency, result
        conn.ping()
        cursor.execute(sql, (person_list[i].SN, person_list[i].major, person_list[i].name, person_list[i].appear, person_list[i].percent, person_list[i].result))
    #데이터 베이스 반영
    conn.commit()
    #데이터 베이스 종료
    conn.close()


###############################

# 영상에 bounding box를 치고, 대상의 이름을 출력하도록 하는 함수들
def diag(x1, y1, x2, y2):
    return np.linalg.norm([x2 - x1, y2 - y1])


def square(x1, y1, x2, y2):
    return abs(x2 - x1) * abs(y2 - y1)


def isOverlap(rect1, rect2):
    x1, x2 = rect1[0], rect1[2]
    y1, y2 = rect1[1], rect1[3]

    x1_, x2_ = rect2[0], rect2[2]
    y1_, y2_ = rect2[1], rect2[3]

    if x1 > x2_ or x2 < x1_: return False
    if y1 > y2_ or y2 < y1_: return False

    rght, lft = x1 < x1_ < x2, x1_ < x1 < x2_
    d1, d2 = 0, diag(x1_, y1_, x2_, y2_)
    threshold = 0.5

    if rght and y1 < y1_:
        d1 = diag(x1_, y1_, x2, y2)
    elif rght and y1 > y1_:
        d1 = diag(x1_, y2_, x2, y1)
    elif lft and y1 < y1_:
        d1 = diag(x2_, y1_, x1, y2)
    elif lft and y1 > y1_:
        d1 = diag(x2_, y2_, x1, y1)

    if d1 / d2 >= threshold and square(x1, y1, x2, y2) < square(x1_, y1_, x2_, y2_): return True
    return False


def draw_box(draw, boxes, names, probs, min_p=0.89):
    font = ImageFont.truetype(os.path.join(ABS_PATH, 'arial.ttf'), size=22)

    not_overlap_inds = []
    for i in range(len(boxes)):
        not_overlap = True
        for box2 in boxes:
            if np.all(boxes[i] == box2): continue
            not_overlap = not isOverlap(boxes[i], box2)
            if not not_overlap: break
        if not_overlap: not_overlap_inds.append(i)

    boxes = [boxes[i] for i in not_overlap_inds]
    probs = [probs[i] for i in not_overlap_inds]
    for box, name, prob in zip(boxes, names, probs):
        if prob >= min_p:
            draw.rectangle(box.tolist(), outline=(255, 255, 255), width=5)
            x1, y1, _, _ = box
            text_width, text_height = font.getsize(f'{name}')
            draw.rectangle(((x1, y1 - text_height), (x1 + text_width, y1)), fill='white')
            draw.text((x1, y1 - text_height), f'{name}: {prob:.2f}', (24, 12, 30), font)

    return boxes, probs

def get_video_embedding(model, x):
    embeds = model(x.to(device))
    return embeds.detach().cpu().numpy()


def face_extract(model, clf, frame, boxes):
    names, prob, idx_list = [], [], []
    if len(boxes):
        x = torch.stack([standard_transform(frame.crop(b)) for b in boxes])
        embeds = get_video_embedding(model, x)
        idx, prob = clf.predict(embeds), clf.predict_proba(embeds).max(axis=1)
        names = [IDX_TO_CLASS[idx_] for idx_ in idx]
        print("names : {}".format(names))
        idx_list = list(set(idx.tolist()))
        print("idx_list : {}".format(idx_list))
    return names, prob, idx_list



# def preprocess_video(detector, face_extractor, clf, path, transform=None, k=3):
def preprocess_video(detector, face_extractor, clf, path, absence_per, attend_per, class_time, transform=None, k=3):
    if not transform: transform = lambda x: x.resize((1280, 1280)) if (np.array(x.shape) > 2000).all() else x

    anum = [0, 0, 0, 0, 0, 0, 0]

    framess = 0

    capture = cv2.VideoCapture(path)
    # 1012 juwoo
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_class_frame = class_time * fps
    late_num = round(total_class_frame * 0.1)


    # attend, 매 frame마다 진행하도록 loop
    while True:
        # 2. 받아온 capture를 ret와 frame으로 읽는다.
        ret, frame = capture.read()
        # fps = capture.get(cv2.CAP_PROP_FPS)
        #


        if ret:
            framess = framess + 1
            '''
            if framess % k != 0:
                pass
            else:
            '''
            # 위에 거 대신에 이걸 추가함.
            if framess == 0 or (framess + 1) % k == 0:
                frame = cv2.flip(frame, 1)  # 좌우반전(거울모드)
                iframe = Image.fromarray(transform(frame))
                # frame_draw = iframe.copy()

                try:
                    boxes, probs = detector.detect(iframe)
                    if boxes is None: boxes, probs = [], []
                    names, prob, idx = face_extract(face_extractor, clf, iframe, boxes)
                    idx2 = list(set(idx))
                    #################################
                    # anum이 1씩 증가하게 된다.
                    for idx_ in idx2:
                        anum[idx_] = anum[idx_] + 1

                    ## k << 매개변수화할 것
                    if (framess + 1) / k == late_num:
                        print("지각 여부 확인")
                        print("When frame is {} : {}".format(late_num, anum))
                        for j in range(len(anum)):
                            # 이때까지 인식이 안 되면, 그 사람의 class의 result를 '지각'으로 설정
                            if anum[j] <= 1:
                                person_list[j].result = '지각(late)'

                    #################################

                    frame_draw = iframe.copy()
                    draw = ImageDraw.Draw(frame_draw)

                    boxes, probs = draw_box(draw, boxes, names, probs)

                except:
                    pass

                frame_np = np.array(frame_draw)
                #cv2.imshow('Face Detection', frame_np)
                rett, buffer = cv2.imencode('.jpg', frame_np)
                frame_np = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_np + b'\r\n')
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if framess >= total_class_frame:
            break

    #capture.release()
    #cv2.destroyAllWindows()

    print(f'Total frames: {framess}')
    ######
    total_time = framess / fps
    framess = framess / k
    atime = [0, 0, 0, 0, 0, 0, 0]           #사람 숫자에 따라 바뀐어야한다.
    # Acheck_TR : Acheck_TR.py 참고하기
    Acheck_TR(anum, atime, person_list, framess, total_time, absence_per, attend_per)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(stream_with_context(preprocess_video(mtcnn, model, clf, 0, absence_per, attend_per, class_time)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/setTime')
def test():
    global class_time
    class_time = int(request.args.get("time",20))
    return "저장 성공"

##### main

if __name__ == '__main__':
    # Define image path
    ABS_PATH = 'F:/PycharmProjects/FinalProject'
    DATA_PATH = os.path.join(ABS_PATH, 'data')

    # Preparing data
    ALIGNED_TRAIN_DIR = 'F:/PycharmProjects/FinalProject/data/train_images_cropped'
    ALIGNED_TEST_DIR = 'F:/PycharmProjects/FinalProject/data/test_images_cropped'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'print Running on device: {device}')

    standard_transform = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    # augmentation 부분
    # 데이터의 양을 늘리는 부분으로 코랩에서 진행하지만,
    # transform에서 aug_mask가 변수로 들어가기 때문에 남겨두었다.
    aug_mask = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.15),
        A.RandomContrast(limit=0.5, p=0.4),
        A.Rotate(30, p=0.2),
        A.RandomSizedCrop((120, 120), 160, 160, p=0.4),
        A.OneOrOther(A.JpegCompression(p=0.2), A.Blur(p=0.2), p=0.66),
        A.OneOf([
            A.Rotate(45, p=0.3),
            A.ElasticTransform(sigma=20, alpha_affine=20, border_mode=0, p=0.2)
        ], p=0.5),
        A.HueSaturationValue(val_shift_limit=10, p=0.3)
    ], p=1)

    transform = {
        'train': transforms.Compose([
            transforms.Lambda(lambd=lambda x: aug_mask(image=np.array(x))['image']),
            standard_transform
        ]),
        'test': standard_transform
    }

    # Original train images
    # trainD : train_cropped image를 stadard_transform를 가한 것.
    trainD = datasets.ImageFolder(ALIGNED_TRAIN_DIR, transform=standard_transform)

    # Get named labels
    # 여기서 폴더에서 뽑아낸 이름이 저장이 된다.
    IDX_TO_CLASS = np.array(list(trainD.class_to_idx.keys()))
    CLASS_TO_IDX = dict(trainD.class_to_idx.items())

    # 학습한 모델 불러오기
    # 불러온 모델을  clf에 저장
    SVM_PATH = os.path.join(ABS_PATH, 'svm_final.sav')
    clf = joblib.load(SVM_PATH)

    standard_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])

    # mtcnn은 얼굴 탐지하는 함수
    mtcnn = MTCNN(keep_all=True, min_face_size=70, device=device)

    # model은 vggface2라는 서양인 9131명의 331만장의 사진을 학습시킨 모델
    model = InceptionResnetV1(pretrained='vggface2', dropout_prob=0.6, device=device).eval()

    # 지각의 기준이 될 frame.
    # 지각/출석/출튀/결석의 기준

    # 총 스트리밍할 시간, 매개변수로 받을 값
    # 시간 단위이다 frame으로 바꾸는 것은 preprocess_video 안에서 구현하였다.
    # class_time 이 지나면 자동으로 종료한다.
    class_time = 20

    # 지각의 기준은 preprocess_video 내부에서 0.1을 기준으로 결정
    # late_num = round(class_time * 0.1)
    # 결석/출석/지각의 퍼센티지 기준이다.
    absence_per = 40
    attend_per = 60


    # live streaming 실행
    print('Processing live stream: ')
    #preprocess_video(mtcnn, model, clf, 0, absence_per, attend_per, class_time)
    app.run(debug=True)

    '''
    ### video streaming version
    VIDEO_PATH = os.path.join(DATA_PATH, 'videos/')
    width, height = 640, 360

    mov1 = os.path.join(VIDEO_PATH, 'test_1.mp4')

    print('Processing mov1: ')
    preprocess_video(mtcnn, model, clf, mov1, late_num, absence_per, attend_per)

    '''