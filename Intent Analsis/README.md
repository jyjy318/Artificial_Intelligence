# 1. 코드 설명
### ㄱ.  train.py
a. 학습데이터의 의도를 숫자화하고, 원 의도와 숫자화된 의도 정보를 2차원 배열 label에 저장함 <br/>
b. 학습 데이터 전처리 <br/>
   - text : 전처리할 텍스트 <br/>
   - okt : okt 객체를 반복적으로 생성하지 않고, 미리 생성후 인자로 받음.<br/>
   - remove_stopword : 불용어를 제거할지 선택하는 인자로 True를 사용.<br/>
   - stop_word : 불용어가 담긴 리스트<br/>

c. 학습데이터를 벡터로 변환 <br/>
d. 훈련 레이블을 벡터로 변환 <br/>
e. 모델학습 <br/>
   - 사용된 activation function : relu, relu, softmax <br/>
f. 학습된 모델 저장(h5파일) <br/>
   
### ㄴ. predict.py
a. train.py 에서 사용된 변수들을 import <br/>
b. 데이터 전처리 <br/>
c. string이 아니면 비어있는 값 추가( 비어있는 데이터에서 멈추지 않도록) <br/>
d. 사전에 학습된 h5추론파일 load <br/>
e. 모델에 대입, 예측값 갱성, 결과txt파일 생성 <br/>

### ㄷ. train 데이터와 dev 데이터의 정확도
test.txt. 의 accuracy<br/>
![test acc](https://user-images.githubusercontent.com/66713459/103552401-351a5300-4eef-11eb-8820-d0eb859e2720.jpg) <br/>
accuracy : 96% <br/>
validation accuracy : 약 88%<br/>

prediction.py의 
~~~
test = pd.read_table('test.txt', header = None) 부분에서 test.txt 대신 dev.txt를 불러오고, 
~~~
~~~
label_predict 함수에 

count = 0
for i in range(len(test_data)):
     if test[0][i] == PREDICT[i]:
        count += 1
print(count)
print("dev 정확도: ", format(count / len(test_data))) 

코드를 추가하여 dev 데이터에 대한 정확도를 구해본 결과
~~~

![wjdghkreh](https://user-images.githubusercontent.com/66713459/103545325-bf10ee80-4ee4-11eb-88b4-7a79270682ee.jpg) <br/>
dev 데이터 정확도 : 약 64.34% <br/>

----------------------------------------------------------
# 2. docker 사용시 명령어 순서대로 나열
필요한 모듈 : 첨부된 requirements.txt 사용
docker 사용을 가정하고 명령어를 순서대로 입력
~~~
apt-get install openjdk-8-jdk python-dev 
~~~
~~~
apt-get install python3-dev
~~~
~~~
pip install -r requirements.txt
~~~
~~~
python prediction.py
~~~
