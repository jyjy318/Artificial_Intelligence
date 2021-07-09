# Bone Classification
1. 프로젝트 목표<br/>
  : 사전에 라벨링이 된 골절이 된 뼈와 골절이 되지 않은 뼈 X-Ray 사진을 학습하여 Test 이미지가 주어졌을 때 해당 사진이 골절이 된 뼈사진인지, 골절이 되지 않은 뼈 사진인지 판단</br>

2. 데이터 분할 <br/>
: 처음 제공된 Trian 데이터에서 20%를 Validation데이터로 설정해주었습니다. 여기서 검증 데이터인 Validation데이터는 모델의 성능을 평가하기 위함이고, Train Data와의 중복을 피하였습니다. 또한 제공된 Test데이터를 통해 해당 모델의 성능을 측정할 수 있습니다. 즉, Train Data를 상요하여 모델을 만든 뒤, Validation Set으로 최종모델을 선택하였고, 최종 모델에서 예상되는 성능을 파악하기 위해 Test Data를 사용하게 됩니다. 데이터 Split의 과정을 그림으로 보면 다음과 같습니다.<br/>

3. 사용할 모델 : CNN<br/>
  ㄱ. CNN이란 <br/>
   : CNN은 데이터로부터 자동으로 특징을 학습하는 대표적인 모델입니다. 사람의 VIision정보 처리 방식을 흉내낸 것으로, 특히 이미지 인식과 분류에서 탁월한 성능을 냅니다. CNN의 학습 알고리즘은 다음과 같습니다. 입력과 가까운 층에서는 가장자리, 곡선과 같은 저수준 특징을 학습합니다. 점차 높은 층으로 올라갈수록 질감, 물체의 일부분과 같은 고수준의 특징을 인식하게 됩니다. 출력층에서는 물체의 종류를 인식하는 복잡한 추론을 수행하게 됩니다. <br/>
  ㄴ. CNN의 층 구성<br/>
    a. 컨볼루션층 : 모든 데이터가 하나의 신경층에서 다른 신경층으로 전파된다는 의미입니다. 이 층에서는 훈련 – 입력 – 입력층 – 은닉층 – 출력층의 과정을 거치게 됩니다. 컨볼루션 층에서는 작은 이미지 영역인 패치를 큰 이미지 위에 돌려가며 각각 다른 특징 활성값을 얻습니다. 이 패치는 특징을 감지한다는 점에서 특징 추출기 또는 커널필터라고 불리기도 합니다. <br/>
    b. pooling 층 : pooling은 학습시간을 줄이면서 이미지의 구성 요소의 위치 변화에 더 잘 대응할 수 있도록 해주는 과정입니다. 일반적으로 사용되는 것은 max pooling이며 주변 영역의 추론 결과값 중 최댓값만을 상위층으로 보내줍니다. 더불어 분류 작업에 유리한 불변성질을 얻을 수 있는 장점 또한 있습니다. <br/>
    c. FC층(Fully Connected Layer) : 최종적인 분류작업을 담당 <br/>
    
4. 학습전 데이터 전처리 과정 <br/>
  : Train과 Validation에 사용되는 데이터의 양은 총 177장이었기에 데이터양이 적어 정확도가 높지 않을것이라고 판단하였습니다. 그렇기에 모델을 생성하고 학습하기 전 데이터 전처리 과정에서 Data augmentation과정이 필요하다고 생각하였습니다. Data augmentation은 이미지를 사용할 때마다 임의로 변형을 가함으로써 마치 훨씬 더 많은 이미지를 보고 공부하는 것과 같은 학습 효과를 냅니다. 이는 결과적으로 overffiting 즉 모델이 학습 데이터에만 맞춰지는 것을 방지하고, 새로운 이미지도 잘 분류할 수 있게 해줍니다. 저는 Data의 양을 늘리기 위해 각각의 이미지들을 상하반전, 좌우반전 하였고, 각각 90도, 180도, 270도 회전하였습니다. 이 과정에서 이미지 데이터를 늘리고 Validation으로 할당하면 안되므로, 미리 분류된 Train 과 Validation 데이터를 각각 augment해주었습니다.  <br/>
  
5. Train데이터와 Validation데이터 준비하기<br/>
  : 지금부터는 모델을 생성하기 전에 필요한 데이터들을 준비하는 과정을 진행하겠습니다. 데이터들은 사전에 Normal 과 Fracture로 라벨링이 되어있으므로, 구분해야 할 class또한 Normal 과 Fracture 두가지 입니다.이미지를 조금 더 정확하게 학습하기 위하여 crop을 사용하였습니다. Crop은 이미지의 주변부를 잘라내는 것으로 주변에 적혀진 글씨를 제거하고, 중앙에 위치한 이미지 데이터들을 더 정확하게 학습시킬 수 있도록 도와줍니다. 이 과정이 모두 끝나면 이미지가 들어있는 경로를 모두 불러와서 Size를 조정해주고, 다차원 배열로 지정을 해줍니다. 그리고 이것을 Data_loader을 사용하여 load해줍니다. Validation도 같은 과정을 반복합니다. 데이터가 load가 잘 되었는지 이미지를 불러와서 확인해볼 수 있습니다. 이 과정이 끝나면 모델 생성 및 학습전까지의 모든 과정이 끝났다고 할 수 있습니다.<br/>
  
6. Build Convolution Neural Net(CNN모델 만들기)<br/>
![1](https://user-images.githubusercontent.com/66713459/125083576-89eb9000-e103-11eb-9dbd-c11b09760493.png)
: 제가 만든 CNN모델은 다음과 같습니다<br/>
    위의 그림은 최종 모델의 Block Diagram입니다.
    학습의 결과가 좋은 모델을 만들기 위해서 다양한 파라미터값을 변경하였습니다. 채널수를 늘리고, 중간중간에 Pooling layer을 삽입하였습니다. 또한 fully connected에서는 Dropout을 적용하여 overfitting을 방지하였습니다. 최종적으로 45*45*64의 결과를 얻었고 이 결과로 fully connected layer과정을 수행하였습니다. 
    해당 수행과정과 상세 결과는 7번 학습 부분에 상세하게 기입하였습니다. <br/>

7. 학습 <br/>
  : 최적의 성능을 가진 모델을 찾기 위해서 모델의 unit과 파라미터를 여러 번 바꾸고, epoch을 바꿔보며 해당과정을 수행해보았습니다. 제가 적용했던 모델들과 overfitting, underfitting을 어떻게 해결하였는지 설명하겠습니다. 
	처음 주어진 reference코드에서는 Train Accuracy와 Validation Accuracy는 각각 약 50%을 기록하였습니다. <br/>
1)Model 1 : Data augmentation 수행 <br/>
Augmentation를 수행한 후 Train Accuracy와 Validation Accuracy는 각각 약 69%와 62%로 높아졌습니다. 여기서 데이터 양을 늘렸을 때 정확도가 소폭 증가한다는 것을 알 수 있었습니다. <br/>
![2](https://user-images.githubusercontent.com/66713459/125083589-8e17ad80-e103-11eb-808b-f6b2a98c0f29.png) <br/>
그러나 Data Agumentation 과정을 거친다 하더라도 데이터의 underfitting은 피할 수 없었습니다. 또한 학습과정에서 중잔중간 과정에서 일어나는 overfitting도 피할 수 없었습니다. 
그렇기에 저는 underfitting을 방지하기 위해 학습모델의 layer을 추가하였습니다. 또한 training과 validation이 잘 수렴하는 지점으로 epoch을 줄였습니다. <br/>

2) Model2 : layer을 추가한 모델  <br/>
이 과정에서는 모델에 있는 layer을 추가하여 다시 학습을 시켜보았습니다. 동시에 epoch을 37로 줄여보았습니다.  <br/>
![3](https://user-images.githubusercontent.com/66713459/125083607-9243cb00-e103-11eb-9ee7-cf25c9eda7d0.png) <br/>
다음과 같은 결과를 얻을 수 있었습니다. 
앞선 모델과의 차이점은 Train Accuracy가 증가하였다는 사실입니다. 그러나, 이 과정에서 Validation Accuracy에는 차이가 거의 생기지 않았고 중간중간 Accuracy가 떨어지고, Train Accuracy와 차이가 점점 커지는 overfitting의 문제점이 더욱 심각해졌음을 알 수 있었습니다. 또한 데이터의 학습시간이 길어지는 단점도 발생하였습니다. <br/>

3) Model3 : model내 유닛수정, dropout반영<br/>
 :  이번 모델에서는 model에서 layer을 추가하기 보다, 모델 내 unit을 수정하고, dropout을 반영해보았습니다. 앞서 발생한 overfitting을 방지하기 위하여 Dropout을 수행하였고, dropout은 0.3으로 지정해주었습니다. <br/>
 ![4](https://user-images.githubusercontent.com/66713459/125083619-953ebb80-e103-11eb-8372-39717506872a.png) <br/>
 다음은 model의 unit을 수정하고, dropout을 반영하여 학습한 결과입니다. 앞선 모델들보다 overfitting이 줄었음을 확인할 수 있고, Valid Accuracy또한 소폭 증가하였음을 알 수 있었습니다. 해당 모델의 epoc을 조정하며 여러 번 돌려 Train Accuracy와 Valid Accuracy가 가장 수렴하는 지점을 찾아 모델을 새로 학습해보았습니다.<br/>
 
  4) Model4 : Epoch조정<br/>
  ![5](https://user-images.githubusercontent.com/66713459/125083628-97087f00-e103-11eb-90f1-5c901e38e7eb.png)<br/>
  다음은 Model3에서 epoch을 조절하여 학습시킨 결과입니다. 앞선 과정보다 overfitting이 훨씬 줄어들었고, Train Accuracy, Valid Accuracy가 모두 소폭 상승하였으며 큰 차이를 보이지 않았습니다. 
그러나 Underfitting문제는 지속되었습니다.<br/>

8. 결과 <br/> 
저는 이렇게 다양한 파라미터를 변경시켜가며 다양한 모델들을 학습시켜보았습니다. 그 결과 각각의 파라미터를 변경할 때 어떤 변화를 가져오는지 확인할 수 있었고, underfitting과 overfitting이 일어났을 때, 어떤 작업을 해야 방지할 수 있는지 직접 경험해볼 수 있었습니다. 저는 최종 model로 model4를 선택하게 되었습니다. 우선은 overfitting 발생이 현저히 떨어졌음을 알 수 있었고, 비록 70%의 정확도라는 다소 낮은 정확도를 나타내지만, 최종 model성능을 나타내는 Validation Accuracy가 가장 높았기 때문입니다. 해당 model에서 나타난 underfitting의 문제점은 데이터 수집, layer 추가 조정 등의 방법을 통해 극복할 수 있을것으로 생각합니다. <br/>

9. Test 에서의 적용 <br/> 
  : 학습을 통해 생성된 model을 test코드에 넣어서 확인해보는 과정입니다. Model에 있는 가중치값을 모두 사용하기 위해 cnn model 생성 내용을 모두 가져오고, model을 탑재하여 test하게 됩니다. Test 데이터는 총 90개가 있고 0은 normal, 1은 fracture로 추론 결과가 나오게 됩니다. 저는 이 부분을 csv로 바로 저장하기 위해 각각의 값들을 list에서 numpy로, numpy에서 dataframe으로 변환하여 csv에 자동으로 저장할 수 있도록 코드를 수정하였습니다. 생성된 csv의 1열에는 test파일의 이름이, 2열에는 추론된 결과가 0과 1의 label로 저장되어있음을 확인할 수 있습니다. <br/>
  
10. Term Project를 통하여 느낀점 <br/> 
  : 한학기동안 패턴인식과 머신러닝 이라는 과목에서 배운 이론적인 내용들을 실제 CNN모델에 탑재해보고, 직접 파라미터들을 수정해나가면서 최적의 결과를 도출하기 위해 layer등을 추가해보며 각각의 과정이 결과에 어떤 결과를 미치는지 직접 확인해볼 수 있었습니다. 









