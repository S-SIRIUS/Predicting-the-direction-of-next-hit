# 클러스터링과 LSTM모델을 통한 다음 흥행작 시나리오 방향성 제시
## 1. 주제 선정 배경
### 1) 리뷰 분석의 목적
주제를 선정하기에 앞서서 리뷰분석의 목적에 대해 명확히 알고 있어야합니다. 우선 리뷰분석을 통해 현재 회사들은 브랜드와 상품에 대한 냉정한 피드백을 통해 소비자와 시장을 더 잘 이해하고 있습니다. 또한 상품기획/개선, 마케팅 전략 등을 올바른 방향으로 잡아가는 것에 리뷰데이터 분석을 이용하고 있습니다.
즉 리뷰분석을 회사의 비즈니스 전략에 반영하고 있습니다.

### 2) 주제선정
대부분의 사람들이 영화나 드라마를 시청하기 전에 리뷰데이터를 참조한다면, 이 요소는 흥행에 크게 관여하는 요소가 됩니다. 다음 드라마의 흥행에 영향을 준다면 역으로 흥행을 이 데이터로 예측할 수 있습니다. 또한 흥행에는 드라마의 설계도와 같은 시나리오가 중요합니다. 따라서 주제를 찾아낸 패턴으로 다음 흥행작 시나리오의 방향성을 제시하는 것으로 정하게 되었습니다.

### 3) 프로젝트 진행 방향
#### 가. 클러스터링 기법
데이터를 처음 보았을 때 데이터 셋에서 타겟이라고 할 만한 요소를 찾지 못하였습니다. 따라서 비지도학습으로 어떤 데이터인지 알아보기로 결정했습니다. 그 중 Kmeans알고리즘을 이용하였을 때 예상분석이 유의미할 것이라고 판단하였습니다.
Kmeans 알고리즘은 “군집 중심점이라는 특정한 임의의 지점을 선택해 해당 중심에 가장 가까운 포인트들을 선택하는 군집화 기법”입니다. 이 Kmeans 알고리즘이 데이터 셋에 유의미 할 것이라 판단한 이유는 만약 “good”, “character”, “watch”, “unique”, “seong-ki-hun”이라는 5개의 단어가 군집화 되어 있다면. 이 5개의 단어가 텍스트에서 가까운 의미를 가진다고 볼 수 있습니다. 따라서 성기훈이라는 캐릭터가 독특해서 보는 것에 재미를 주었다. 이런 식의 해석이 가능할 것이라고 생각하였습니다. 그러나 2가지 문제점이 생겼습니다. 첫번째는 모든 데이터가 가치가 있는 것인지의 문제였습니다. 두번째는 Kmeans 군집화를 해석하는 과정에서 인간의 생각이 크게 관여된다는 점이었습니다.

#### 나. 데이터의 가치
첫번째 문제를 해결하기 위해 생각해낸 것이 바로 4분위수였습니다. 어떤 값을 기준으로 상위 25%데이터만 추출해서 보면 그것은 그 데이터 내에서 좀 더 정확한 분석이 될 것이라 확신하였습니다. 그렇다면 이 데이터에서는 어떤 값이 기준이 될 수 있을지 생각해보았습니다. 그 결과 두가지 요소가 있었는데 첫번째는 ‘공감수’였고 두번째는 ‘공감수에 조회수를 나누고 100을 곱한 값’이었습니다.
이 중에서 후자를 기준으로 선택하였습니다. 그 이유는 최신자료의 경우 조회수가 적지만 조회수에 비해 공감이 많은 데이터가 있었기 때문입니다. 따라서 이것을 column으로 만들어서 이 값을 기준으로 상위 25% 데이터를 추출하기로 하였습니다.

#### 다. 인간의 해석
두번째 문제는 Kmeans를 해석하는 부분에서 인간의 생각이 들어간다는 점이었습니다. 따라서 LSTM이라는 딥러닝 모델을 적용하였습니다. LSTM은 sequence모델로 시계열 데이터나, 자연어를 처리하는 부분에서 쓰입니다. RNN과는 다르게 장기기억을 보존할 수 있다는 장점을 가진 덕에 sequence한 데이터를 보다 정확하게 예측할 수 있습니다. 따라서 텍스트로 학습을 시키고 어떤 단어를 초기값으로 입력하면 문장을 생성할 수 있는 모델입니다. 군집화 시킨 문서들을 LSTM에 넣어 학습시키고 군집의 단어들 중 가장 유의미한 단어를 LSTM에 넣어 문장을 생성하여 이를 통해 시나리오의 방향성을 제시할 수 있겠다고 확신하였습니다.

## 2. 데이터 프레임화 & 전처리
### 1) 데이터 프레임화
데이터 분석을 위해 텍스트 파일들을 데이터 프레임 형태로 변환하는 과정을 설명합니다. 먼저, 파일들의 경로를 지정하고, 모든 파일명을 all_files에 저장하여 경로를 포함시킵니다. 이 파일들은 read_table을 사용해 읽히고, 문자열 형태로 변환된 후 review_text 리스트에 저장됩니다. 파일명은 처리하여 filename_list에 저장되며, 이는 파일명을 기준으로 정렬하는 데 사용됩니다. 정렬 과정에서 인덱스가 섞이기 때문에, 인덱스를 재설정하고 기존 인덱스는 제거합니다. 이 과정을 통해 최종적으로 기본 데이터 프레임이 완성됩니다.

<img width="286" alt="그림1" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/d68935e2-1ddc-42cf-a690-7634e00544b7">

### 2) 데이터 전처리
그러나 이 데이터프레임이 데이터 전처리 작업을 거치지 않은 데이터이기 때문에 데이터 정제작업을 해주어야 더 정확한 분석 결과를 얻을 수 있습니다.
 
<img width="472" alt="그림2" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/03622a94-c2ba-42ef-8510-12f959006fac">

첫번째 문제점은 개행문자+정수 이 값이 문장의 끝마다 추가되어 있다는 점이었습니다. 테이블로 읽어오면서 생겼던 문제점이라 추측하였습니다. 따라서 개행문자+정수 부분을 제거해 주어야 합니다.

 
<img width="393" alt="그림3" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/cf383de9-ebeb-4f46-a37f-280f7b92e082">

개행문자와 정수는 delenter라는 함수를 직접 만들어서 제거했습니다. 데이터를 살펴보았을 때 개행이 3자리수를 넘어가는 데이터가 없었기 때문에 두 자리 수 정수부터 개행문자에 연결시켜서 연결 값이 데이터안에 있으면 제거하였습니다. 또한 파일 100.txt 에서 탭문자만 들어가 있는 문제가 있어서 이 또한 추가로 replace함수로 제거하였습니다.

<img width="435" alt="그림4" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/805ea5fa-eb76-4a04-9b4a-be9b52595c64">

두번째 문제점은 필요 없는 문구와 필요한 문구를 구분해야 한다는 점이었습니다. 그림 2 14를 보면 “67 out of 144”라는 문구가 보입니다. 144명중 67명이 이 리뷰에 공감을 하였다는 뜻입니다. 따라서 데이터의 가치를 판단하는 것에 있어 중요한 요소가 될 수 있다고 판단하였습니다. 따라서 이 부분을 추출하여 데이터프레임의 column으로 만들 것입니다. 또한 데이터를 추출한 후 남은 “Was this review helpful?” 부터 텍스트의 끝까지는 필요 없는 요소입니다. 따라서 이 부분은 데이터에서 제거하도록 할 것입니다.
 
<img width="498" alt="그림5" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/b2b60667-2554-4eb0-b9bd-066e433565c7">

이 문제는 그림 2-15의 makeviews 함수를 만들어서 처리하였습니다. 조회수 column은 “out of”라는 글귀를 기준으로 +7의 위치부터 값이 정수인지 계산하여 만약 정수라면 조회수 변수에 저장하였습니다. 만일 정수가 아니면 결측치 np.nan을 채워 넣었습니다. 한가지 애로사항이 있었는데 그것은 텍스트 데이터 중에 ‘정수 out of 정수’이렇게 데이터가 끝나는 경우였습니다. 함수의 알고리즘이 out of 다음 값을 계속 검사해 나가는데 이런 경우에 다음 값을 검사할 수 없어서 문제가 발생하였습니다. 따라서 문장의 끝에 도달하면 검사를 종료하는 조건을 추가하였습니다.

<img width="452" alt="그림6" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/c6477d60-7086-4967-b2a4-f85a246b840f">

공감수는 makeempathy 함수를 만들어서 처리하였습니다. 공감수 column은 “out of”라는 글귀를 기준으로 -2의 위치부터 값이 정수인지 계산하여 만약 정수라면 공감수 변수에 저장하였습니다. 만일 정수가 아니면 결측치 np.nan을 채워 넣었습니다.
각 값의 검증은 원본데이터인 document_df의 길이와 추출한 views리스트와 empathy 리스트의 길이가 같음을 확인하였습니다. 그 후 이 리스트 2개를 데이터 프레임에 추가하였습니다.


## 3. 상위 25% 데이터 추출
<img width="397" alt="그림7" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/8b07a070-7885-48de-903a-86890f721a99">

views와 empathy의 결측치를 확인한 결과 데이터에서 12.5%가 결측치인 것으로 확인하였습니다. 결측치가 10%를 넘어가면 결측치를 채워 넣는 것이 데이터 분석에 유리하게 작용합니다. 그러나 아직 어떤 것을 기준으로 결측치를 채워 넣어야 할지 정하지 않았습니다.

<img width="452" alt="그림8" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/83c6f4b7-5d45-48fd-ba42-83893dac6613">

이 부분에서 비지도 학습 기반의 감성분석인 VADER를 이용하였습니다. VADER는 주로 소셜 미디어의 텍스트에 대한 감성 분석을 제공하기 위한 패키지로 뛰어난 감성 분석 결과를 제공하며 비교적 빠른 수행 시간을 보장해 대용량 텍스트 데이터에 잘 사용됩니다. 이 VADER를 통해 Sentiment라는 감정 칼럼을 만들어서 긍정과 부정여부를 저장하였습니다.
### 1) EV Column생성

<img width="452" alt="그림9" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/4a2655c2-e168-43c4-8e11-5638cfadc7cd">

우선 ‘공감수 나누기 조회수 곱하기 100’의 column을 EV라는 이름으로 생성하였습니다. 

<img width="441" alt="그림10" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/d23ccf22-cbce-4760-9c9b-0ecfb13883b8">

EV값을 기준으로 quantile함수를 사용해서 상위 25%, 하위 25%의 데이터들을 추출하였습니다.

### 2) 결측치 EDA

![그림11](https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/3b263d1d-63c6-4838-822e-332f43236302)

EV기준 상위 25% 데이터의 전체 감성분포를 분석하였습니다. 그 결과 그림 3-5처럼 전체 데이터에서는 긍정의 수가 2배갸량 많았습니다.

![그림12](https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/c9084384-dc5d-4585-80ee-19d87616cd9d)

</br>
</br>

![그림13](https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/19e123ac-8196-4dea-9d59-a81abdf65a5a)

상위 25%의 데이터와 하위 25%데이터의 감성 비율을 분석하였습니다. 그 결과 그림 3-6처럼 상위 25%에서는 부정이 더 많았습니다. 반면 그림 3-7을 보면 하위 25%에서는 긍정이 압도적으로 많았습니다.
전체 데이터의 긍정 부정의 비율을 살펴보았을 때 긍정의 수가 2배가량 많은 것을 고려하면, 상위 25%에서 부정이 더 많았다는 것은 큰 의미로 작용할 수 있다고 생각했습니다.
</br>
</br>

![그림14](https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/0bc1f1bf-7d9b-4f4e-89a6-fcf74e77ac73)

![그림15](https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/d5bf10b4-e8c9-4c49-af8b-8197f696fdf9)

추가로 전체데이터에서 긍정 부정 별 EV의 평균과 중앙값을 비교해보았습니다. 그 결과 그림 3-8과그림 3-9처럼 EV의 평균, 중앙값 모두 부정이 높았습니다. 앞의 내용까지 고려하면 EV값은 부정이 대체적으로 높을 것이라고 추측할 수 있습니다.

그러나 아직 수학적인 수치로 증명이 된 것이 아니기 때문에 수학적인 근거를 찾아보려고 하였습니다.
 
![그림16](https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/192ea1ad-434d-40cc-be27-eac2a10615e1)

그래서 상관관계를 보기로 하였습니다. 그러나 EV와 긍정, 부정지수는 상관관계가 없었습니다. 

<img width="351" alt="그림17" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/e673dad4-1ef2-46e6-aee1-516ba11c3ba9">

선형관계 또한 살펴보았지만 Linear regression을 비롯하여 릿지, 라소 모두 R스퀘어 지수가 마이너스 혹은 0으로 평균값으로 예측하는 것보다 성능이 좋지 않은 즉 전혀 선형관계가 없다는 것을 확인하였습니다. 이로써 결측치를 수학적인 함수로 예측을 할 수 없습니다.

따라서 처음 분석의 결과를 고려하여 EV 결측치의 감성 column이 긍정이면 긍정의 중앙값으로 EV 결측치의 감성column이 부정이면 부정의 중앙값으로 값을 채워 넣는 것으로 정하였습니다.

### 3) 결측치 처리 및 상위 25% 데이터 추출

<img width="499" alt="그림18" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/bdacef03-ec35-475e-b822-029b6ee1c172">

조회수와 공감 수가 모두 0인 경우에 나눗셈 연산을 한 경우 결측치가 EV에 들어갑니다. 따라서 이부분을 0으로 초기화합니다.

<img width="513" alt="그림19" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/a210d54c-37ae-48a5-89a7-09a1fc3cc955">

처음 분석의 결과를 고려하여 EV 결측치의 감성 column이 긍정이면 긍정의 중앙값으로 EV 결측치의 감성column이 부정이면 부정의 중앙값으로 값을 채워 넣었습니다.

<img width="299" alt="그림20" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/7e71d9e1-a8e4-4625-a493-432ad9e39f43">

그 후 그림 3-14처럼 EV값을 기준으로 상위 25%의 데이터(top)를 추출하였습니다

## 4. K-Means 적용
### 1) TFIDF

<img width="358" alt="그림21" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/97eb942e-b7e4-4b83-b6db-60f5bcf50756">

Kmeans 알고리즘을 적용하기 이전에 우리는 기계가 데이터를 알아들을 수 있는 형식으로 바꾸어 주어야합니다. 여기서 TFIDF라는 개념이 나오는데 이 TFIDF는 특정 단어가 문서 내에 등장(가중치)하는 빈도와 그 단어가 문서 전체 집합에서 등장(패널티)하는 빈도를 고려하여 벡터화 하는 방법입니다.

<img width="491" alt="그림22" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/4b6f04dc-3bb9-4a0b-a23b-25497e01005a">

사이킷런의 TFIDF에는 여러 파라미터를 통해 벡터화를 조절할 수 있습니다. ngram으로 문맥적요소를 반영하였으며 stop_words리스트에 수동으로 더 불용어를 추가해서 불용어를 추가적으로 제거하였습니다. 또한 max_df, min_df 수치를 주어 의미 없는 단어가 나타나는 수를 조절하여 피처에 반영하였습니다.

### 2) Lemmatization

<img width="452" alt="그림23" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/893b320a-1ba7-4a43-94c0-e53db3774305">

그 후에는 어근추출을 진행하였는데 LemNormalize함수를 통해 소문자로 변환 후 특수기호를 제거하였습니다. 그 후 토큰화를 진행하였고 LemTokens함수에 넘겨서 어근을 추출하고 리스트에 넣어서 반환하였습니다.

### 3) 최적의 군집수
이렇게 해서 EV값 기준 상위 25% 텍스트의 단어들이 벡터화 된 정형데이터가 만들어졌습니다. 그러나 여기서 한가지 더 생각해야할 부분이 있습니다. 바로 KMeans의 군집수는 인간이 설정해야 한다는 부분입니다.

#### 가. Elbow Method

<img width="302" alt="그림24" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/97d5e5be-1b96-443c-8d52-3f7948653cd0">

‘elbow method’를 통해 최적의 군집수를 찾을 수 있습니다. 우측하단 그림을 보면 y값이 계속 줄어들다가 어느 지점부터 그 정도가 작아지는 부분을 볼 수 있습니다. 마치 팔처럼 굽어지는 이부분을 찾는 것을 elbow method라고 합니다.

<img width="407" alt="그림25" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/599d421c-e947-4282-ac0c-a89675abacf3">

이 값은 inertia라는 군집에 속한 샘플들의 거리 수치를 통해 계산할 수 있습니다. 같은 클러스터 안에 요소들의 거리가 가깝다는 것은 상당히 군집화를 잘했다는 것이기 때문에 좋은 수치입니다. 따라서 그림 4-5의 반복문을 통해 이 값을 살펴보았습니다. 그러나 그림 4-6을 보시면 elbowpoint라고 할만한 지점을 찾지 못하였습니다.

![그림26](https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/b46f85d4-ab3e-4d80-8610-97c21369ca14)

#### 나. Silhouette Analysis

<img width="212" alt="그림27" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/b6aa39df-4fc8-4457-b678-b657c7709c8e">

그래서 선택한 방법이 실루엣 분석입니다. 실루엣 분석은 각 군집 간의 거리가 얼마나 효율적으로 분리돼 있는지를 나타냅니다. 효율적으로 잘 분리됐다는 것은 다른 군집과의 거리는 떨어져 있고 동일 군집끼리의 데이터는 서로 가깝게 잘 뭉쳐 있다는 의미입니다. 실루엣 분석은 실루엣 계수를 기반으로 합니다. 실루엣 계수는 개별 데이터가 가지는 군집화의 지표입니다. 이는 해당 데이터가 같은 군집내의 데이터들이 얼마나 가까운지 다른 군집에 있는 데이터와는 얼마나 멀리 분포되어 있는지를 총괄적으로 고려한 수치로 높을수록 좋은 값입니다.
그림 4-7을 보시면 특정 데이터 포인트의 실루엣 계수 값은 해당 데이터 포인트와 같은 군집 내에 있는 데이터 포인트와의 거리를 평균한 값 a(i), 해당 데이터 포인트가 속하지 않은 군집 중 가장 가까운 군집화의 평균 거리 b(i)를 기반으로 계산됩니다. 두 군집 간의 거리가 얼마나 떨어져 있는가의 값은 b(i) – a(i)이며 이 값을 정규화하기 위해 MAX( a(i), b(i) )값으로 나눕니다.  
 
<img width="374" alt="그림28" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/944fa547-2ea9-45db-b521-744e7415693c">

따라서 이것 역시 그림 4-8처럼 반복문을 통해 최적값을 찾으려 했습니다. 결국 그림 4-9를 보면 K=8일 때 실루엣계수가 가장 높게 나타났기에 군집수를 8개로 정하였습니다.

<img width="512" alt="그림29" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/d5a278f6-86cb-4a82-bf98-c90ddf2856fc">

### 4) 군집화 수행 및 결과

<img width="451" alt="그림30" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/230655d8-8e59-4cb2-88f1-4c18b03e5d70">

그렇게 군집화를 수행하였고 그 군집화의 결과를 column으로 집어넣었습니다. 또한 군집 내 단어들의 value를 통해 한 군집을 대표하는 단어 top 5를 출력할 수 있게 설정하였습니다. 그림 4-10의 코드를 보면 cluster_centers_라는 속성이 보입니다. KMeans객체는 각 군집을 구성하는 단어 피처가 군집의 중심을 기준으로 얼마나 가깝게 위치해 있는지 이 속성을 통해 제공합니다. 이 속성을 이용하여 단어마다 value값을 만들어 주었고 이 값은 값이 클수록 즉 1에 가까울수록 군집의 중심과 가까운 값을 의미합니다. 다시 말해 그 군집을 대표할 수 있는 단어로 해석이 가능합니다.

<img width="452" alt="그림31" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/98342209-f44e-4456-91ea-507870332dd2">

그러면 각각의 클러스터에는 위와 같이 단어들이 묶여 있습니다. 


| Cluster  | Top1     | Top2          | Top3     | Top4     | Top5    |
|----------|----------|---------------|----------|----------|---------|
| Cluster0 | royal    | battle royal  | battle   | better   | watched |
| Cluster1 | show     | great         | good     | watched  | acting  |
| Cluster2 | episode  | id            | hype     | season   | someone |
| Cluster3 | old man  | guy           | Bad      | character| old     |
| Cluster4 | end      | episode       | first    | done     | well    |
| Cluster5 | life     | style         | play     | acting   | squid   |
| Cluster6 | people   | character     | show     | make     | way     |
| Cluster7 | last     | last episode  | episode  | worth    | three   |

위의 표를 살펴보겠습니다. 클러스터 0에는 battle royal에 관련한 단어들이 묶여 있습니다. 비슷한 의미의 단어들이 묶인 것으로 보아 클러스터링이 잘되었다고 추측할 수 있습니다. 클러스터 1에는 쇼, 최고, 연기라는 단어가 묶여 있는 것으로 보아 ‘연기력이 훌륭했다’고 추측해볼 수 있습니다. 클러스터2에는 에피소드id, 시즌 단어들이 묶여 있습니다. 클러스터3에는 old man 캐릭터와 관련된 단어들이 묶여 있습니다. 클러스터4에는 Episode와 순서에 관련한 단어가 묶여 있습니다. 클러스터 5에는 삶, 연기, 오징어 게임 등의 단어가 묶여 있습니다. 클러스터6에는 사람, 캐릭터, 쇼 등의 단어가 묶여 있습니다. 클러스터7에는 마지막 에피소드에 관한 단어들이 묶여 있습니다. 
그러나 이렇게만 봐서는 인간의 해석이 강하게 들어갈 수밖에 없는 문제점이 발생합니다. 프로젝트의 목표는 인간이 군집화 된 단어들을 해석하는 것이 아니기때문에 프로젝트의 마지막 단계인 다음장으로 넘어가겠습니다.

## 5. LSTM 적용
### 1) 학습데이터 구성

<img width="452" alt="그림32" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/97a0433e-cec5-4be7-88b2-51ebce9cdf03">

각각의 군집의 텍스트를 LSTM에 학습시키기 위해서는 토큰화된 단어를 수치화하고 이를 연속적인 리스트로 만들어주어야 합니다. 그 이유는 “모델이 단어를 예측하기 위해 이전에 등장한 단어를 모두 활용하기 위함”입니다. 그림 5-1은 데이터를 연속적인 리스트로 만들어주는 코드입니다. 이렇게 하면 데이터의 단어를 토큰화하고 데이터의 리스트들은 연속적인 값을 가지게 됩니다. 그러나 이런 식의 값은 문장의 길이가 모두 다르기 때문에 기계가 읽을 수 없습니다.

<img width="452" alt="그림33" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/5a2935cb-0742-497e-9d81-9df10cfaff28">

따라서 비어 있는 요소들에 패딩이라는 작업을 거쳐야 하는데 그림5-2의 0으로 채워주는 작업입니다. 또한 데이터가 앞장에 비해 오른쪽으로 밀려 있는 경향을 가지고 있습니다. 이것은 모델의 마지막 값을 타겟으로 계속해서 예측을 해 나가기 때문에 필요한 데이터의 구성입니다.

### 2) LSTM 모델 구성

<img width="452" alt="그림34" src="https://github.com/S-SIRIUS/Predicting-the-direction-of-next-hit/assets/109223193/7a189e0c-c1f0-4cae-b04f-31d1032d2fdf">

LSTM모델은 위의 그림처럼 구성하였습니다. embedding layer는 단어를 의미론적 기하 공간에 매핑할 수 있도록 벡터화 시키는 layer입니다. 그 다음은 128개의 유닛의 lstm layer를 추가하였고 dense layer는 입력과 출력을 연결해주는 layer입니다.

### 3) LSTM 모델로 TEXT 생성 및 해석
| Cluster  | 단어        | LSTM TEXT 생성 |
|----------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Cluster0 | Royal watched | Royal watched the whole lot and its just plain silly the cast bring overacting to a new level all the characters are spectacularly annoying i wanted them all to die by episode |
| Cluster1 | Show          | Show of the best shows ive watched everyone of the actors was so so good and the second like which it really slow i often have the feeling that many scenes are there just to make the show longer adding very little to the plot |
| Cluster2 | Episode       | Episode 1 was amazing but things went downhill a little bit each episode episode 4 i was totally over it i did watch all 9 episodes because id already invested so much time into it |
| Cluster3 | Old man       | Old man the korean version of hunger game with some cheap metaphors to criticize capitalism democracy ethics etc terrible acting cliche dialogues and predictable scenes that drag for minutes and minutes |
| Cluster4 | First         | First started squid game i didnt expect much just something to pass time at work after the first two episodes i was hooked the show is thrilling and makes your heart beat increase i highly recommend watching if you are looking to have some entertainment for a couple hours hours |
| Cluster5 | Life          | Life overrated it keeps the same drum about what youd do in a game if it was your life versus anothers life its got a certain style but the acting was amateurish and melodramatic sorta like anime |
| Cluster6 | Character     | Character mean am an ardent follower of the korean movies even though most of the movies are laden with violence gore and bloodbath ... i risked my time on the rave reviews the show is getting and boy it has paid back i cannot recall any single show more edge of the seat than this yes there may be better shows in terms of intrinsic production valueson the rave reviews the show is getting and boy it has paid back i cannot recall any single show more edge of the seat than this yes there may be better shows in terms of intrinsic production values |
| Cluster7 | Last Episode  | Last Episode very very very minor i really didnt like the last three episodes and the wrapup was dissatisfying gganbu was one of the strongest episodes in an emotional perspective |

최종 단계입니다. 군집 단어의 value값을 우선순위를 두고 학습한 LSTM모델에 넣어 문장을 뽑았습니다. 만약 첫번째 단어를 입력으로 넣었을 때 문장이 이상하다면 다음 우선순위의 단어를 넣었습니다
우선 군집0의 Royal watched를 넣었을 경우입니다. “모든 캐릭터는 엄청나게 짜증이 납니다. 에피소드별로 모두 죽기를 원했습니다.”라는 문장이 생성되었습니다. 여기서 캐릭터들에 대한 부정적인 의견을 볼 수 있는데 이것이 작품에 너무 몰입을 해서 부정적인 것인지 알 수가 없었습니다.
두번째는 군집1의 ‘show’를 넣었을 경우입니다. “내가 본 최고의 쇼의 쇼는 모든 배우들이 너무 좋았고 두 번째는 정말 느려서 쇼를 더 길게 만들기 위해 많은 장면이 거기에 있다는 느낌이 듭니다.”라는 문장이 생성되었습니다. 이를 통해 배우들에 대한 긍정적인 의견을 엿볼 수 있고 쇼를 의도적으로 길게 만든 것이 눈에 보였다는 것을 해석할 수 있습니다.
세번째는 군집2의 ‘episode’를 넣었을 경우입니다. “1화는 굉장했지만 각 에피소드 에피소드 4화는 완전히 끝났습니다. 이미 너무 많은 시간을 투자했기 때문에 9화를 모두 시청했습니다.”라는 텍스트에서는 에피소드 1화는 좋았으나 중반부터 부정적임을 해석할 수 있고 앞에 거를 다 보아서 그냥 끝까지 보았다는 것을 알 수 있습니다.
네번째는 군집3의 ‘Old man’을 넣었을 경우입니다. “올드맨 한국판 헝거게임 자본주의 민주주의 윤리 등을 비판하는 싸구려 은유 등 진부한 연기의 진부한 대사와 예측 가능한 장면들이 몇 분씩 질질 끌린다.” 전체적으로 비판적인 의견으로 또 다시 내용이 길어진다 라는 의견이 나왔습니다.
다섯번째는 군집4의 ‘First’를 넣었을 경우입니다. “처음 시작한 오징어 게임 나는 첫 두 에피소드 후 직장에서 시간이 지나갈 것을 기대하지 않았습니다. 나는 쇼가 스릴 있고 심장 박동을 증가시킵니다. 몇 시간 동안 엔터테인먼트를 찾고 있다면 시청하는 것이 좋습니다.” 전체적으로 긍정적인 의견이며 킬링타임용으로 좋다고 해석해 볼 수 있습니다.
여섯번째는 군집5의 Life를 넣었을 경우입니다. “과대평가된 인생은 당신의 삶과 다른 사람의 삶이라면 게임에서 하는 일에 대해 같은 북을 유지합니다. 특정 스타일이 있지만 연기는 애니메이션처럼 아마추어적이고 멜로드라마적이었습니다.” 문장이 매끄럽지는 않지만 연기가 애니메이션과 같이 깊이가 깊지 않다는 의견으로 추측됩니다.
일곱번째 군집6의 character를 넣었을 경우입니다. “캐릭터 의미는 대부분의 영화가 폭력적이지만 나는 약간 불안한 한국 영화의 열렬한 추종자입니다. 한국 웹시리즈가 엄청나게 인기 있고 오징어 게임이라는 이름 넷플릭스의 최신 제품이 그다지 매력적이지 않은 것 같지만 쇼가 받고 있는 격찬에 내 시간을 걸었습니다. 그리고 보답을 받았습니다.” 제목이 매력적이지 않았지만 재미있었다고 해석됩니다.
여덟번째 군집 7의 Last Episode를 넣었을 경우입니다.” 마지막화 아주아주아주사소함 마지막 3화가 정말 마음에 안들었고 마무리가 불만족스러웠음 깐부는 감성적으로 가장 강한 에피소드중 하나였다.” 마지막 3화 그리고 결말에 부정적인 의견을 엿볼 수 있으며 깐부라는 에피소드는 임팩트가 있었음을 알 수 있습니다.

## 6. 시나리오 방향성 제시
이제 앞의 해석을 토대로 시나리오의 방향성을 제시해 보려 합니다. 
첫째, 에피소드를 늘리고자 억지요소의 스토리 라인을 구성하면 안 됩니다. 군집1와 군집3에서의 생성된 텍스트에서는 쇼를 의도적으로 만든 것이 보였고 대사, 연기, 장면들이 예측이 가능할 정도로 끄는 경향이 있음을 파악하였기 때문입니다.
둘째, 배우들이 매우 중요합니다. 군집 5에서 연기가 애니메이션처럼 아마추어적이고 맬로틱하다는 텍스트가 생성되었습니다. 즉 단순한 연기지만 개성 있게 할 수 있는 배우가 필요합니다. 또한 군 집1에서는 배우들에 대한 긍정적인 의견을 볼 수 있습니다. 따라서 오디션을 통해 개성 있는 배우를 캐스팅하는 것이 중요합니다. 
세번째 ‘깐부’와 같은 임팩트 있는 에피소드가 하나는 있어야 합니다. 군집 7을 보면 후반부와 결말까지의 부정적인 의견에 비해 깐부는 임팩트가 있었다는 텍스트가 생성되었습니다. 따라서 외국인들이 알 수 없는 고유의 단어의 사용을 통해 임팩트 있는 에피소드를 한 개정도 만드는 것이 흥행에 기여할 것이라고 추측됩니다.
네번째 에피소드 초반에 비해 중반부터 결말까지의 부정적인 의견을 고려해야합니다. 앞서 말한 것처럼 쇼가 의도적으로 끄는 느낌이 든다는 의견이 강했습니다. 군집 2에서는 에피소드 중반부터 부정적인 의견 그리고 앞을 봐서 뒤까지 그냥 보았다는 텍스트가 있었습니다. 군집 7역시 마지막 3화에 대한 부정적인 의견이 있었습니다. 따라서 깐부와 같은 임팩트 있는 요소를 뒤에 배치하여 마지막까지 긴장감을 놓치지 않게 하는 것 등이 대안이 될 수 있습니다.
