## WatchOut : TwinLiteNet 및 YOLOv8 기반 실시간 1인칭 시점에서의 주행 차선 진입 차량 감지 시스템

## 1. 연구 배경 및 목표
자율 주행과 같은 자동차 기술 발전과 함께, 자동차 회사에서는 안전 관련 문제가 생겨나게 되었다. 이를 해결하고자 자동차 회사에서는 ADAS를 도입하였다. 그러나, ADAS의 Lidar와 같은 고비용의 외부센서는 자동차 회사의 입장에서 가격 경쟁력 확보에 어려움을 준다. 또한, 부품 교체 시 사용자 입장에서도 부담이 되는 편이다.
본 팀에서는 블랙박스를 이용하여 운전자에게 주행 차선 차량 진입 감지 시스템을 제공하고자 한다.

블랙박스(웹 캠)와 Edge Device(Jetson Orin Nano), 스마트폰의 상호작용을 통해 운전자에게 교통 위험을 알려주는 서비스 개발을 목표로 한다.

## 2. System Flow
![image](https://github.com/FreshMeYeok/WatchOut/blob/main/Readme/System_flow.png)


## 3. Algorithm
![image](https://github.com/FreshMeYeok/WatchOut/blob/main/Readme/Algorithm.png)


## 4. Demo 및 발표 영상
[데스크탑을 이용한 실시간 데모](https://youtu.be/ri0G3heXln0)
[젯슨을 이용한 실시간 데모](https://youtu.be/jZuLtD2kars)


## 5. 실행 방법

### 5.1 [Requirments](https://github.com/FreshMeYeok/WatchOut/blob/main/requirements.txt)
```shell
pip install -r requirements.txt
```

### 5.2 파이썬  실행
```shell
git clone https://github.com/FreshMeYeok/WatchOut.git
cd WatchOut
python3 process_tensorrt.py
```



