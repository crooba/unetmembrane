# unetmembrane

세포막 세그맨테이션 프로젝트

세포막경계를 분할하는 인공신경망코드입니다.
대략 20번의 반복학습을 통해 95%정도의 정확도를 확보했습니다.

데이터구조는 깃허브에 올려진 데이터 구조대로 사용하시길 권장드립니다.
그리고 코드상의 데이터경로는 각자가 가지고 있는 데이터경로로 업데이트 하셔야 합니다.
업데이트 할 데이터경로는 main.py 코드상의 train폴더와 test폴더 경로입니다.

사용법

데이터셋과 코드를 다운로드하시고 main.py를 실행하시면 unet신경망이 훈련되고 결과물이 test폴더에 저장됩니다.

아래 이미지는 세포현미경이미지(왼쪽)와 그것을 토대로 훈련받은 유넷신경망이 세그맨테이션 한 세포경계선이미지(오른쪽)입니다.
전문가들의 손작업을 기준으로 약95%의 정확도로 훈련되었습니다.



<img width="1280" alt="스크린샷 2020-01-28 오후 8 44 45" src="https://user-images.githubusercontent.com/45910733/73261269-160b3f80-420f-11ea-8a70-160da0253f72.png">
