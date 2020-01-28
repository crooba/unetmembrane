from model import *
from data import *

#훈련데이터셋을 늘리기위하여 변형될 이미지들의 회전,높이,좌우반전등의 설정값들을 임의로 지정합니다.
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

#훈련데이터셋을 늘려줍니다.                    
myGene = trainGenerator(2,'/content/drive/My Drive/Colab Notebooks/unetmembraneup/unet/data/membrane/train','image','label',data_gen_args,save_to_dir = None)

#준비 된 유넷신경망을 불러와 훈련시킵니다.그리고 가중치를 저장합니다.
model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=200,epochs=20,callbacks=[model_checkpoint])

#학습된 유넷신경망으로 테스트셋을 돌려서 예측된 이미지를 추출 후 확인해봅니다.
testGene = testGenerator("/content/drive/My Drive/Colab Notebooks/unetmembraneup/unet/data/membrane/test")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("/content/drive/My Drive/Colab Notebooks/unetmembraneup/unet/data/membrane/test",results)