import torch
import pickle
import matplotlib.pyplot as plt
#바이너리파일
broken_image = torch.FloatTensor( pickle.load(open('./broken_image_t.p','rb'),encoding='latin1'))
#인코딩을 할 때 실제언어에 따른 인코팅방식을 반영하기 위한 "latin1"에 대한 방식을

plt.imshow(broken_image.view(100,100))
## 이미지를 오염시키는 함수
def weird_function(x,n_iter=5):
    h=x
    filt=torch.tensor([-1./3, 1./3, -1./3])
    for i in range(n_iter):
        zero_tensor = torch.tensor([1.0*0])
        h_1=torch.cat((zero_tensor, h[:-1]), 0)
        h_r=torch.cat((h[1:], zero_tensor), 0)
        h=filt[0]*h+filt[2]*h_1+filt[1]*h_r
        if i%2== 0:
            h=torch.cat((h[h.shape[0]//2:], h[:h.shape[0]//2]), 0)
    return  h
def distance_loss(hypothesis, broken_image):
    return torch.dist(hypothesis, broken_image)
random_tensor=torch.randn(10000, dtype=torch.float)
lr=0.8   #적절한 매개변수를 선정하는 추가적인 방법은 없는지?

for i in range(0,20000):
    random_tensor.requires_grad_(True)
    hypothesis = weird_function(random_tensor)
    print(random_tensor,hypothesis) ##가설에 기존의 수식들이 역추척되어 기록이 남아 있음을 확인
    loss=distance_loss(hypothesis, broken_image)
    loss.backward()
    with torch.no_grad():
         random_tensor=random_tensor-lr*random_tensor.grad
    if i % 1000 == 0:
         print('Loss at {} = {}'.format(i, loss.item()))
##복원된 이미지 시각화
plt.imshow(random_tensor.view(100,100).data)
plt.show()