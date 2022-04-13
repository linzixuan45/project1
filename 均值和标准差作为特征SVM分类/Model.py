import torch
'''依据原始论文搭建VGG网络'''
import torch.nn as nn
import torch
import cv2
class CNN(nn.Module):
    def __init__(self, num_channels=3):  #由于传入的是三维坐标
        super(CNN, self).__init__()
        #【input 244x244】 ->【conv（3）-（64） ，con（3）-（ 64）】->【maxpool】 3层 特征维度3->64
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=64,kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(64, 64, 3, padding='same')
        #->【conv（3）-（128） ，con（3）-（128）】->【maxpool】 2层 特征维度64->128
        self.conv3 = nn.Conv1d(64, 128, 3, padding='same')
        self.conv4 = nn.Conv1d(128, 128, 3, padding='same')
        # ->【conv（3）-（256） ，con（3）-（256），conv（3）-（256）】->【maxpool】 2层 特征维度128->256
        self.conv5 = nn.Conv1d(128,256,3,padding='same')
        self.conv6 = nn.Conv1d(256,256,3,padding='same')
        self.conv7 = nn.Conv1d(256,256,3,padding='same')
        #->【conv（3）-（512） ，con（3）-（512），conv（3）-（512）】->【maxpool】 2层 特征维度256->512
        self.conv8 = nn.Conv1d(256,512,3,padding='same')
        self.conv9 = nn.Conv1d(512,512,3,padding='same')
        self.conv10 = nn.Conv1d(512,512,3,padding='same')
        #->【conv（3）-（512） ，con（3）-（512），conv（3）-（512）】->【maxpool】 2层 特征维度512->512
        self.conv11 = nn.Conv1d(512,512,3,padding='same')
        self.conv12 = nn.Conv1d(512,512,3,padding='same')
        self.conv13 = nn.Conv1d(512,512,3,padding='same')
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2,stride=2)
        self.sofmax = nn.Softmax(dim = 0)
        self.dropout = nn.Dropout()

    def forward(self,input):  # input 为 RGB 图片，此时我们关注一下图片的大小
        temp  = self.relu(self.conv1(input)) #244x244
        temp  = self.relu(self.conv2(temp))
        temp  = self.maxpool(temp) # (M-K+2p)/S+1  （244-2）/2+1 =122  此时图片大小 122x122
        temp  = self.relu(self.conv3(temp))
        temp  = self.relu(self.conv4(temp))
        temp  = self.maxpool(temp)#  （122-2）/2+1 = 61  61x61
        temp  = self.relu(self.conv5(temp))
        temp  = self.relu(self.conv6(temp))
        temp  = self.relu(self.conv7(temp))
        temp  = self.maxpool(temp)#  向下取整（61-2）/2+1 =  30x30
        temp  = self.relu(self.conv8(temp))
        temp  = self.relu(self.conv9(temp))
        temp  = self.relu(self.conv10(temp))
        temp  = self.maxpool(temp) # （30-2）/2 +1 = 15  15x15
        temp  = self.relu(self.conv11(temp))
        temp  = self.relu(self.conv12(temp))
        temp  = self.relu(self.conv13(temp))
        temp  = self.maxpool(temp)# （15-2）/2+1 = 7x7
        # torch.Size([1, 512, 7, 7])
        print(f"temp_shape  is : {temp.shape}")
        temp = temp.reshape(-1)  # 一维  25088

        linear1 = nn.Linear(in_features=temp.shape[0] , out_features=64)
        linear2 = nn.Linear(in_features = 64,out_features=2)
        temp = self.relu(linear1(temp))
        temp = self.dropout(temp)
        temp = self.relu(linear2(temp))
        temp = self.sofmax(temp)
        return temp

    def observe_pool(self,input):
        '''
        此函数主要用来观测maxpool下采样后的样子 ，可以传入1 通道gray  或者三通道RGB，
        格式要求：(batch_size,channels,Height, width)
        或者是 （channels,Height, width)
        '''
        out1 = self.maxpool(input)  #（244-2）/2+1 =122  此时图片大小 122x122
        out2 = self.maxpool(out1) #（122-2）/2+1 = 61  61x61
        out3 = self.maxpool(out2)
        out4 = self.maxpool(out3)
        out5 = self.maxpool(out4)
        return [out1,out2,out3,out4,out5]
if __name__ == "__main__":
    cnn = CNN()
    input = torch.ones(3,3, 32)  #N是一个批处理大小，C表示多个通道， L 是信号序列的长度
    output = cnn.forward(input)
    print(output.shape,output)