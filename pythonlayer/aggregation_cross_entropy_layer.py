import sys
# sys.path.insert(0, '/media/f/jiakao/project/wheeling/PSPNet')
sys.path.insert(0, '/work/ocr/certificate/car_certificate/model/CRNN_ACE/caffe-lstm-ocr/python')
import caffe
import math
import numpy as np


def SoftMax_T(net_ans, T_):
    tmp_net = [math.exp(i) for i in net_ans]
    sum_exp = sum(tmp_net)*T_*1.0
    return np.array([i/sum_exp for i in tmp_net])

def get_label_cnts(label, dict_size, T_):
    def get_cnt(single_label, dict_size, T_):
        return np.array([single_label.count(np.float(i))/(T_*1.0) for i in range(dict_size)])

    def get_label_list(alabel):
        return [i[0][0] for i in alabel]
        
    return np.array([get_cnt(get_label_list(i), dict_size, T_) for i in label])

def get_score_cnts(score, T_):
    i_max, j_max = score.shape[: 2]
    softmax_score = np.array([np.array([SoftMax_T(score[i][j], T_) for j in range(j_max)]) for i in range(i_max)])
    return np.sum(softmax_score, 0), softmax_score
    
def theta_i_j(i, j):
    return 1 if i == j else 0

def get_diff_1(T_, batch_size, label_norm_Ns, score_norm_ys, softmax_score, dict_size):
    # print("Hi I am in get diff!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # -np.sum([label_norm_Ns[b][i]*softmax_score[t][b][i]*T_*(theta_i_j(k, i) - softmax_score[t][b][k]*T_)/(score_norm_ys[b][i]) for i in range(dict_size)])/T_
    diff = np.array([np.array([np.array([-np.sum([label_norm_Ns[b][i]*softmax_score[t][b][i]*T_*(theta_i_j(k, i) - softmax_score[t][b][k]*T_)/(score_norm_ys[b][i]*T_) for i in range(dict_size)]) for k in range(dict_size)]) for b in range(batch_size)]) for t in range(T_)])
    # print("Hi I am done about diff")
    return diff

def get_diff(T_, batch_size, label_norm_Ns, score_norm_ys, softmax_score, dict_size):
    diff = np.array([np.array([-1.0/T_ * np.sum((np.identity(dict_size) - (softmax_score[t][b]*T_).reshape(-1, 1))*(softmax_score[t][b]*T_ * label_norm_Ns[b] / score_norm_ys[b]), 1) for b in range(batch_size)]) for t in range(T_)])
    # part1 = np.identity(dict_size) - (softmax_score[0][0]*T_).reshape(-1, 1)
    # part2 = softmax_score[0][0]*T_ * label_norm_Ns[0] / score_norm_ys[0]
    # -1.0/T_ * np.sum(part1*part2, 1)
    return diff 



class AggregationCrossEntropyLayer(caffe.Layer):
    """
    Comput the Aggregation Cross Entropy loss for ocr rec plan
    """

    def setup(self, bottom, top):
        self.T_, self.batch_size, self.dict_size = bottom[0].data.shape
        # print("========= Hi I am setup ========")
        if len(bottom) != 2:
            raise Exception("Need two inputs to computer loss.")

    def reshape(self, bottom, top):
        # print("========= Hi I am reshape ========")
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)
    
    def forward(self, bottom, top):
        score = bottom[0].data
        label = bottom[1].data

        T_, batch_size, dict_size = self.T_, self.batch_size, self.dict_size
        label_norm_Ns = get_label_cnts(label, dict_size, T_)
        score_norm_ys, softmax_score = get_score_cnts(score, T_)
        loss = (-np.sum(np.log(score_norm_ys)*label_norm_Ns))/batch_size
        # print("========= Hi I am forward ========")
        # # print("label_norm_Ns: ")
        # # print(label_norm_Ns)
        # # print("label norm Ns shape: %s" %(str(label_norm_Ns.shape)))
        # # print(label_norm_Ns[0])

        
        # # print("score_norm_ys: ")
        # # print(score_norm_ys)
        # # print("score_norm_ys shape: %s" %(str(score_norm_ys.shape)))
        # # print(score_norm_ys[0])

        # # print("softmax_score:")
        # # print(softmax_score)
        self.label_norm_Ns = label_norm_Ns
        self.score_norm_ys = score_norm_ys
        self.softmax_score = softmax_score

        
        # print("======== loss: %s ========" %(str(loss)))

        top[0].data[...] = loss


    def backward(self, top, propagate_down, bottom):
        T_, batch_size, dict_size = self.T_, self.batch_size, self.dict_size
        label_norm_Ns = self.label_norm_Ns
        score_norm_ys = self.score_norm_ys
        softmax_score = self.softmax_score

        self.diff[...] = get_diff(T_, batch_size, label_norm_Ns, score_norm_ys, softmax_score, dict_size)
        # self.diff[...] = bottom[0].data
        # print("========= diff : =========")
        # print(self.diff)
        # print(self.diff.shape)
        # print("========= Hi I am backward ========")
        bottom[0].diff[...] = self.diff



