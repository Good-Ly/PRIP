import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import word2vec
from sklearn.metrics.pairwise import cosine_similarity
index2word = ['L', 'E', 'A', 'V', 'K', 'R', 'G', 'I', 'S', 'D', 'P', 'T', 'F', 'N', 'Q', 'Y', 'H', 'M', 'W', 'C']
def calc_cosin_simi(word_vec,save_path,file_name):
    plt.rcParams['figure.figsize'] = (16, 16)
    path_save_file_name = save_path + file_name
    corr_matrix = cosine_similarity(word_vec)
    fig, ax_f = plt.subplots()
    ax_c = ax_f.twinx()
    ax_d = ax_f.twiny()
    ax = sns.heatmap(np.round(np.array(corr_matrix),2),xticklabels=index2word, yticklabels=index2word,
                cmap='YlGnBu', center=0, annot=True,linewidths=.5)
    ax_c.set_xticklabels([])
    ax_c.set_yticklabels([])
    ax_c.set_xticks([])
    ax_c.set_yticks([])

    plt.rcParams['figure.figsize'] = (16,16)
    #plt.title('Cosine similarity of word vectors', fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(path_save_file_name)
    plt.show()
def sn_sp_acc_mcc(true_label,predict_label,pos_label=1):
    pos_num = np.sum(true_label==pos_label)
    neg_num = true_label.shape[0]-pos_num
    tp =np.sum((true_label==pos_label) & (predict_label==pos_label))
    tn = np.sum(true_label==predict_label)-tp
    sn = tp/pos_num
    sp = tn/neg_num
    acc = (tp+tn)/(pos_num+neg_num)
    fn = pos_num - tp
    fp = neg_num - tn
    tp = np.array(tp,dtype=np.float64)
    tn = np.array(tn,dtype=np.float64)
    fp = np.array(fp,dtype=np.float64)
    fn = np.array(fn,dtype=np.float64)
    mcc = (tp*tn-fp*fn)/(np.sqrt((tp+fn)*(tp+fp)*(tn+fp)*(tn+fn)))
    return sn,sp,acc,mcc

def word_vector_list(index2word):
    word_vec = []
    for i in index2word:
        word_vec.append(word2vec.wv[str(i)])
    return word_vec
def hot_map(corr_array,save_path,file_name):
    path_save_file_name = save_path + file_name
    fig, ax_f = plt.subplots()
    ax_c = ax_f.twinx()
    ax_d = ax_f.twiny()
    ax = sns.heatmap(np.round(np.array(corr_array),2),xticklabels=index2word, yticklabels=index2word,
                cmap='YlGnBu', center=0, annot=True,linewidths=.5)
    ax_c.set_xticklabels([])
    ax_c.set_yticklabels([])
    ax_c.set_xticks([])
    ax_c.set_yticks([])

    plt.rcParams['figure.figsize'] = (16,16)
    #plt.title('Cosine similarity of word vectors', fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(path_save_file_name)
    plt.show()



def read_file(path, file_name):
    """****read file *****
           return list
    """
    file_path_name = path + file_name
    file = open(file_path_name, 'r', encoding="utf-8")  # open file
    lis = []
    for f in file:
        s = f.strip()
        lis.append(s)
    file.close()
    return lis


def head_end_str_connect(head_end_str, init_str):
    "*********  padding X *******"
    str_list = []
    str_list = "".join([head_end_str, init_str])
    str_list = "".join([str_list, head_end_str])
    return str_list


def init_seq_slice_no_X(path, file_name, len_seq=31, head_notes=0):
    Pro_Id = []
    seqs = []
    Bind_Sit_Seq = []
    Seq_X = []
    seq_labels = []
    slice_seq_n = []
    file_path_name = path + file_name
    data = read_file(path, file_name)
    file_seq_num = int(((len(data) - head_notes) / 3))

    # x = 'X'*((len_seq-1)//2)#(31-1)/2 = 15
    for j in range(file_seq_num):
        P_ID = data[(head_notes) + j * 3]
        seq = data[(head_notes + 1) + j * 3]
        bind_sit_seq = data[(head_notes + 2) + j * 3]
        if len(seq) == len(bind_sit_seq):
            Pro_Id.append(P_ID.strip(">"))
            seqs.append(seq)
            Bind_Sit_Seq.append(bind_sit_seq)
            index = 0
            # Seqx = head_end_str_connect(x,str(seq))
            # Seq_X.append(Seqx)
            for i in range(len(seq) - len_seq):  # Sequence division
                slice_seq_n.append(seq[i:i + len_seq])
                seq_labels.append(bind_sit_seq[i])
        else:
            print("Sequence error, please check,\n P_ID: %s \n  seq:%s \n bind_sit_seq: %s \n" % (
            P_ID, seq, bind_sit_seq))
    return slice_seq_n, seq_labels


def read_file_splic(path, file_name, len_seq=31, head_notes=0):
    """序列分割函数"""
    Pro_Id = []
    seqs = []
    Bind_Sit_Seq = []
    Seq_X = []
    seq_labels = []
    slice_seq = []
    file_path_name = path + file_name
    data = read_file(path, file_name)
    file_seq_num = int(((len(data) - head_notes) / 3))
    x = 'X' * ((len_seq - 1) // 2)  # (31-1)/2 = 15

    for j in range(file_seq_num):
        P_ID = data[(head_notes) + j * 3]
        seq = data[(head_notes + 1) + j * 3]
        bind_sit_seq = data[(head_notes + 2) + j * 3]
        if len(seq) == len(bind_sit_seq):
            Pro_Id.append(P_ID.strip(">"))
            seqs.append(seq)
            Bind_Sit_Seq.append(bind_sit_seq)
            index = 0
            Seqx = head_end_str_connect(x, str(seq))
            Seq_X.append(Seqx)

            for i in range(len(seq)):
                slice_seq.append(Seqx[i:i + len_seq])
                seq_labels.append(bind_sit_seq[i])
        else:
            print("Sequence error, please check,\n P_ID: %s \n  seq:%s \n bind_sit_seq: %s \n" % (
            P_ID, seq, bind_sit_seq))
    return Pro_Id, seqs, slice_seq, seq_labels


def write_list_to_txt(path, file_name, list_data):
    """write file"""
    f = open(path + file_name, "w")
    for line in list_data:
        f.write(str(line) + '\n')
    f.close()
    print("成功写入 %s" % (path + file_name))


def file_Connect(path1, file1, path2, file2, out_path, out_file_name):
    """connect file"""
    temp_list = []
    file1_list = read_file(path1, file1)
    file2_list = read_file(path2, file2)
    temp_list = file1_list + file2_list
    write_list_to_txt(out_path, out_file_name, temp_list)
    return temp_list


def words_feature_extr(feature_extr_list):
    """word2vec  """
    words_feature_lis = []
    for i in range(len(feature_extr_list)):
        words_feature_lis.append(word2vec.wv[str(feature_extr_list[i])])
    return words_feature_lis


def Letter_to_word_vector(path, file_name, fea_dim, len_seq):
    """sequence to vector"""
    list_fea = []
    words_feature = "ACDEFGHIKLMNPQRSTVWY"
    X_fea = X_fea = np.zeros((1, fea_dim))
    a = np.array(words_feature_extr(words_feature))
    b = np.insert(a, 1, X_fea, axis=0)
    b = np.insert(b, 9, X_fea, axis=0)
    b = np.insert(b, 14, X_fea, axis=0)
    b = np.insert(b, 20, X_fea, axis=0)
    b = np.insert(b, 23, X_fea, axis=0)
    b = np.insert(b, 25, X_fea, axis=0)
    file = read_file(path, file_name)
    for i in range(int(len(file))):
        for j in range(len_seq):
            index = ord(file[i][j]) - ord('A')
            list_fea.append(b[index])
    return list_fea


def split_list_by_n(list_collection, n):
    """
    :param list_collection:
    :param n:
    :return:返回的结果为评分后的每份可迭代对象
    """
    list_str = []
    for i in range(len(list_collection)):
        list_seq = []
        for j in range(0, len_seq - n + 1):
            list_seq.append(list_collection[i][j: j + n])
            list_str.append(list_seq)
    return list_str


def split_list_head_end_n(list_collection, n):
    """
    将集合均分，每份n个元素
    :param list_collection:
    :param n:
    :return:返回的结果为评分后的每份可迭代对象
    """
    list_str = []
    for i in range(len(list_collection)):
        list_seq = []
        list_seq2 = []
        list_seq.append(list_collection[i][0:n])
        list_str.append(list_seq)
        list_seq2.append(list_collection[i][-n:])
        list_str.append(list_seq2)
    return list_str


def Sequence_sliding_segmentation(sequence):
    """Sequence slidding
    sequence
    return
    """
    return_list = []
    for i in range(len(sequence)):
        Single_temporary_sequence = []
        replace_X_seq = sequence[i].replace("X", "")
        if len(str(replace_X_seq)) != 0:
            Single_temporary_sequence.append(replace_X_seq[0:1])
            for j in range(0, len(replace_X_seq) - 2 + 1):
                Single_temporary_sequence.append(replace_X_seq[j: j + 2])
            Single_temporary_sequence.append(replace_X_seq[-1:])
            return_list.append(Single_temporary_sequence)
    return return_list


def Sequence_sliding_segmentation_single(sequence):
    return_list = []
    for i in range(len(sequence)):
        Single_temporary_sequence = []
        replace_X_seq = sequence[i].replace("X", "")
        if len(str(replace_X_seq)) != 0:
            for j in range(0, len(replace_X_seq)):
                Single_temporary_sequence.append(replace_X_seq[j: j + 1])
            return_list.append(Single_temporary_sequence)
    return return_list

len_seq = 39



def write_seq_to_txt(path, file_name, list_data):
    """write file"""
    f = open(path + file_name, "w")

    for line in list_data:
        str_var = ""
        for i in line:
            str_var += str(i)
        f.write(str(str_var) + '\n')
    f.close()
    print("成功写入 %s" % (path + file_name))


def shuffle_seq(data, perc):
    return_list = []
    for seq_i in range(len(data)):
        seq = list(data[seq_i])
        # print(seq)
        num = int(len(seq) * perc)
        index = [index_nun for index_nun in range(len(seq))]
        shuffle_num = random.sample(index, num)
        # print(shuffle_num)
        # print(shuffle_num[::-1])
        j = 0
        iter_list = [seq[i] for i in shuffle_num]
        c = iter_list[::-1]
        j = 0
        for i in shuffle_num:
            seq[i] = c[j]
            j = j + 1
        return_list.append(seq)
    return return_list